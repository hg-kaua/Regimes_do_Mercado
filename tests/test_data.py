"""Testes da Etapa 01 (sem rede): normalização, corte, parser B3, validações e snapshots."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regimes.data import (
    DataCollectionError,
    analysis_series,
    build_canonical,
    compare_snapshots,
    compare_sources,
    normalize_yfinance,
    parse_cotahist,
    read_snapshot,
    resolve_cutoff,
    save_snapshot,
    validate_ohlcv,
)

FIXTURES = Path(__file__).parent / "fixtures"
TZ = "America/Sao_Paulo"


def _yf_raw(dates: list[str], tz_offsets: list[str] | None = None) -> pd.DataFrame:
    """Resposta sintética no formato do yfinance (índice com fuso)."""
    n = len(dates)
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(TZ)
    close = np.linspace(100, 100 + n - 1, n)
    return pd.DataFrame({
        "Open": close - 0.5, "High": close + 1, "Low": close - 1, "Close": close,
        "Adj Close": close, "Volume": np.arange(1, n + 1) * 1000,
        "Dividends": 0.0, "Stock Splits": 0.0,
    }, index=idx.rename("Date"))


def _clean(n: int = 6, start: str = "2018-01-02") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n)
    return normalize_yfinance(_yf_raw([d.strftime("%Y-%m-%d") for d in dates]), TZ)


# --- normalização -----------------------------------------------------------

def test_normalize_keeps_local_date_during_brazilian_dst():
    # 2018-02-16 estava em horário de verão (UTC-2); o pregão não pode mudar de dia.
    raw = _yf_raw(["2018-02-16", "2018-02-19"])
    out = normalize_yfinance(raw, TZ)
    assert list(out["date"].dt.strftime("%Y-%m-%d")) == ["2018-02-16", "2018-02-19"]
    assert list(out.columns[:6]) == ["date", "open", "high", "low", "close", "adj_close"]
    assert out["volume"].dtype == "int64"


def test_normalize_from_csv_roundtrip_with_offsets(tmp_path):
    raw = _yf_raw(["2018-02-16", "2018-02-19"])
    path = tmp_path / "raw.csv"
    raw.reset_index().to_csv(path, index=False)  # grava "2018-02-16 00:00:00-02:00"
    out = normalize_yfinance(pd.read_csv(path), TZ)
    assert list(out["date"].dt.strftime("%Y-%m-%d")) == ["2018-02-16", "2018-02-19"]


def test_normalize_flattens_multiindex_columns():
    raw = _yf_raw(["2018-01-02", "2018-01-03"])
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["BOVA11.SA"]])
    out = normalize_yfinance(raw, TZ)
    assert "close" in out.columns and "adj_close" in out.columns


def test_normalize_rejects_naive_index():
    raw = _yf_raw(["2018-01-02"])
    raw.index = raw.index.tz_localize(None)
    with pytest.raises(ValueError):
        normalize_yfinance(raw, TZ)


# --- corte do último pregão --------------------------------------------------

@pytest.mark.parametrize("hour_local,expected", [(18, "2026-11-06"), (19, "2026-11-09")])
def test_resolve_cutoff_excludes_incomplete_session(hour_local, expected):
    dates = pd.to_datetime(["2026-11-05", "2026-11-06", "2026-11-09"])
    # 2026: Brasil sem horário de verão -> UTC-3
    now = datetime(2026, 11, 9, hour_local + 3, 0, tzinfo=timezone.utc)
    assert resolve_cutoff(dates, now, 19, TZ) == pd.Timestamp(expected)


def test_resolve_cutoff_requires_aware_datetime_and_data():
    with pytest.raises(ValueError):
        resolve_cutoff(["2026-11-06"], datetime(2026, 11, 9, 20), 19, TZ)
    with pytest.raises(DataCollectionError):
        resolve_cutoff(["2026-11-09"], datetime(2026, 11, 9, 12, tzinfo=timezone.utc), 19, TZ)


# --- parser COTAHIST -----------------------------------------------------------

def test_parse_cotahist_real_lines_only_spot_bova11():
    lines = (FIXTURES / "cotahist_sample.txt").read_bytes().splitlines(keepends=True)
    df = parse_cotahist(lines, "BOVA11", "010")
    assert len(df) == 2  # ignora cabeçalho, trailer, PETR4 e BOVA11T (termo)
    first = df.iloc[0]
    assert first["date"] == pd.Timestamp("2018-02-14")
    assert first["ticker"] == "BOVA11"
    assert (first["open"], first["high"], first["low"]) == (79.72, 80.86, 79.58)
    assert first["close"] == 80.60
    assert first["quote_factor"] == 1
    assert first["volume_qty"] > 0 and first["n_trades"] > 0
    assert df["date"].is_monotonic_increasing


def test_parse_cotahist_accepts_str_lines():
    lines = (FIXTURES / "cotahist_sample.txt").read_text(encoding="latin-1").splitlines()
    assert len(parse_cotahist(lines, "BOVA11", "010")) == 2
    assert len(parse_cotahist(lines, "BOVA11T", "030")) == 1


# --- validações --------------------------------------------------------------

def _checks(issues, severity=None):
    return {i["check"] for i in issues if severity is None or i["severity"] == severity}


def test_validate_clean_series_has_no_errors():
    df = _clean()
    issues = validate_ohlcv(df, "2018-01-01", f"{df['date'].max():%Y-%m-%d}",
                            extreme_log_return=0.08, max_gap_days=4)
    assert _checks(issues, "error") == set()


@pytest.mark.parametrize("defect,check", [
    ("duplicate", "V3"), ("unordered", "V3"), ("weekend", "V3"), ("nan_close", "V4"),
    ("nonpositive", "V5"), ("negative_volume", "V5"), ("before_start", "V2"),
])
def test_validate_detects_errors(defect, check):
    df = _clean()
    if defect == "duplicate":
        df.loc[2, "date"] = df.loc[1, "date"]
    elif defect == "unordered":
        df.loc[[1, 2], "date"] = df.loc[[2, 1], "date"].to_numpy()
    elif defect == "weekend":
        df.loc[5, "date"] = pd.Timestamp("2018-01-13")  # sábado
    elif defect == "nan_close":
        df.loc[3, "close"] = np.nan
    elif defect == "nonpositive":
        df.loc[3, "low"] = 0
    elif defect == "negative_volume":
        df.loc[3, "volume"] = -1
    elif defect == "before_start":
        df.loc[0, "date"] = pd.Timestamp("2017-12-29")
    issues = validate_ohlcv(df, "2018-01-01", None, extreme_log_return=0.08, max_gap_days=4)
    assert check in _checks(issues, "error")


def test_validate_flags_warnings_without_removing_rows():
    df = _clean(8)
    df.loc[2, "high"] = df.loc[2, "close"] - 0.1      # V5 high < close
    df.loc[4, ["open", "high", "low", "close", "volume"]] = \
        df.loc[3, ["open", "high", "low", "close", "volume"]].to_numpy()  # V10
    df.loc[6, "close"] = df.loc[5, "close"] * 1.2     # V11 extremo
    df.loc[7, "dividends"] = 0.5                      # V12
    n = len(df)
    issues = validate_ohlcv(df, "2018-01-01", None, extreme_log_return=0.08, max_gap_days=4)
    assert {"V5", "V10", "V11", "V12"} <= _checks(issues)
    assert len(df) == n
    v11 = next(i for i in issues if i["check"] == "V11")
    assert f"{df.loc[6, 'date']:%Y-%m-%d}" in v11["dates"]


def test_validate_gap_detection():
    df = _clean(4)
    df.loc[3, "date"] = df.loc[2, "date"] + pd.Timedelta(days=7)
    issues = validate_ohlcv(df, "2018-01-01", None, extreme_log_return=0.08, max_gap_days=4)
    assert "V9" in _checks(issues)


# --- conferência com a B3 ---------------------------------------------------

def _b3_from(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "date": df["date"], "ticker": "BOVA11", "especi": "CI",
        "open": df["open"].round(2), "high": df["high"].round(2), "low": df["low"].round(2),
        "close": df["close"].round(2), "avg_price": df["close"].round(2), "n_trades": 10,
        "volume_qty": df["volume"], "volume_brl": df["volume"] * df["close"], "quote_factor": 1,
    })


def test_compare_sources_calendar_price_and_volume():
    yf = _clean(6)
    b3 = _b3_from(yf)
    b3 = b3.drop(index=2)                         # pregão ausente na B3
    b3.loc[3, "close"] += 0.05                    # divergência de preço
    b3.loc[4, "volume_qty"] *= 2                  # divergência de volume
    _, issues, summary = compare_sources(yf, b3, tolerance=0.01)
    assert summary["only_yfinance"] == 1 and summary["only_b3"] == 0
    msgs = {(i["check"], i["message"].split()[0]) for i in issues}
    assert ("V6", "data") in msgs
    assert any(i["check"] == "V7" and "close" in i["message"] for i in issues)
    assert any(i["check"] == "V8" for i in issues)


def test_compare_sources_identical_has_no_issues():
    yf = _clean(5)
    _, issues, summary = compare_sources(yf, _b3_from(yf), tolerance=0.01)
    assert issues == []
    assert summary["volume_exact_match_share"] == 1.0


def test_build_canonical_flags_and_keeps_all_rows():
    yf = _clean(6)
    b3 = _b3_from(yf).drop(index=1)
    yf_only_b3 = yf.drop(index=3)                    # pregão presente só na B3
    yf_only_b3.loc[4, ["open", "high", "low"]] = yf_only_b3.loc[4, "close"]
    yf_only_b3.loc[4, "volume"] = 0                  # candle "congelado" do Yahoo
    out = build_canonical(yf_only_b3, b3, tolerance=0.01, extreme_log_return=0.08)
    assert len(out) == 6                             # união das datas, nada removido
    assert out["flag_not_in_b3"].sum() == 1
    assert out["flag_not_in_yfinance"].sum() == 1
    assert out["flag_zero_volume"].sum() == 1
    assert out["flag_yf_flat_candle"].sum() == 1
    assert out["flag_price_mismatch"].sum() == 1     # o candle congelado diverge da B3
    assert out["date"].is_monotonic_increasing


def test_analysis_series_selects_source_calendar():
    yf = _clean(6)
    b3 = _b3_from(yf).drop(index=1)
    canon = build_canonical(yf.drop(index=3), b3, tolerance=0.01, extreme_log_return=0.08)
    s_b3 = analysis_series(canon, "b3")
    s_yf = analysis_series(canon, "yfinance")
    assert list(s_b3.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert set(s_b3["date"]) == set(b3["date"])
    assert set(s_yf["date"]) == set(yf.drop(index=3)["date"])
    with pytest.raises(ValueError):
        analysis_series(canon, "outra")


# --- snapshots ----------------------------------------------------------------

def test_snapshot_hash_and_immutability(tmp_path):
    df = _clean(3)
    path = tmp_path / "snap.csv"
    meta = save_snapshot(df, path, {"source": "teste"})
    assert len(meta["sha256"]) == 64 and meta["n_rows"] == 3
    back, meta2 = read_snapshot(path)
    assert len(back) == 3 and meta2["sha256"] == meta["sha256"]
    with pytest.raises(FileExistsError):
        save_snapshot(df, path, {"source": "teste"})
    path.write_text(path.read_text() + "\n# alterado\n")
    with pytest.raises(DataCollectionError):
        read_snapshot(path)


def test_compare_snapshots_detects_revision():
    a = _clean(4)
    b = a.copy()
    assert compare_snapshots(a, b)["identical"]
    b.loc[2, "close"] += 0.01
    res = compare_snapshots(a, b)
    assert not res["identical"] and res["columns"]["close"]["n_diff"] == 1
