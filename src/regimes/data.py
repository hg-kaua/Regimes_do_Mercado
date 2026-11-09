"""Coleta, preservação, validação e conferência dos dados diários (Etapa 01).

Fonte principal: Yahoo Finance via ``yfinance``. Fonte de conferência: séries
históricas COTAHIST da B3. As funções de transformação e validação são puras
(recebem e devolvem DataFrames); as de I/O ficam isoladas e nunca sobrescrevem
snapshots brutos.

Uso::

    python -m regimes.data collect   # baixa yfinance + B3 e grava snapshots brutos
    python -m regimes.data build     # valida, confere com a B3 e gera a tabela canônica
    python -m regimes.data recheck   # nova consulta ao yfinance comparada ao snapshot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from regimes.config import load_config, resolve_path

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

YF_RENAME = {
    "Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close",
    "Adj Close": "adj_close", "Volume": "volume", "Dividends": "dividends",
    "Stock Splits": "stock_splits", "Capital Gains": "capital_gains",
}
CANONICAL_PRICE = ["open", "high", "low", "close", "adj_close"]
CANONICAL_COLUMNS = ["date", *CANONICAL_PRICE, "volume", "dividends", "stock_splits"]

# Layout COTAHIST (registro tipo 01): posições 1-indexadas e inclusivas.
COTAHIST_LAYOUT: dict[str, tuple[int, int]] = {
    "tipreg": (1, 2), "date": (3, 10), "codbdi": (11, 12), "codneg": (13, 24),
    "tpmerc": (25, 27), "especi": (40, 49), "open": (57, 69), "high": (70, 82),
    "low": (83, 95), "avg_price": (96, 108), "close": (109, 121), "n_trades": (148, 152),
    "volume_qty": (153, 170), "volume_brl": (171, 188), "quote_factor": (211, 217),
}
COTAHIST_IMPLIED_2_DECIMALS = ("open", "high", "low", "avg_price", "close", "volume_brl")
COTAHIST_INT_FIELDS = ("n_trades", "volume_qty", "quote_factor")
B3_COLUMNS = ["date", "ticker", "especi", "open", "high", "low", "close", "avg_price",
              "n_trades", "volume_qty", "volume_brl", "quote_factor"]


class DataCollectionError(RuntimeError):
    """Falha de coleta: nunca é contornada com dados sintéticos."""


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Hash SHA-256 do conteúdo do arquivo."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _issue(check: str, severity: str, message: str, dates: Iterable[Any] = ()) -> dict:
    """Registro padronizado de ocorrência de validação."""
    ds = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]
    return {"check": check, "severity": severity, "message": message,
            "n": len(ds), "dates": ds}


def _environment() -> dict[str, str]:
    import yfinance

    return {"python": platform.python_version(), "pandas": pd.__version__,
            "numpy": np.__version__, "yfinance": yfinance.__version__,
            "platform": platform.platform()}


# ---------------------------------------------------------------------------
# yfinance
# ---------------------------------------------------------------------------

def fetch_yfinance(ticker: str, start: str, end_exclusive: str, interval: str,
                   params: dict[str, Any]) -> pd.DataFrame:
    """Consulta o Yahoo Finance. Levanta ``DataCollectionError`` se vier vazio ou falhar."""
    import yfinance as yf

    try:  # yfinance >= 1.x: exceções visíveis em vez de DataFrame vazio silencioso
        yf.config.debug.hide_exceptions = False
    except AttributeError:
        pass
    try:
        raw = yf.Ticker(ticker).history(start=start, end=end_exclusive, interval=interval,
                                        **params)
    except Exception as exc:  # noqa: BLE001 - repassa qualquer falha da biblioteca
        raise DataCollectionError(f"Falha ao consultar yfinance para {ticker}: {exc}") from exc
    if raw is None or raw.empty:
        raise DataCollectionError(f"yfinance devolveu tabela vazia para {ticker}")
    return raw


def normalize_yfinance(raw: pd.DataFrame, tz: str = "America/Sao_Paulo") -> pd.DataFrame:
    """Padroniza a resposta do yfinance (vinda da API ou do CSV bruto).

    - achata colunas MultiIndex (nível 0);
    - converte o carimbo de tempo para o fuso local *antes* de extrair a data,
      evitando deslocar o dia em períodos de horário de verão;
    - renomeia para snake_case, ordena por data e tipa as colunas.
    """
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Date" in df.columns:
        stamps = pd.to_datetime(df.pop("Date"), utc=True)
        stamps = pd.DatetimeIndex(stamps)
    else:
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is None:
            raise ValueError("Índice sem fuso horário: data do pregão ambígua")
        stamps = idx.tz_convert("UTC")
    local = stamps.tz_convert(tz)
    df = df.reset_index(drop=True)
    df.insert(0, "date", local.tz_localize(None).normalize())
    df = df.rename(columns=lambda c: YF_RENAME.get(c, str(c).strip().lower().replace(" ", "_")))
    for col in df.columns.drop("date"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df and df["volume"].notna().all():
        df["volume"] = df["volume"].astype("int64")
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def resolve_cutoff(dates: Sequence[Any], collected_at: datetime, cutoff_hour: int,
                   tz: str = "America/Sao_Paulo") -> pd.Timestamp:
    """Último pregão completo na hora da coleta.

    O pregão da própria data da coleta só conta se a coleta ocorreu a partir de
    ``cutoff_hour`` (hora local); antes disso pode estar incompleto.
    """
    if collected_at.tzinfo is None:
        raise ValueError("collected_at precisa ter fuso horário")
    local = collected_at.astimezone(ZoneInfo(tz))
    today = pd.Timestamp(local.date())
    valid = [pd.Timestamp(d) for d in dates
             if pd.Timestamp(d) < today or (pd.Timestamp(d) == today and local.hour >= cutoff_hour)]
    if not valid:
        raise DataCollectionError("Nenhum pregão completo disponível até a data da coleta")
    return max(valid)


# ---------------------------------------------------------------------------
# B3 — COTAHIST
# ---------------------------------------------------------------------------

def _field(line: str, name: str) -> str:
    a, b = COTAHIST_LAYOUT[name]
    return line[a - 1:b]


def parse_cotahist_line(line: str) -> dict[str, Any]:
    """Converte uma linha de registro tipo 01 do COTAHIST em dicionário tipado."""
    rec: dict[str, Any] = {
        "date": pd.Timestamp(datetime.strptime(_field(line, "date"), "%Y%m%d")),
        "ticker": _field(line, "codneg").strip(),
        "tpmerc": _field(line, "tpmerc"),
        "codbdi": _field(line, "codbdi"),
        "especi": _field(line, "especi").strip(),
    }
    for name in COTAHIST_IMPLIED_2_DECIMALS:
        rec[name] = int(_field(line, name)) / 100.0
    for name in COTAHIST_INT_FIELDS:
        rec[name] = int(_field(line, name))
    return rec


def parse_cotahist(lines: Iterable[str | bytes], codneg: str, tpmerc: str = "010") -> pd.DataFrame:
    """Filtra e converte os registros de ``codneg`` no mercado ``tpmerc``.

    Linhas de cabeçalho (00) e trailer (99) são ignoradas. O filtro por posição
    evita confundir, p.ex., ``BOVA11`` (à vista) com ``BOVA11T`` (termo).
    """
    key = f"{codneg:<12}{tpmerc}"
    key_b = key.encode("latin-1")
    rows = []
    for line in lines:
        if isinstance(line, bytes):
            if line[:2] != b"01" or line[12:27] != key_b:
                continue
            line = line.decode("latin-1")
        elif line[:2] != "01" or line[12:27] != key:
            continue
        rows.append(parse_cotahist_line(line.rstrip("\r\n")))
    df = pd.DataFrame(rows, columns=[*B3_COLUMNS, "tpmerc", "codbdi"])
    return df.drop(columns=["tpmerc", "codbdi"]).sort_values("date").reset_index(drop=True)


def iter_zip_lines(zip_path: str | Path) -> Iterator[bytes]:
    """Itera as linhas (bytes) do único arquivo TXT dentro do ZIP do COTAHIST."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if len(names) != 1:
            raise DataCollectionError(f"{zip_path}: esperado 1 arquivo no ZIP, há {names}")
        with zf.open(names[0]) as fh:
            yield from fh


def download_cotahist(year: int, url_template: str, dest_dir: Path, timeout: int = 300) -> Path:
    """Baixa o ZIP anual da B3 se ainda não existir localmente (nunca sobrescreve)."""
    import requests

    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"COTAHIST_A{year}.ZIP"
    if path.exists():
        return path
    url = url_template.format(year=year)
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise DataCollectionError(f"Falha ao baixar {url}: {exc}") from exc
    tmp = path.with_suffix(".part")
    tmp.write_bytes(resp.content)
    tmp.rename(path)
    return path


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

def save_snapshot(df: pd.DataFrame, path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Grava CSV + ``.meta.json`` com SHA-256. Recusa sobrescrever arquivos existentes."""
    meta_path = path.with_suffix(".meta.json")
    if path.exists() or meta_path.exists():
        raise FileExistsError(f"Snapshot já existe e é imutável: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, date_format="%Y-%m-%d" if "date" in df.columns else None)
    meta = {**metadata, "file": path.name, "n_rows": int(len(df)),
            "columns": [str(c) for c in df.columns], "sha256": sha256_file(path)}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str),
                         encoding="utf-8")
    return meta


def read_snapshot(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Lê snapshot e confere o hash registrado (V15)."""
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    actual = sha256_file(path)
    if actual != meta["sha256"]:
        raise DataCollectionError(f"V15: hash divergente em {path.name} "
                                  f"({actual} != {meta['sha256']})")
    return pd.read_csv(path), meta


# ---------------------------------------------------------------------------
# Validações (funções puras)
# ---------------------------------------------------------------------------

def validate_ohlcv(df: pd.DataFrame, start: str, end: str | None, *,
                   extreme_log_return: float, max_gap_days: int) -> list[dict]:
    """Executa V1–V5 e V9–V12 sobre a tabela normalizada do yfinance."""
    issues: list[dict] = []
    # V1 — esquema
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        issues.append(_issue("V1", "error", f"colunas ausentes: {missing}"))
        return issues
    extra = [c for c in df.columns if c not in CANONICAL_COLUMNS]
    if extra:
        issues.append(_issue("V1", "info", f"colunas adicionais preservadas: {extra}"))
    non_numeric = [c for c in CANONICAL_COLUMNS[1:] if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        issues.append(_issue("V1", "error", f"colunas não numéricas: {non_numeric}"))
    # V2 — não vazio / período
    if df.empty:
        issues.append(_issue("V2", "error", "tabela vazia"))
        return issues
    d = pd.to_datetime(df["date"])
    if d.min() < pd.Timestamp(start):
        issues.append(_issue("V2", "error", f"primeira data {d.min():%Y-%m-%d} anterior a {start}"))
    if end is not None and d.max() != pd.Timestamp(end):
        issues.append(_issue("V2", "error", f"última data {d.max():%Y-%m-%d} diferente do corte {end}"))
    # V3 — datas
    dup = d[d.duplicated(keep=False)]
    if len(dup):
        issues.append(_issue("V3", "error", "datas duplicadas", dup.unique()))
    if not d.is_monotonic_increasing:
        bad = d[d.diff() <= pd.Timedelta(0)]
        issues.append(_issue("V3", "error", "datas fora de ordem", bad))
    weekend = d[d.dt.day_of_week >= 5]
    if len(weekend):
        issues.append(_issue("V3", "error", "datas em fim de semana", weekend))
    # V4 — ausentes
    na = df[CANONICAL_COLUMNS[1:]].isna().sum()
    for col, n in na[na > 0].items():
        sev = "error" if col in ("open", "high", "low", "close") else "warning"
        issues.append(_issue("V4", sev, f"{n} valores ausentes em {col}", d[df[col].isna()]))
    # V5 — consistência OHLC
    eps = 1e-9
    lo_bad = df["low"] > df[["open", "close"]].min(axis=1) + eps
    hi_bad = df["high"] < df[["open", "close"]].max(axis=1) - eps
    if lo_bad.any():
        issues.append(_issue("V5", "warning", "low acima de min(open, close)", d[lo_bad]))
    if hi_bad.any():
        issues.append(_issue("V5", "warning", "high abaixo de max(open, close)", d[hi_bad]))
    nonpos = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if nonpos.any():
        issues.append(_issue("V5", "error", "preço não positivo", d[nonpos]))
    if (df["volume"] < 0).any():
        issues.append(_issue("V5", "error", "volume negativo", d[df["volume"] < 0]))
    zero_vol = df["volume"] == 0
    if zero_vol.any():
        issues.append(_issue("V5", "warning", "volume zero", d[zero_vol]))
    # V9 — lacunas
    gaps = d.diff().dt.days
    big = gaps > max_gap_days
    if big.any():
        issues.append(_issue("V9", "info", f"intervalo > {max_gap_days} dias corridos desde o "
                             "pregão anterior", d[big]))
    # V10 — dados congelados
    cols = ["open", "high", "low", "close", "volume"]
    frozen = (df[cols] == df[cols].shift(1)).all(axis=1)
    if frozen.any():
        issues.append(_issue("V10", "warning", "OHLCV idêntico ao pregão anterior", d[frozen]))
    # V11 — extremos
    lr = np.log(df["close"] / df["close"].shift(1))
    ext = lr.abs() > extreme_log_return
    if ext.any():
        issues.append(_issue("V11", "info", f"|log-retorno| > {extreme_log_return}", d[ext]))
    # V12 — eventos corporativos
    for col in ("dividends", "stock_splits", "capital_gains"):
        if col in df and (df[col].fillna(0) != 0).any():
            issues.append(_issue("V12", "warning", f"eventos em {col}", d[df[col].fillna(0) != 0]))
    ratio = df["adj_close"] / df["close"]
    if (ratio - 1).abs().max() > 1e-6:
        issues.append(_issue("V12", "warning", f"adj_close/close varia de {ratio.min():.6f} "
                             f"a {ratio.max():.6f}"))
    return issues


def compare_sources(yf_df: pd.DataFrame, b3_df: pd.DataFrame, tolerance: float
                    ) -> tuple[pd.DataFrame, list[dict], dict[str, Any]]:
    """Confere o yfinance com a B3 (V6–V8 e FATCOT).

    Retorna a tabela combinada (junção externa por data), as ocorrências e um
    resumo numérico.
    """
    issues: list[dict] = []
    b3 = b3_df.rename(columns={c: f"b3_{c}" for c in b3_df.columns if c != "date"})
    merged = yf_df.merge(b3, on="date", how="outer", indicator=True).sort_values("date")
    only_yf = merged.loc[merged["_merge"] == "left_only", "date"]
    only_b3 = merged.loc[merged["_merge"] == "right_only", "date"]
    if len(only_yf):
        issues.append(_issue("V6", "warning", "data no yfinance ausente na B3", only_yf))
    if len(only_b3):
        issues.append(_issue("V6", "warning", "pregão da B3 ausente no yfinance", only_b3))
    both = merged[merged["_merge"] == "both"]
    summary: dict[str, Any] = {"dates_both": int(len(both)), "only_yfinance": int(len(only_yf)),
                               "only_b3": int(len(only_b3)), "price_max_abs_diff": {}}
    for col in ("open", "high", "low", "close"):
        diff = (both[col] - both[f"b3_{col}"]).abs()
        summary["price_max_abs_diff"][col] = float(diff.max()) if len(diff) else None
        bad = diff > tolerance + 1e-9
        if bad.any():
            issues.append(_issue("V7", "warning", f"|{col} − b3_{col}| > R$ {tolerance}",
                                 both.loc[bad, "date"]))
    ratio = both["volume"] / both["b3_volume_qty"]
    summary["volume_ratio"] = {k: float(v) for k, v in ratio.describe(
        percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).items()}
    summary["volume_exact_match_share"] = float((both["volume"] == both["b3_volume_qty"]).mean())
    off = (ratio - 1).abs() > 0.01
    if off.any():
        issues.append(_issue("V8", "info", "volume yfinance difere > 1% de QUATOT (B3)",
                             both.loc[off, "date"]))
    fat = both["b3_quote_factor"] != 1
    if fat.any():
        issues.append(_issue("V12", "warning", "FATCOT da B3 diferente de 1", both.loc[fat, "date"]))
    return merged.drop(columns="_merge"), issues, summary


def build_canonical(yf_df: pd.DataFrame, b3_df: pd.DataFrame, *, tolerance: float,
                    extreme_log_return: float) -> pd.DataFrame:
    """Tabela canônica: união dos pregões das duas fontes, com flags (sem remover linhas).

    Colunas sem prefixo vêm do yfinance; colunas ``b3_*`` vêm do COTAHIST.
    ``flag_extreme_return`` usa o fechamento da B3 (série de análise, D-021).
    """
    keep = ["date", "open", "high", "low", "close", "volume_qty", "n_trades", "quote_factor"]
    b3 = b3_df[keep].rename(columns={c: f"b3_{c}" for c in keep if c != "date"})
    base_cols = [c for c in CANONICAL_COLUMNS if c in yf_df.columns]
    out = yf_df[base_cols].merge(b3, on="date", how="outer").sort_values("date")
    out = out.reset_index(drop=True)
    for c in ("b3_volume_qty", "b3_n_trades", "b3_quote_factor"):
        out[c] = out[c].astype("Int64")
    out["flag_not_in_b3"] = out["b3_close"].isna()
    out["flag_not_in_yfinance"] = out["close"].isna()
    diffs = [(out[c] - out[f"b3_{c}"]).abs() > tolerance + 1e-9 for c in ("open", "high", "low", "close")]
    out["flag_price_mismatch"] = pd.concat(diffs, axis=1).any(axis=1)
    out["flag_zero_volume"] = out["volume"] == 0
    flat = out[["open", "high", "low"]].eq(out["close"], axis=0).all(axis=1)
    out["flag_yf_flat_candle"] = flat & (out["volume"] == 0)
    b3_close = out["b3_close"].where(out["b3_close"].notna()).dropna()
    lr = np.log(b3_close / b3_close.shift(1))
    out["flag_extreme_return"] = (lr.abs() > extreme_log_return).reindex(out.index, fill_value=False)
    return out


def analysis_series(canonical: pd.DataFrame, source: str = "b3") -> pd.DataFrame:
    """Série OHLCV usada nos indicadores (D-021), restrita aos pregões da fonte escolhida.

    ``source='b3'``: preços e quantidade negociada oficiais (COTAHIST).
    ``source='yfinance'``: colunas do Yahoo (mantido para análise de sensibilidade).
    """
    if source == "b3":
        cols = {"b3_open": "open", "b3_high": "high", "b3_low": "low",
                "b3_close": "close", "b3_volume_qty": "volume"}
    elif source == "yfinance":
        cols = {c: c for c in ("open", "high", "low", "close", "volume")}
    else:
        raise ValueError(f"fonte desconhecida: {source}")
    out = canonical[["date", *cols]].rename(columns=cols)
    out = out.dropna(subset=["close"]).reset_index(drop=True)
    out["volume"] = out["volume"].astype("float64")
    return out


def compare_snapshots(a: pd.DataFrame, b: pd.DataFrame, atol: float = 1e-6) -> dict[str, Any]:
    """V14: compara duas coletas normalizadas (mesmo intervalo)."""
    m = a.merge(b, on="date", how="outer", suffixes=("_a", "_b"), indicator=True)
    res: dict[str, Any] = {"only_a": int((m["_merge"] == "left_only").sum()),
                           "only_b": int((m["_merge"] == "right_only").sum()), "columns": {}}
    both = m[m["_merge"] == "both"]
    for col in CANONICAL_COLUMNS[1:]:
        if f"{col}_a" in both and f"{col}_b" in both:
            diff = (both[f"{col}_a"] - both[f"{col}_b"]).abs()
            res["columns"][col] = {"n_diff": int((diff > atol).sum()),
                                   "max_abs_diff": float(diff.max())}
    res["identical"] = (res["only_a"] == 0 and res["only_b"] == 0
                        and all(v["n_diff"] == 0 for v in res["columns"].values()))
    return res


# ---------------------------------------------------------------------------
# Orquestração (I/O)
# ---------------------------------------------------------------------------

def _yf_dir(cfg: dict) -> Path:
    return resolve_path(cfg["data"]["raw_dir"]) / "yfinance"


def _b3_dir(cfg: dict) -> Path:
    return resolve_path(cfg["data"]["raw_dir"]) / "b3"


def _latest_snapshot(cfg: dict) -> Path:
    name = cfg["data"].get("yfinance_snapshot")
    if name:
        return _yf_dir(cfg) / name
    files = sorted(_yf_dir(cfg).glob(f"{cfg['asset']['ticker']}_*.csv"))
    if not files:
        raise DataCollectionError("Nenhum snapshot yfinance encontrado; execute 'collect'")
    return files[-1]


def collect(cfg: dict, now: datetime | None = None) -> dict[str, Any]:
    """Coleta yfinance e B3, gravando snapshots brutos imutáveis com metadados."""
    tz = cfg["data"]["timezone"]
    now = now or datetime.now(timezone.utc)
    local_now = now.astimezone(ZoneInfo(tz))
    start = cfg["data"]["start"]
    frozen_end = cfg["data"].get("end")
    end_excl = (pd.Timestamp(frozen_end) + timedelta(days=1) if frozen_end
                else pd.Timestamp(local_now.date()) + timedelta(days=1)).strftime("%Y-%m-%d")
    params = dict(cfg["data"]["yfinance"])
    ticker = cfg["asset"]["ticker"]

    raw = fetch_yfinance(ticker, start, end_excl, cfg["data"]["interval"], params)
    norm = normalize_yfinance(raw, tz)
    cutoff = (pd.Timestamp(frozen_end) if frozen_end
              else resolve_cutoff(norm["date"], now, cfg["data"]["cutoff_hour"], tz))
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    yf_path = _yf_dir(cfg) / f"{ticker}_{cfg['data']['interval']}_{start}_{cutoff:%Y-%m-%d}_{stamp}.csv"
    yf_meta = save_snapshot(raw.reset_index(), yf_path, {
        "source": "Yahoo Finance via yfinance (API não oficial)",
        "source_terms": "Uso sujeito aos termos do Yahoo; dados para fins acadêmicos.",
        "ticker": ticker, "interval": cfg["data"]["interval"],
        "call_params": {"start": start, "end": end_excl, "interval": cfg["data"]["interval"], **params},
        "requested_start": start, "requested_end_exclusive": end_excl,
        "rows_returned": int(len(raw)),
        "first_date_returned": f"{norm['date'].min():%Y-%m-%d}",
        "last_date_returned": f"{norm['date'].max():%Y-%m-%d}",
        "effective_end": f"{cutoff:%Y-%m-%d}", "cutoff_rule": (
            f"pregão do dia da coleta só entra se coletado após {cfg['data']['cutoff_hour']}h "
            f"({tz})" if not frozen_end else "data final congelada na configuração"),
        "collected_at_utc": now.isoformat(), "collected_at_local": local_now.isoformat(),
        "timezone_index": str(getattr(raw.index, "tz", None)),
        "environment": _environment(),
    })

    # B3
    zips_dir = _b3_dir(cfg) / "zips"
    years = range(pd.Timestamp(start).year, cutoff.year + 1)
    zips = [download_cotahist(y, cfg["data"]["b3_url_template"], zips_dir) for y in years]
    frames = [parse_cotahist(iter_zip_lines(z), cfg["asset"]["b3_code"],
                             cfg["asset"]["b3_market_type"]) for z in zips]
    b3 = pd.concat(frames, ignore_index=True)
    b3 = b3[(b3["date"] >= pd.Timestamp(start)) & (b3["date"] <= cutoff)].reset_index(drop=True)
    b3_path = _b3_dir(cfg) / f"{cfg['asset']['b3_code']}_cotahist_{start}_{cutoff:%Y-%m-%d}_{stamp}.csv"
    b3_meta = save_snapshot(b3, b3_path, {
        "source": "B3 — Séries Históricas (COTAHIST), arquivos anuais",
        "filter": {"CODNEG": cfg["asset"]["b3_code"], "TPMERC": cfg["asset"]["b3_market_type"],
                   "date_range": [start, f"{cutoff:%Y-%m-%d}"]},
        "price_note": "preços não ajustados; campos com 2 casas decimais implícitas divididos por 100",
        "zip_files": [{"name": z.name, "url": cfg["data"]["b3_url_template"].format(year=y),
                       "bytes": z.stat().st_size, "sha256": sha256_file(z),
                       "file_mtime_utc": datetime.fromtimestamp(z.stat().st_mtime,
                                                                timezone.utc).isoformat()}
                      for y, z in zip(years, zips)],
        "extracted_at_utc": now.isoformat(), "environment": _environment(),
    })
    return {"yfinance": yf_meta, "b3": b3_meta}


def build(cfg: dict) -> dict[str, Any]:
    """Valida o snapshot, confere com a B3 e grava a tabela canônica + relatório."""
    yf_path = _latest_snapshot(cfg)
    raw, yf_meta = read_snapshot(yf_path)
    end = yf_meta["effective_end"]
    start = cfg["data"]["start"]
    tz = cfg["data"]["timezone"]
    norm = normalize_yfinance(raw, tz)
    excluded = norm[norm["date"] > pd.Timestamp(end)]
    norm = norm[norm["date"] <= pd.Timestamp(end)].reset_index(drop=True)

    b3_files = sorted(_b3_dir(cfg).glob(f"{cfg['asset']['b3_code']}_cotahist_{start}_{end}_*.csv"))
    if not b3_files:
        raise DataCollectionError(f"Extrato B3 para {start}..{end} não encontrado")
    b3, b3_meta = read_snapshot(b3_files[-1])
    b3["date"] = pd.to_datetime(b3["date"])

    v = cfg["validation"]
    issues = validate_ohlcv(norm, start, end, extreme_log_return=v["extreme_log_return"],
                            max_gap_days=v["max_gap_days"])
    if len(excluded):
        issues.append(_issue("V13", "info", "linhas após o corte excluídas da tabela canônica "
                             "(preservadas no bruto)", excluded["date"]))
    _, cmp_issues, cmp_summary = compare_sources(norm, b3, v["price_tolerance"])
    issues += cmp_issues
    canonical = build_canonical(norm, b3, tolerance=v["price_tolerance"],
                                extreme_log_return=v["extreme_log_return"])

    out_dir = resolve_path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    canon_path = out_dir / "bova11_daily.csv"
    canonical.to_csv(canon_path, index=False, date_format="%Y-%m-%d")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"yfinance": {"file": yf_path.name, "sha256": yf_meta["sha256"]},
                   "b3": {"file": b3_files[-1].name, "sha256": b3_meta["sha256"]}},
        "output": {"file": canon_path.name, "sha256": sha256_file(canon_path),
                   "n_rows": int(len(canonical)),
                   "first_date": f"{canonical['date'].min():%Y-%m-%d}",
                   "last_date": f"{canonical['date'].max():%Y-%m-%d}"},
        "b3_rows": int(len(b3)),
        "comparison": cmp_summary,
        "flags": {c: int(canonical[c].sum()) for c in canonical.columns if c.startswith("flag_")},
        "issues": issues,
        "errors": sum(i["severity"] == "error" for i in issues),
    }
    (out_dir / "quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return report


def recheck(cfg: dict) -> dict[str, Any]:
    """V14: consulta o yfinance novamente com o mesmo intervalo e compara ao snapshot."""
    yf_path = _latest_snapshot(cfg)
    raw, meta = read_snapshot(yf_path)
    tz = cfg["data"]["timezone"]
    end = pd.Timestamp(meta["effective_end"])
    a = normalize_yfinance(raw, tz)
    a = a[a["date"] <= end]
    p = meta["call_params"]
    params = {k: v for k, v in p.items() if k not in ("start", "end", "interval")}
    b = normalize_yfinance(fetch_yfinance(meta["ticker"], p["start"], p["end"], p["interval"],
                                          params), tz)
    b = b[b["date"] <= end]
    res = compare_snapshots(a, b)
    res.update({"snapshot": yf_path.name, "rechecked_at_utc": datetime.now(timezone.utc).isoformat()})
    out = resolve_path(cfg["data"]["processed_dir"]) / "reproducibility_check.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return res


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Coleta e validação de dados (Etapa 01)")
    parser.add_argument("command", choices=["collect", "build", "recheck"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # console do Windows (cp1252)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_config(args.config) if args.config else load_config()
    if args.command == "collect":
        res = collect(cfg)
        print(json.dumps({k: {"file": v["file"], "n_rows": v["n_rows"], "sha256": v["sha256"]}
                          for k, v in res.items()}, indent=2))
    elif args.command == "build":
        rep = build(cfg)
        print(json.dumps({k: rep[k] for k in ("output", "flags", "errors")}, indent=2))
        for i in rep["issues"]:
            print(f"[{i['severity']:7}] {i['check']}: {i['message']} (n={i['n']})")
        return 1 if rep["errors"] else 0
    else:
        print(json.dumps(recheck(cfg), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
