"""Testes da Etapa 02: fórmulas dos indicadores, causalidade e estatísticas."""

import numpy as np
import pandas as pd
import pytest

from regimes.features import (
    build_features,
    distance_to_sma,
    feature_names,
    log_return,
    relative_volume,
    rolling_volatility,
    warmup_length,
)
from regimes.stats import covariance_correlation, descriptive_stats, extreme_analysis


def _series(n: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    volume = rng.integers(1_000, 5_000, n).astype(float)
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n),
                         "close": close, "volume": volume})


# --- fórmulas com valores calculados à mão ----------------------------------

def test_log_return_hand_values():
    r = log_return(pd.Series([100.0, 110.0, 99.0]))
    assert np.isnan(r[0])
    assert r[1] == pytest.approx(np.log(1.1))
    assert r[2] == pytest.approx(np.log(0.9))


def test_log_return_rejects_nonpositive():
    with pytest.raises(ValueError):
        log_return(pd.Series([100.0, 0.0]))


def test_rolling_volatility_hand_values():
    r = pd.Series([np.nan, 0.01, -0.01, 0.02])
    v = rolling_volatility(r, window=3)
    assert v.iloc[:3].isna().all()
    assert v.iloc[3] == pytest.approx(np.std([0.01, -0.01, 0.02], ddof=1))


def test_distance_to_sma_hand_values():
    c = pd.Series([10.0, 11.0, 12.0, 15.0])
    d = distance_to_sma(c, window=3)
    assert d.iloc[:2].isna().all()
    assert d.iloc[2] == pytest.approx(12 / 11 - 1)          # MM(10,11,12) = 11
    assert d.iloc[3] == pytest.approx(15 / (38 / 3) - 1)    # MM(11,12,15)


def test_relative_volume_uses_previous_window_only():
    v = pd.Series([100.0, 200.0, 300.0, 600.0])
    rv = relative_volume(v, window=3)
    assert rv.iloc[:3].isna().all()
    assert rv.iloc[3] == pytest.approx(np.log(600 / 200))    # média de 100, 200, 300


def test_relative_volume_nonpositive_becomes_nan():
    v = pd.Series([100.0, 100.0, 100.0, 0.0, 100.0])
    rv = relative_volume(v, window=3)
    assert np.isnan(rv.iloc[3])


def test_constant_series_gives_zero_indicators():
    df = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=30),
                       "close": 50.0, "volume": 1000.0})
    f = build_features(df, window=5).dropna()
    for c in feature_names(5):
        assert np.allclose(f[c], 0.0)


# --- aquecimento e estrutura -------------------------------------------------

def test_warmup_rows_are_exactly_window():
    df = _series(60)
    f = build_features(df, window=20)
    complete = f[feature_names(20)].notna().all(axis=1)
    assert (~complete).sum() == warmup_length(20) == 20
    assert complete.iloc[20:].all() and not complete.iloc[:20].any()


def test_build_features_rejects_duplicated_dates():
    df = _series(30)
    df.loc[5, "date"] = df.loc[4, "date"]
    with pytest.raises(ValueError):
        build_features(df, window=5)


# --- causalidade: dados futuros não alteram o passado ------------------------

@pytest.mark.parametrize("t", [25, 40, 58])
def test_features_invariant_to_future_data(t):
    df = _series(60)
    base = build_features(df, window=20)
    altered = df.copy()
    rng = np.random.default_rng(7)
    altered.loc[t + 1:, "close"] *= rng.uniform(0.5, 1.5, len(altered) - t - 1)
    altered.loc[t + 1:, "volume"] *= rng.uniform(0.1, 10, len(altered) - t - 1)
    alt = build_features(altered, window=20)
    cols = feature_names(20)
    pd.testing.assert_frame_equal(base.loc[:t, cols], alt.loc[:t, cols])
    assert not np.allclose(base.loc[t + 1:, "log_return"], alt.loc[t + 1:, "log_return"])


def test_features_on_truncated_series_match_full_series():
    df = _series(60)
    full = build_features(df, window=20)
    for t in (30, 45):
        trunc = build_features(df.iloc[: t + 1], window=20)
        pd.testing.assert_frame_equal(full.iloc[: t + 1], trunc)


# --- estatísticas ------------------------------------------------------------

def test_descriptive_stats_match_numpy():
    df = _series(80)
    f = build_features(df, window=20)
    s = descriptive_stats(f, ["log_return"]).loc["log_return"]
    x = f["log_return"].dropna().to_numpy()
    assert s["n"] == x.size
    assert s["mean"] == pytest.approx(x.mean())
    assert s["median"] == pytest.approx(np.median(x))
    assert s["var"] == pytest.approx(x.var(ddof=1))
    assert s["std"] == pytest.approx(x.std(ddof=1))
    assert (s["min"], s["max"]) == (pytest.approx(x.min()), pytest.approx(x.max()))


def test_covariance_correlation_consistent():
    f = build_features(_series(120), window=20)
    cols = feature_names(20)
    cc = covariance_correlation(f, cols)
    cov, corr = cc["covariance"].to_numpy(), cc["pearson"].to_numpy()
    sd = np.sqrt(np.diag(cov))
    assert np.allclose(corr, cov / np.outer(sd, sd))
    assert np.allclose(np.diag(corr), 1.0)
    assert cc["n"] == f[cols].dropna().shape[0]


def test_extreme_analysis_counts_outliers():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 1000)
    r[[100, 500]] = [0.2, -0.25]
    df = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=1000), "log_return": r})
    ext = extreme_analysis(df, "log_return", k=3.0, top=2)
    assert ext["count_beyond_k"] >= 2
    assert ext["expected_if_normal"] == pytest.approx(0.0026998 * 1000, rel=1e-3)
    assert set(ext["largest"]["log_return"].round(2)) == {0.2, -0.25}
