"""Testes da Etapa 03: aquecimento, divisão cronológica, padronização e vazamento temporal."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from regimes.features import build_features, feature_names
from regimes.preprocessing import apply_scaler, check_scaling, fit_scaler, prepare, scaled_names
from regimes.split import (
    chronological_split,
    drop_warmup,
    split_sizes,
    split_summary,
    validate_split,
)

COLS = feature_names(20)


def _prices(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    volume = rng.integers(1_000_000, 9_000_000, n).astype(float)
    return pd.DataFrame({"date": pd.bdate_range("2018-01-02", periods=n),
                         "close": close, "volume": volume})


def _features(n: int = 400, seed: int = 42) -> pd.DataFrame:
    return build_features(_prices(n, seed), window=20)


# --- aquecimento -------------------------------------------------------------

def test_drop_warmup_removes_only_leading_block():
    feat = _features(100)
    clean, dropped = drop_warmup(feat, COLS)
    assert len(dropped) == 20 and len(clean) == 80
    assert clean[COLS].notna().all().all()
    assert clean["date"].iloc[0] == feat["date"].iloc[20]


def test_drop_warmup_rejects_nan_after_warmup():
    feat = _features(100)
    feat.loc[50, "volatility_20"] = np.nan
    with pytest.raises(ValueError, match="após o aquecimento"):
        drop_warmup(feat, COLS)


# --- divisão cronológica -------------------------------------------------------

def test_split_sizes_floor_and_remainder():
    assert split_sizes(2180, 0.70, 0.15, 0.15) == (1526, 327, 327)
    assert split_sizes(101, 0.70, 0.15, 0.15) == (70, 15, 16)
    with pytest.raises(ValueError):
        split_sizes(100, 0.7, 0.2, 0.2)
    with pytest.raises(ValueError):
        split_sizes(3, 0.70, 0.15, 0.15)


def test_chronological_split_is_contiguous_and_ordered():
    clean, _ = drop_warmup(_features(400), COLS)
    df = chronological_split(clean.sample(frac=1, random_state=1))  # entrada embaralhada
    validate_split(df)
    s = split_summary(df)
    assert list(s["n"]) == list(split_sizes(len(df), 0.70, 0.15, 0.15))
    assert s.loc["train", "end"] < s.loc["validation", "start"] < s.loc["test", "start"]
    assert df["date"].is_monotonic_increasing


def test_split_depends_only_on_dates_not_values():
    clean, _ = drop_warmup(_features(400), COLS)
    a = chronological_split(clean)
    other = clean.copy()
    other[COLS] = other[COLS] * -3 + 1
    b = chronological_split(other)
    assert (a["split"] == b["split"]).all()


def test_validate_split_detects_overlap_and_disorder():
    clean, _ = drop_warmup(_features(200), COLS)
    df = chronological_split(clean)
    bad = df.copy()
    i = bad.index[bad["split"] == "validation"][0]
    bad.loc[i, "split"] = "test"                      # bloco de teste dentro da validação
    with pytest.raises(AssertionError):
        validate_split(bad)
    dup = df.copy()
    dup.loc[5, "date"] = dup.loc[4, "date"]
    with pytest.raises(AssertionError):
        validate_split(dup)


# --- padronização: ajuste só no treino --------------------------------------------

def _split_df(n: int = 400) -> pd.DataFrame:
    clean, _ = drop_warmup(_features(n), COLS)
    return chronological_split(clean)


def test_scaler_parameters_equal_train_statistics():
    df = _split_df()
    sc = fit_scaler(df, COLS)
    train = df.loc[df["split"] == "train", COLS]
    assert sc.n_samples_seen_ == len(train)
    np.testing.assert_allclose(sc.mean_, train.mean().to_numpy())
    np.testing.assert_allclose(sc.scale_, train.std(ddof=0).to_numpy())


def test_validation_and_test_values_do_not_affect_scaler():
    df = _split_df()
    sc1 = fit_scaler(df, COLS)
    altered = df.copy()
    mask = altered["split"] != "train"
    altered.loc[mask, COLS] = altered.loc[mask, COLS] * 50 + 7
    sc2 = fit_scaler(altered, COLS)
    np.testing.assert_array_equal(sc1.mean_, sc2.mean_)
    np.testing.assert_array_equal(sc1.scale_, sc2.scale_)


def test_out_of_sample_transform_uses_train_parameters():
    df = _split_df()
    sc = fit_scaler(df, COLS)
    out = apply_scaler(df, sc, COLS)
    train = df.loc[df["split"] == "train", COLS]
    test = df.loc[df["split"] == "test", COLS]
    expected = (test - train.mean()) / train.std(ddof=0)
    np.testing.assert_allclose(out.loc[test.index, scaled_names(COLS)].to_numpy(),
                               expected.to_numpy(), rtol=1e-10)
    # validação/teste NÃO ficam com média 0 e desvio 1 (não houve fit nesses conjuntos)
    z_test = out.loc[test.index, scaled_names(COLS)]
    assert not np.allclose(z_test.mean().to_numpy(), 0.0, atol=1e-6)


def test_check_scaling_rejects_scaler_fitted_on_all_rows():
    df = _split_df()
    leaky = StandardScaler().fit(df[COLS])            # vazamento: ajuste com todas as linhas
    out = apply_scaler(df, leaky, COLS)
    with pytest.raises(AssertionError):
        check_scaling(out, leaky, COLS)


def test_apply_scaler_rejects_different_column_order():
    df = _split_df()
    sc = fit_scaler(df, COLS)
    with pytest.raises(ValueError):
        apply_scaler(df, sc, list(reversed(COLS)))


def test_prepare_end_to_end_checks_pass_and_preserve_dates():
    feat = _features(400)
    df, dropped, sc = prepare(feat, COLS, 0.70, 0.15, 0.15)
    assert len(df) + len(dropped) == len(feat)
    assert set(df["date"]) | set(dropped["date"]) == set(feat["date"])
    assert {"date", "split", *COLS, *scaled_names(COLS)} <= set(df.columns)
    check_scaling(df, sc, COLS)


def test_pipeline_invariant_to_changes_after_training_period():
    """Alterar preços/volumes do período de teste não muda nada no treino e na validação."""
    prices = _prices(400)
    df_a, _, sc_a = prepare(build_features(prices, 20), COLS, 0.70, 0.15, 0.15)
    test_start = df_a.loc[df_a["split"] == "test", "date"].min()
    altered = prices.copy()
    m = altered["date"] >= test_start
    rng = np.random.default_rng(3)
    altered.loc[m, "close"] *= rng.uniform(0.7, 1.3, m.sum())
    altered.loc[m, "volume"] *= rng.uniform(0.2, 5.0, m.sum())
    df_b, _, sc_b = prepare(build_features(altered, 20), COLS, 0.70, 0.15, 0.15)
    np.testing.assert_array_equal(sc_a.mean_, sc_b.mean_)
    np.testing.assert_array_equal(sc_a.scale_, sc_b.scale_)
    keep = df_a["split"] != "test"
    zc = scaled_names(COLS)
    pd.testing.assert_frame_equal(df_a.loc[keep, ["date", *zc]], df_b.loc[keep, ["date", *zc]])
    assert not np.allclose(df_a.loc[~keep, zc].to_numpy(), df_b.loc[~keep, zc].to_numpy())
