"""Preparação dos atributos para PCA e K-Means (Etapa 03).

Fluxo: indicadores (Etapa 02) → remoção explícita do aquecimento → divisão
cronológica 70/15/15 → ``StandardScaler`` ajustado **somente no treino** →
mesma transformação aplicada a validação e teste.

Uso::

    python -m regimes.preprocessing
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler

from regimes.config import load_config, resolve_path
from regimes.data import sha256_file
from regimes.features import feature_names
from regimes.split import (
    SPLITS,
    chronological_split,
    drop_warmup,
    split_summary,
    validate_split,
)

SCALED_SUFFIX = "_z"


def scaled_names(columns: list[str]) -> list[str]:
    return [f"{c}{SCALED_SUFFIX}" for c in columns]


def fit_scaler(df: pd.DataFrame, columns: list[str]) -> StandardScaler:
    """Ajusta o ``StandardScaler`` usando apenas as linhas do treino."""
    train = df.loc[df["split"] == "train", columns]
    if train.empty:
        raise ValueError("conjunto de treino vazio")
    return StandardScaler().fit(train)


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler, columns: list[str]) -> pd.DataFrame:
    """Aplica o scaler já ajustado a todas as linhas (somente ``transform``)."""
    if list(scaler.feature_names_in_) != list(columns):
        raise ValueError("ordem/nomes das colunas diferem dos usados no ajuste")
    z = scaler.transform(df[columns])
    out = df.copy()
    for i, name in enumerate(scaled_names(columns)):
        out[name] = z[:, i]
    return out


def check_scaling(df: pd.DataFrame, scaler: StandardScaler, columns: list[str],
                  atol: float = 1e-9) -> dict:
    """Verificações de integridade da transformação; levanta ``AssertionError`` se falhar."""
    zcols = scaled_names(columns)
    train = df[df["split"] == "train"]
    assert scaler.n_samples_seen_ == len(train), "scaler viu linhas fora do treino"
    assert np.allclose(scaler.mean_, train[columns].mean().to_numpy(), atol=atol), \
        "média do scaler difere da média do treino"
    assert np.allclose(scaler.scale_, train[columns].std(ddof=0).to_numpy(), atol=atol), \
        "escala do scaler difere do desvio do treino"
    assert np.allclose(train[zcols].mean().to_numpy(), 0.0, atol=1e-9), "treino padronizado sem média 0"
    assert np.allclose(train[zcols].std(ddof=0).to_numpy(), 1.0, atol=1e-9), "treino sem desvio 1"
    assert np.isfinite(df[zcols].to_numpy()).all(), "valores não finitos após a padronização"
    back = scaler.inverse_transform(df[zcols].to_numpy())
    assert np.allclose(back, df[columns].to_numpy(), rtol=1e-10, atol=1e-12), \
        "inverse_transform não reproduz os valores originais"
    by_split = df.groupby("split", observed=True)[zcols].agg(["mean", "std"])
    return {"by_split": by_split}


def prepare(feat: pd.DataFrame, columns: list[str], train: float, validation: float,
            test: float) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Executa aquecimento → split → scaler (treino) → transformação, com verificações."""
    clean, dropped = drop_warmup(feat, columns)
    df = chronological_split(clean, train, validation, test)
    validate_split(df)
    scaler = fit_scaler(df, columns)
    df = apply_scaler(df, scaler, columns)
    check_scaling(df, scaler, columns)
    return df, dropped, scaler


def run(cfg: dict) -> dict:
    processed = resolve_path(cfg["data"]["processed_dir"])
    feat_path = processed / "features.csv"
    if not feat_path.exists():
        raise FileNotFoundError(f"{feat_path} não existe; execute 'python -m regimes.eda'")
    feat = pd.read_csv(feat_path, parse_dates=["date"])
    cols = feature_names(cfg["features"]["window"])
    s = cfg["split"]
    df, dropped, scaler = prepare(feat, cols, s["train"], s["validation"], s["test"])
    checks = check_scaling(df, scaler, cols)

    out_cols = ["date", "split", "close", "volume", *cols, *scaled_names(cols)]
    out_path = processed / "dataset_etapa03.csv"
    df[out_cols].to_csv(out_path, index=False, date_format="%Y-%m-%d")

    art_dir = resolve_path(cfg["outputs"]["artifacts_dir"]) / "etapa03"
    art_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = art_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    summary = split_summary(df)
    rep_dir = resolve_path("reports") / "etapa03"
    rep_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(rep_dir / "split_summary.csv", date_format="%Y-%m-%d")
    raw_stats = df.groupby("split", observed=True)[cols].agg(["mean", "std", "min", "max"])
    raw_stats.to_csv(rep_dir / "raw_stats_by_split.csv")
    checks["by_split"].to_csv(rep_dir / "scaled_stats_by_split.csv")
    from regimes import plots

    fig_path = plots.split_overview(
        df, resolve_path(cfg["outputs"]["figures_dir"]) / "etapa03" / "fig07_divisao_temporal.png",
        f"Fonte: B3 (COTAHIST), BOVA11, {df['date'].min():%d/%m/%Y}–{df['date'].max():%d/%m/%Y}.")

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"file": feat_path.name, "sha256": sha256_file(feat_path)},
        "output": {"file": out_path.name, "sha256": sha256_file(out_path), "n_rows": int(len(df))},
        "warmup_dropped": {"n": int(len(dropped)),
                           "dates": [f"{dropped['date'].min():%Y-%m-%d}",
                                     f"{dropped['date'].max():%Y-%m-%d}"]},
        "split": {k: {"start": f"{summary.loc[k, 'start']:%Y-%m-%d}",
                      "end": f"{summary.loc[k, 'end']:%Y-%m-%d}", "n": int(summary.loc[k, "n"])}
                  for k in SPLITS},
        "features": cols,
        "scaler": {"file": scaler_path.name, "sha256": sha256_file(scaler_path),
                   "fitted_on": "train", "n_samples_seen": int(scaler.n_samples_seen_),
                   "mean": dict(zip(cols, scaler.mean_.tolist())),
                   "scale": dict(zip(cols, scaler.scale_.tolist())),
                   "sklearn": sklearn.__version__},
        "checks_passed": ["validate_split", "check_scaling"],
        "figure": fig_path.name,
    }
    (processed / "dataset_etapa03.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split temporal e padronização (Etapa 03)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_config(args.config) if args.config else load_config()
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
