"""Etapa 02: indicadores, estatísticas descritivas e figuras.

Uso::

    python -m regimes.eda
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from regimes import plots
from regimes.config import load_config, resolve_path
from regimes.data import analysis_series, sha256_file
from regimes.features import build_features, feature_names
from regimes.stats import (
    covariance_correlation,
    descriptive_stats,
    extreme_analysis,
    yearly_summary,
)


def load_canonical(cfg: dict) -> tuple[pd.DataFrame, Path]:
    """Lê a tabela canônica da Etapa 01."""
    path = resolve_path(cfg["data"]["processed_dir"]) / "bova11_daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe; execute 'python -m regimes.data build'")
    return pd.read_csv(path, parse_dates=["date"]), path


def run(cfg: dict) -> dict:
    canonical, canon_path = load_canonical(cfg)
    fc = cfg["features"]
    series = analysis_series(canonical, fc["source"])
    feat = build_features(series, fc["window"], fc["price_column"], fc["volume_column"])
    cols = feature_names(fc["window"])

    processed = resolve_path(cfg["data"]["processed_dir"])
    feat_path = processed / "features.csv"
    feat.to_csv(feat_path, index=False, date_format="%Y-%m-%d")

    rep_dir = resolve_path("reports") / "etapa02"
    fig_dir = resolve_path(cfg["outputs"]["figures_dir"]) / "etapa02"
    rep_dir.mkdir(parents=True, exist_ok=True)

    desc_ind = descriptive_stats(feat, cols)
    desc_px = descriptive_stats(feat, ["close", "volume"])
    cc = covariance_correlation(feat, cols)
    ext = extreme_analysis(feat, "log_return", k=3.0, top=10)
    yearly = yearly_summary(feat)

    pd.concat([desc_px, desc_ind]).to_csv(rep_dir / "descriptive_stats.csv")
    cc["covariance"].to_csv(rep_dir / "covariance.csv")
    cc["pearson"].to_csv(rep_dir / "correlation_pearson.csv")
    cc["spearman"].to_csv(rep_dir / "correlation_spearman.csv")
    ext["largest"].to_csv(rep_dir / "largest_returns.csv", index=False, date_format="%Y-%m-%d")
    yearly.to_csv(rep_dir / "yearly_summary.csv")

    src = (f"Fonte: B3 (COTAHIST), BOVA11, {feat['date'].min():%d/%m/%Y}–"
           f"{feat['date'].max():%d/%m/%Y}." if fc["source"] == "b3" else "Fonte: Yahoo Finance.")
    figs = [
        plots.price_volume(feat, fig_dir / "fig01_preco_volume.png", src),
        plots.returns_distribution(feat, fig_dir / "fig02_distribuicao_retornos.png", src),
        plots.volatility(feat, fig_dir / "fig03_volatilidade_20.png", src),
        plots.correlation_matrix(cc["pearson"], fig_dir / "fig04_correlacao_pearson.png", src,
                                 f"Correlação de Pearson (n = {cc['n']})"),
        plots.extremes(feat, ext, fig_dir / "fig05_extremos.png", src),
        plots.indicators_panel(feat, cols, fig_dir / "fig06_indicadores.png", src),
    ]

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": fc["source"], "window": fc["window"],
        "input": {"file": canon_path.name, "sha256": sha256_file(canon_path)},
        "features_file": {"file": feat_path.name, "sha256": sha256_file(feat_path),
                          "n_rows": int(len(feat)),
                          "complete_rows": int(feat[cols].notna().all(axis=1).sum())},
        "nan_per_column": {c: int(feat[c].isna().sum()) for c in cols},
        "first_complete_date": f"{feat.loc[feat[cols].notna().all(axis=1), 'date'].min():%Y-%m-%d}",
        "extremes": {k: v for k, v in ext.items() if k != "largest"},
        "figures": [p.name for p in figs],
    }
    (rep_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                     default=str), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Indicadores e análise estatística (Etapa 02)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_config(args.config) if args.config else load_config()
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
