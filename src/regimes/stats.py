"""Estatísticas descritivas e análise de valores extremos (Etapa 02). Funções puras."""

from __future__ import annotations

from math import erf, sqrt

import numpy as np
import pandas as pd


def descriptive_stats(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Média, mediana, variância, desvio-padrão, mínimo, máximo, assimetria e curtose.

    Cada coluna usa seus próprios valores não ausentes. Variância e desvio são
    amostrais (ddof=1); curtose é o excesso em relação à normal.
    """
    rows = {}
    for c in columns:
        x = df[c].dropna()
        rows[c] = {
            "n": int(x.size), "mean": x.mean(), "median": x.median(), "var": x.var(ddof=1),
            "std": x.std(ddof=1), "min": x.min(), "max": x.max(),
            "skew": x.skew(), "excess_kurtosis": x.kurt(),
            "date_min": df.loc[x.idxmin(), "date"], "date_max": df.loc[x.idxmax(), "date"],
        }
    return pd.DataFrame(rows).T


def covariance_correlation(df: pd.DataFrame, columns: list[str]) -> dict[str, pd.DataFrame]:
    """Covariância e correlações de Pearson e Spearman nas linhas completas."""
    x = df[columns].dropna()
    return {"covariance": x.cov(ddof=1), "pearson": x.corr("pearson"),
            "spearman": x.corr("spearman"), "n": len(x)}


def _normal_two_sided_tail(k: float) -> float:
    """P(|Z| > k) para Z normal padrão."""
    return 1.0 - erf(k / sqrt(2.0))


def extreme_analysis(df: pd.DataFrame, column: str = "log_return", k: float = 3.0,
                     top: int = 10) -> dict:
    """Frequência de |z| > k comparada à normal e os ``top`` maiores |valores|.

    O z-score usa média e desvio da amostra inteira: é uma descrição, não um
    parâmetro de modelo (nenhum ajuste é reaproveitado nas etapas seguintes).
    """
    x = df[["date", column]].dropna()
    mu, sd = x[column].mean(), x[column].std(ddof=1)
    z = (x[column] - mu) / sd
    beyond = x[z.abs() > k]
    expected = _normal_two_sided_tail(k) * len(x)
    order = x[column].abs().sort_values(ascending=False).index[:top]
    largest = x.loc[order].assign(z=z.loc[order])
    return {"n": int(len(x)), "k": k, "mean": float(mu), "std": float(sd),
            "count_beyond_k": int(len(beyond)), "expected_if_normal": float(expected),
            "ratio_observed_expected": float(len(beyond) / expected) if expected else np.nan,
            "dates_beyond_k": beyond["date"].dt.strftime("%Y-%m-%d").tolist(),
            "largest": largest}


def yearly_summary(df: pd.DataFrame, column: str = "log_return") -> pd.DataFrame:
    """Retorno acumulado (soma de log-retornos), desvio e nº de pregões por ano."""
    x = df[["date", column]].dropna()
    g = x.groupby(x["date"].dt.year)[column]
    return pd.DataFrame({"n": g.size(), "sum_log_return": g.sum(), "std": g.std(ddof=1)})
