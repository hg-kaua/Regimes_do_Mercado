"""Figuras do relatório (matplotlib, backend sem interface gráfica)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"  # azul
SERIES_2 = "#eb6834"  # laranja
NEUTRAL = "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("div", [SERIES_2, NEUTRAL, SERIES_1])

LABELS = {
    "log_return": "Log-retorno diário",
    "volatility_20": "Volatilidade 20 pregões (desvio do log-retorno)",
    "dist_sma_20": "Distância à MM20 (C/MM20 − 1)",
    "rel_volume_20": "Volume relativo: ln(V / média dos 20 anteriores)",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
        "legend.frameon": False, "legend.fontsize": 8, "font.size": 9,
    })


def _finish(fig, path: Path, source: str) -> Path:
    # Abaixo de toda a área dos eixos: bbox_inches="tight" inclui o texto sem sobrepor rótulos.
    fig.text(0.01, -0.04, source, fontsize=7, color=TEXT_2, ha="left", va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _dates(ax) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def price_volume(feat: pd.DataFrame, path: Path, source: str) -> Path:
    """Fechamento com MM20 (painel superior) e volume (painel inferior, eixo próprio)."""
    _style()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.2]})
    sma = feat["close"].rolling(20, min_periods=20).mean()
    a1.plot(feat["date"], feat["close"], color=SERIES_1, lw=1.0, label="Fechamento")
    a1.plot(feat["date"], sma, color=SERIES_2, lw=1.0, label="Média móvel 20")
    a1.set_ylabel("R$ por cota")
    a1.set_title("BOVA11 — fechamento diário e média móvel de 20 pregões")
    a1.legend(loc="upper left")
    a2.plot(feat["date"], feat["volume"] / 1e6, color=SERIES_1, lw=0.6)
    a2.set_ylabel("Volume (milhões de cotas)")
    _dates(a2)
    return _finish(fig, path, source)


def returns_distribution(feat: pd.DataFrame, path: Path, source: str) -> Path:
    """Histograma do log-retorno com normal de mesma média/desvio e gráfico Q-Q."""
    from scipy import stats as st

    _style()
    r = feat["log_return"].dropna()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.hist(r, bins=120, density=True, color=SERIES_1, alpha=0.85, edgecolor=SURFACE, lw=0.3,
            label="Observado")
    x = np.linspace(r.min(), r.max(), 400)
    a1.plot(x, st.norm.pdf(x, r.mean(), r.std(ddof=1)), color=SERIES_2, lw=1.6,
            label="Normal (mesma média e desvio)")
    a1.set_title("Distribuição dos log-retornos diários")
    a1.set_xlabel("Log-retorno")
    a1.set_ylabel("Densidade")
    a1.legend(loc="upper left")
    (osm, osr), (slope, intercept, _) = st.probplot(r, dist="norm")
    a2.scatter(osm, osr, s=8, color=SERIES_1, alpha=0.7, linewidths=0)
    a2.plot(osm, slope * osm + intercept, color=SERIES_2, lw=1.4, label="Referência normal")
    a2.set_title("Gráfico Q-Q contra a normal")
    a2.set_xlabel("Quantis teóricos (normal padrão)")
    a2.set_ylabel("Quantis observados")
    a2.legend(loc="upper left")
    return _finish(fig, path, source)


def volatility(feat: pd.DataFrame, path: Path, source: str) -> Path:
    """Evolução da volatilidade de 20 pregões (unidade diária)."""
    _style()
    fig, ax = plt.subplots(figsize=(11, 3.8))
    v = feat["volatility_20"]
    ax.plot(feat["date"], v, color=SERIES_1, lw=1.0, label="Volatilidade 20 pregões")
    med = v.median()
    ax.axhline(med, color=TEXT_2, lw=0.8, ls="--", label=f"Mediana do período ({med:.4f})")
    ax.legend(loc="upper right")
    ax.set_title("Volatilidade móvel de 20 pregões (desvio-padrão do log-retorno diário)")
    ax.set_ylabel("Desvio-padrão diário")
    _dates(ax)
    return _finish(fig, path, source)


def correlation_matrix(corr: pd.DataFrame, path: Path, source: str, title: str) -> Path:
    """Mapa de calor divergente (−1 a 1, cinza em 0) com valores anotados."""
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.grid(False)
    im = ax.imshow(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    labels = [c.replace("_", " ") for c in corr.columns]
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=9, color=TEXT)
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=TEXT_2)
    return _finish(fig, path, source)


def extremes(feat: pd.DataFrame, extreme: dict, path: Path, source: str) -> Path:
    """Log-retornos com faixas de ±k desvios; rotula o maior e o menor de cada ano com |z| > k."""
    _style()
    fig, ax = plt.subplots(figsize=(11, 4))
    r = feat[["date", "log_return"]].dropna()
    mu, sd, k = extreme["mean"], extreme["std"], extreme["k"]
    ax.plot(r["date"], r["log_return"], color=SERIES_1, lw=0.6)
    for s in (-1, 1):
        ax.axhline(mu + s * k * sd, color=TEXT_2, lw=0.8, ls="--")
    ax.annotate(f"±{k:g} desvios", (r["date"].iloc[-1], mu + k * sd), xytext=(0, 3),
                textcoords="offset points", fontsize=8, color=TEXT_2, ha="right")
    out = r[((r["log_return"] - mu) / sd).abs() > k]
    ax.scatter(out["date"], out["log_return"], s=22, color=SERIES_2, zorder=3,
               edgecolors=SURFACE, linewidths=1.2, label=f"|z| > {k:g} (n = {len(out)})")
    picks = pd.concat([out.loc[g["log_return"].agg(["idxmin", "idxmax"])]
                       for _, g in out.groupby(out["date"].dt.year)]).drop_duplicates()
    for _, row in picks.iterrows():
        if abs(row["log_return"] - mu) < 0.06:  # rotula só os mais afastados
            continue
        ax.annotate(f"{row['date']:%d/%m/%Y}", (row["date"], row["log_return"]),
                    xytext=(6, -3 if row["log_return"] < 0 else 2), textcoords="offset points",
                    fontsize=7, color=TEXT)
    ax.set_title("Log-retornos diários e valores extremos")
    ax.set_ylabel("Log-retorno")
    ax.legend(loc="lower left")
    _dates(ax)
    return _finish(fig, path, source)


SPLIT_COLORS = {"train": SERIES_1, "validation": SERIES_2, "test": "#1baf7a"}
SPLIT_LABELS = {"train": "Treino", "validation": "Validação", "test": "Teste"}


def split_overview(df: pd.DataFrame, path: Path, source: str) -> Path:
    """Fechamento colorido por conjunto (treino/validação/teste) com rótulos diretos."""
    _style()
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for name, color in SPLIT_COLORS.items():
        part = df[df["split"] == name]
        n = len(part)
        ax.plot(part["date"], part["close"], color=color, lw=1.0,
                label=f"{SPLIT_LABELS[name]} ({part['date'].min():%d/%m/%Y}–"
                      f"{part['date'].max():%d/%m/%Y}, n = {n})")
        ax.axvline(part["date"].min(), color=GRID, lw=1.0, zorder=0)
    ax.set_title("Divisão cronológica 70/15/15 (após remover o aquecimento das janelas)")
    ax.set_ylabel("R$ por cota")
    ax.legend(loc="upper left")
    _dates(ax)
    return _finish(fig, path, source)


def indicators_panel(feat: pd.DataFrame, columns: list[str], path: Path, source: str) -> Path:
    """Pequenos múltiplos: um painel por indicador, eixo de tempo compartilhado."""
    _style()
    fig, axes = plt.subplots(len(columns), 1, figsize=(11, 2.0 * len(columns)), sharex=True)
    for ax, c in zip(axes, columns, strict=True):
        ax.plot(feat["date"], feat[c], color=SERIES_1, lw=0.6)
        ax.axhline(0, color=TEXT_2, lw=0.6)
        ax.set_title(LABELS.get(c, c), loc="left", fontsize=9)
    _dates(axes[-1])
    fig.suptitle("Indicadores por pregão (calculados só com dados até t)", fontweight="bold",
                 color=TEXT, fontsize=11)
    fig.tight_layout()
    return _finish(fig, path, source)
