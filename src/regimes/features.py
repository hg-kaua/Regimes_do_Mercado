"""Indicadores causais por pregão (Etapa 02).

Todas as funções são puras e usam apenas observações até o pregão ``t``
(janelas móveis "para trás", sem centralização e sem ``shift`` negativo).

Indicadores (``w`` = janela, 20 pregões por padrão):

- ``log_return``    r_t  = ln(C_t / C_{t-1})
- ``volatility_w``  σ_t  = desvio-padrão amostral (ddof=1) de r_{t-w+1}, …, r_t
- ``dist_sma_w``    d_t  = C_t / MM_w(C)_t − 1, com MM_w(C)_t = média de C_{t-w+1}, …, C_t
- ``rel_volume_w``  v_t  = ln(V_t / média de V_{t-w}, …, V_{t-1})
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ("log_return", "volatility_{w}", "dist_sma_{w}", "rel_volume_{w}")


def feature_names(window: int = 20) -> list[str]:
    """Nomes das colunas de indicadores para a janela ``window``."""
    return [c.format(w=window) for c in FEATURE_COLUMNS]


def log_return(close: pd.Series) -> pd.Series:
    """Log-retorno diário; primeiro valor é NaN."""
    if (close <= 0).any():
        raise ValueError("preços devem ser positivos")
    return np.log(close / close.shift(1))


def rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Desvio-padrão amostral dos últimos ``window`` retornos (inclui t)."""
    return returns.rolling(window, min_periods=window).std(ddof=1)


def distance_to_sma(close: pd.Series, window: int = 20) -> pd.Series:
    """Distância relativa do fechamento à média móvel simples que inclui t."""
    sma = close.rolling(window, min_periods=window).mean()
    return close / sma - 1.0


def relative_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    """Log da razão entre o volume de t e a média dos ``window`` pregões anteriores.

    A média exclui t para comparar o pregão corrente com o histórico recente.
    Volume não positivo gera NaN (tratado como falha na Etapa 03, não removido aqui).
    """
    base = volume.rolling(window, min_periods=window).mean().shift(1)
    ratio = volume / base
    return np.log(ratio.where((volume > 0) & (base > 0)))


def build_features(series: pd.DataFrame, window: int = 20,
                   price_col: str = "close", volume_col: str = "volume") -> pd.DataFrame:
    """Tabela de indicadores alinhada por data (mantém NaN do aquecimento das janelas)."""
    df = series.sort_values("date").reset_index(drop=True)
    if df["date"].duplicated().any():
        raise ValueError("datas duplicadas na série de entrada")
    close, volume = df[price_col].astype(float), df[volume_col].astype(float)
    r = log_return(close)
    names = feature_names(window)
    out = pd.DataFrame({
        "date": df["date"],
        "close": close,
        "volume": volume,
        names[0]: r,
        names[1]: rolling_volatility(r, window),
        names[2]: distance_to_sma(close, window),
        names[3]: relative_volume(volume, window),
    })
    return out


def warmup_length(window: int = 20) -> int:
    """Número de linhas iniciais sem todos os indicadores definidos.

    A volatilidade precisa de ``window`` retornos (o primeiro retorno é da linha 1)
    e o volume relativo de ``window`` volumes anteriores: ambos começam na linha ``window``.
    """
    return window
