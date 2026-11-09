"""Leitura e validação da configuração do projeto."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"

_REQUIRED = {
    "asset": ["ticker", "b3_code", "b3_market_type"],
    "data": ["start", "interval", "timezone", "cutoff_hour", "yfinance", "raw_dir", "processed_dir"],
    "validation": ["price_tolerance", "extreme_log_return", "max_gap_days"],
    "features": ["price_column", "volume_column", "window", "columns"],
    "split": ["train", "validation", "test"],
}


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Lê o YAML e valida as chaves obrigatórias e as proporções do split."""
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    missing = [f"{sec}.{key}" for sec, keys in _REQUIRED.items() for key in keys
               if key not in (cfg.get(sec) or {})]
    if missing:
        raise ValueError(f"Configuração incompleta, faltam: {missing}")
    total = sum(cfg["split"][k] for k in ("train", "validation", "test"))
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Proporções do split somam {total}, esperado 1.0")
    return cfg


def resolve_path(relative: str | Path) -> Path:
    """Converte um caminho da configuração em caminho absoluto a partir da raiz."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p
