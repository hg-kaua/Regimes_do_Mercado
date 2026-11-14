"""Máximo Sharpe com PyPortfolioOpt: todas as entradas são anuais e decimais.

Retorno esperado = média aritmética semanal do treinamento × 52.
Covariância anual = covariância amostral semanal do treinamento × 52.
Sharpe estimado = (retorno esperado anual − taxa livre de risco anual) / vol anual.
A taxa é uma hipótese informada pelo usuário; não se consulta uma taxa vigente.
Não usa o teste, clean_weights, renormalização ou pesos substitutos.
"""
from numbers import Real

import numpy as np
import pandas as pd

from otimizar_carteira import (
    validar_covariancia, validar_limites, validar_pesos, SOLVER, OPCOES_SOLVER,
)


def validar_taxa_livre_risco(taxa):
    """Unidade decimal anual: 0.10 significa 10% ao ano, não 0,10%."""
    if isinstance(taxa, (bool, np.bool_)) or not isinstance(taxa, Real) or not np.isfinite(taxa) or taxa <= -1:
        raise ValueError("A taxa livre de risco deve ser um número decimal anual finito, maior que -100%.")
    return float(taxa)


def otimizar_maximo_sharpe(retornos_anuais, covariancia_anual, taxa_livre_risco, maximo=0.40, solver=SOLVER):
    """Maximiza Sharpe com excesso positivo e pesos entre zero e o teto informado."""
    taxa = validar_taxa_livre_risco(taxa_livre_risco)
    validar_covariancia(covariancia_anual)
    validar_limites(len(covariancia_anual), 0, maximo)
    if not isinstance(retornos_anuais, pd.Series) or not retornos_anuais.index.equals(covariancia_anual.index):
        raise ValueError("Retornos esperados e covariância precisam ter os mesmos ativos na mesma ordem.")
    if not np.isfinite(retornos_anuais.to_numpy(dtype=float)).all():
        raise ValueError("Há retornos esperados ausentes ou infinitos.")
    # Limite superior exato do retorno com teto uniforme: aloca o orçamento
    # disponível aos maiores retornos, apenas para verificar viabilidade.
    # Não gera uma carteira substituta nem usa outro método de otimização.
    restante, retorno_maximo = 1.0, 0.0
    for retorno in sorted(retornos_anuais, reverse=True):
        parcela = min(maximo, restante)
        retorno_maximo += parcela * retorno
        restante -= parcela
    if retorno_maximo <= taxa + 1e-12:
        raise ValueError("Nenhuma carteira admissível supera a taxa livre de risco com as estimativas do treinamento "
                         "e o limite informado. max_sharpe() requer excesso de retorno positivo.")
    try:
        import cvxpy as cp
        from pypfopt import EfficientFrontier
        from pypfopt.exceptions import OptimizationError
    except ImportError as erro:
        raise RuntimeError("PyPortfolioOpt ou suas dependências não estão disponíveis; confira requirements.txt.") from erro
    if solver not in cp.installed_solvers():
        raise RuntimeError(f"Solver {solver} indisponível. Nenhum solver alternativo foi usado.")
    try:
        fronteira = EfficientFrontier(
            expected_returns=retornos_anuais, cov_matrix=covariancia_anual,
            weight_bounds=(0, maximo), solver=solver,
            solver_options=OPCOES_SOLVER if solver == 'CLARABEL' else None,
        )
        fronteira.max_sharpe(risk_free_rate=taxa)
    except (OptimizationError, cp.error.SolverError, cp.error.DCPError, ValueError, TypeError) as erro:
        raise RuntimeError(f"Falha no Máximo Sharpe com {solver}: {erro}. Nenhum peso substituto foi gerado.") from erro
    pesos = pd.Series(fronteira.weights, index=fronteira.tickers, name='peso_maximo_sharpe_decimal')
    validar_pesos(pesos, covariancia_anual.index, minimo=0, maximo=maximo)
    variancia = float(pesos @ covariancia_anual @ pesos)
    excesso = float(pesos @ retornos_anuais) - taxa
    if variancia <= 1e-16 or excesso <= 0 or not np.isfinite(excesso / np.sqrt(variancia)):
        raise ValueError("Sharpe indefinido ou solução numericamente inválida: confira o risco e o excesso de retorno.")
    return pesos
