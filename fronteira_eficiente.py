"""Fronteira de Markowitz com soluções reais do PyPortfolioOpt, em memória.

Recebe apenas retornos semanais DECIMAIS do treinamento. A média aritmética
e a covariância amostral são multiplicadas por 52. Cada alvo minimiza a
variância com retorno >= alvo, soma 1 e pesos entre zero e o teto informado.
Não usa teste, taxa livre de risco, capital, CSV, suavização ou pesos fictícios.
"""
from numbers import Integral

import numpy as np
import pandas as pd

from analisar_carteira import SEMANAS_ANO
from otimizar_carteira import (
    validar_covariancia, validar_limites, validar_pesos, SOLVER, OPCOES_SOLVER,
)


def calcular_fronteira(treinamento, peso_maximo=0.40, quantidade=31):
    """Retorna pontos anuais decimais, pesos por alvo e falhas explícitas.

Falhas interiores permanecem como NaN para interromper a linha no gráfico;
somente os índices presentes em pesos podem ser selecionados na interface.
Se o intervalo de retornos colapsar, retorna só a mínima volatilidade.
"""
    if (not isinstance(treinamento, pd.DataFrame) or len(treinamento) < 2
            or treinamento.empty or treinamento.columns.has_duplicates
            or treinamento.index.has_duplicates or not treinamento.index.is_monotonic_increasing):
        raise ValueError("Treinamento inválido: use retornos semanais ordenados, sem datas ou ativos duplicados.")
    valores = treinamento.to_numpy(dtype=float)
    if not np.isfinite(valores).all() or (valores <= -1).any():
        raise ValueError("Retornos de treinamento devem ser decimais, finitos e maiores que -100%.")
    if isinstance(quantidade, bool) or not isinstance(quantidade, Integral) or not 2 <= quantidade <= 101:
        raise ValueError("Informe entre 2 e 101 pontos para a fronteira.")
    validar_limites(len(treinamento.columns), 0, peso_maximo)
    medias = treinamento.mean() * SEMANAS_ANO
    covariancia = treinamento.cov() * SEMANAS_ANO
    validar_covariancia(covariancia)
    if np.max(np.abs(covariancia.to_numpy())) <= 1e-16:
        raise ValueError("Covariância nula: não há variação de risco suficiente para traçar a fronteira.")
    try:
        import cvxpy as cp
        from pypfopt import EfficientFrontier
        from pypfopt.exceptions import OptimizationError
    except ImportError as erro:
        raise RuntimeError("PyPortfolioOpt ou suas dependências não estão disponíveis.") from erro
    if SOLVER not in cp.installed_solvers():
        raise RuntimeError(f"Solver {SOLVER} indisponível para calcular a fronteira.")

    def nova_fronteira():
        return EfficientFrontier(medias, covariancia, weight_bounds=(0, peso_maximo),
                                 solver=SOLVER, solver_options=OPCOES_SOLVER)

    def avaliar(fronteira):
        pesos = pd.Series(fronteira.weights, index=medias.index, dtype=float)
        validar_pesos(pesos, medias.index, minimo=0, maximo=peso_maximo)
        retorno = float(pesos @ medias)
        variancia = float(pesos @ covariancia @ pesos)
        # Tolerância apenas para arredondamento de uma matriz PSD singular.
        if not np.isfinite([retorno, variancia]).all() or variancia < -1e-12:
            raise ValueError("Solução não finita ou variância negativa.")
        return pesos, retorno, float(np.sqrt(max(0.0, variancia)))

    erros_solver = (OptimizationError, cp.error.SolverError, cp.error.DCPError, ValueError, TypeError)
    try:
        minima = nova_fronteira()
        minima.min_volatility()
        pesos_min, retorno_min, vol_min = avaliar(minima)
        maxima = nova_fronteira()
        # Objetivo linear público do PyPortfolioOpt: determina o extremo viável.
        maxima.convex_objective(lambda w: -(w @ medias.to_numpy()))
        _, retorno_max, _ = avaliar(maxima)
    except erros_solver as erro:
        raise RuntimeError(f"Não foi possível determinar os extremos da fronteira: {erro}") from erro
    tolerancia = 1e-8 * max(1.0, abs(retorno_min), abs(retorno_max))
    if retorno_max < retorno_min - tolerancia:
        raise RuntimeError("Extremos inconsistentes: retorno máximo inferior ao da mínima volatilidade.")
    alvos = ([retorno_min] if retorno_max - retorno_min <= tolerancia
             else np.linspace(retorno_min, retorno_max, quantidade))
    pontos, pesos_validos, falhas = [], {}, []
    ultimo_retorno, ultima_vol = retorno_min, vol_min
    for indice, alvo in enumerate(alvos):
        try:
            if indice == 0:
                pesos, retorno, vol = pesos_min, retorno_min, vol_min
            else:
                fronteira = nova_fronteira()
                # Formulação de efficient_return pela API pública. Evita a
                # comparação estrita com um segundo LP no limite superior.
                fronteira.add_constraint(lambda w, alvo=float(alvo): w @ medias.to_numpy() >= alvo)
                fronteira.min_volatility()
                pesos, retorno, vol = avaliar(fronteira)
            if abs(retorno - alvo) > tolerancia:
                raise ValueError("Retorno obtido não corresponde ao alvo dentro da tolerância numérica.")
            if retorno < ultimo_retorno - tolerancia or vol < ultima_vol - tolerancia:
                raise ValueError("Ponto não preserva a ordenação do ramo eficiente.")
            pontos.append((alvo, retorno, vol))
            pesos_validos[indice] = pesos
            ultimo_retorno, ultima_vol = retorno, vol
        except erros_solver as erro:
            pontos.append((alvo, np.nan, np.nan))
            falhas.append(f"Alvo {alvo:.6%}: {erro}")
    return {
        "pontos": pd.DataFrame(pontos, columns=["retorno_alvo", "retorno_esperado", "volatilidade"]),
        "pesos": pd.DataFrame.from_dict(pesos_validos, orient="index").rename_axis("ponto"),
        "falhas": falhas,
        "retorno_minimo": retorno_min, "retorno_maximo": retorno_max,
    }
