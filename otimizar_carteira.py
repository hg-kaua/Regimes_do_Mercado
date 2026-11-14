"""Etapa 03: mínima volatilidade, sem backtesting e sem alterações na interface.

Execute: python otimizar_carteira.py
Testes locais: python otimizar_carteira.py --testar

Reutiliza os dados e as funções da Etapa 02, conferindo seus quatro CSVs antes
da otimização. Somente o treinamento define a covariância e os pesos.
A covariância semanal, em retorno decimal², é multiplicada por 52 UMA vez
para o EfficientFrontier. A média histórica serve apenas para a comparação.

Saídas em dados/carteiras/: pesos_otimizados.csv (pesos decimais) e
resumo_otimizacao.csv (médias e volatilidades decimais). A média anualizada
é aritmética, não um retorno realizado. Arredondamento ocorre só no terminal.
Não usa clean_weights(), não renormaliza e não substitui falhas por outros pesos.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from analisar_carteira import (
    COLUNAS_RESUMO, SEMANAS_ANO, analisar_retornos, ler_dados, salvar_resultados,
)


PASTA_DADOS = Path(__file__).resolve().parent / "dados" / "carteiras"
PESO_MINIMO = 0.0
PESO_MAXIMO = 0.40
SOLVER = "CLARABEL"
# Tolerâncias menores mantêm a solução estável ao mudar a escala da covariância.
OPCOES_SOLVER = {"tol_gap_abs": 1e-11, "tol_gap_rel": 1e-11, "tol_feas": 1e-11}
TOLERANCIA_PESOS = 1e-7
TOLERANCIA_RISCO = 1e-8


def carregar_resultados(pasta=PASTA_DADOS):
    """Confere que os CSVs da Etapa 02 correspondem aos dados e ao treino atuais."""
    pasta = Path(pasta)
    resultados = analisar_retornos(ler_dados(pasta))
    for nome, esperado in resultados.items():
        caminho = pasta / nome
        if not caminho.is_file():
            raise ValueError(f"Arquivo ausente: {nome}. Execute analisar_carteira.py antes de otimizar.")
        salvo = pd.read_csv(caminho, index_col=0, converters={
            coluna: float for coluna in esperado.select_dtypes(include="number").columns
        })
        try:
            pd.testing.assert_frame_equal(salvo, esperado, check_dtype=False,
                                          check_exact=False, rtol=1e-10, atol=1e-12)
        except AssertionError as erro:
            raise ValueError(f"{nome} não corresponde aos dados e ao treinamento atuais. "
                             "Revise e execute analisar_carteira.py antes de otimizar.") from erro
    return resultados


def validar_limites(quantidade, minimo, maximo):
    """Para limites uniformes, n*mínimo <= 1 <= n*máximo garante viabilidade."""
    if quantidade < 1 or not np.isfinite([minimo, maximo]).all() or not 0 <= minimo <= maximo <= 1:
        raise ValueError("Use ativos e limites válidos entre 0 e 1, sem posições vendidas ou alavancagem.")
    if quantidade * minimo > 1 + 1e-12 or quantidade * maximo < 1 - 1e-12:
        raise ValueError(f"Limites inviáveis: {quantidade} ativos com pesos entre {minimo:.2%} e "
                         f"{maximo:.2%} não permitem soma de 100%.")


def validar_covariancia(matriz):
    """Rejeita assimetria, rótulos inconsistentes e matriz não semidefinida positiva."""
    if not isinstance(matriz, pd.DataFrame) or matriz.empty:
        raise ValueError("A covariância deve ser uma tabela não vazia com os nomes dos ativos.")
    if matriz.index.has_duplicates or not matriz.index.equals(matriz.columns):
        raise ValueError("As linhas e colunas da covariância devem ter os mesmos ativos na mesma ordem.")
    valores = matriz.to_numpy(dtype=float)
    if valores.shape != (len(matriz), len(matriz)) or not np.isfinite(valores).all():
        raise ValueError("A covariância deve ser quadrada, numérica e finita.")
    if not np.allclose(valores, valores.T, rtol=1e-10, atol=1e-12):
        raise ValueError("A matriz de covariância não é simétrica.")
    autovalores = np.linalg.eigvalsh(valores)
    tolerancia = 1e-10 * max(float(np.max(np.abs(autovalores))), np.finfo(float).eps)
    if autovalores.min() < -tolerancia:
        raise ValueError("A matriz de covariância não é positiva semidefinida; revise a Etapa 02.")


def validar_pesos(pesos, ativos, minimo=PESO_MINIMO, maximo=PESO_MAXIMO):
    if not isinstance(pesos, pd.Series) or not pesos.index.equals(ativos):
        raise ValueError("A ordem e os ativos dos pesos devem coincidir com a covariância.")
    valores = pesos.to_numpy(dtype=float)
    if not np.isfinite(valores).all():
        raise ValueError("Há pesos ausentes ou infinitos.")
    if not np.isclose(valores.sum(), 1, rtol=0, atol=TOLERANCIA_PESOS):
        raise ValueError("Os pesos não somam 100% dentro da tolerância numérica.")
    if (valores < minimo - TOLERANCIA_PESOS).any() or (valores > maximo + TOLERANCIA_PESOS).any():
        raise ValueError("O resultado ultrapassou os limites de alocação.")


def otimizar_pesos(covariancia_anual, minimo=PESO_MINIMO, maximo=PESO_MAXIMO, solver=SOLVER):
    """Minimiza w.T @ S @ w: variância anual, com soma dos pesos igual a 1."""
    validar_covariancia(covariancia_anual)
    validar_limites(len(covariancia_anual), minimo, maximo)
    try:
        import cvxpy as cp
        from pypfopt import EfficientFrontier
        from pypfopt.exceptions import OptimizationError
    except ImportError as erro:
        raise RuntimeError("PyPortfolioOpt ou suas dependências não estão disponíveis. "
                           "Execute python -m pip install -r requirements.txt no ambiente do projeto.") from erro
    if solver not in cp.installed_solvers():
        raise RuntimeError(f"Solver {solver} indisponível. Disponíveis: {cp.installed_solvers()}. "
                           "Confira a instalação; nenhum solver alternativo foi usado.")
    try:
        fronteira = EfficientFrontier(
            expected_returns=None, cov_matrix=covariancia_anual,
            weight_bounds=(minimo, maximo), solver=solver,
            solver_options=OPCOES_SOLVER if solver == "CLARABEL" else None,
        )
        fronteira.min_volatility()
    except (OptimizationError, cp.error.SolverError, cp.error.DCPError, ValueError, TypeError) as erro:
        raise RuntimeError(f"Falha na otimização com {solver}: {erro}. Nenhum peso substituto foi gerado.") from erro
    # weights contém a solução do método, antes de qualquer limpeza para apresentação.
    pesos = pd.Series(fronteira.weights, index=fronteira.tickers, name="peso_otimizado_decimal", dtype=float)
    validar_pesos(pesos, covariancia_anual.index, minimo, maximo)
    return pesos


def calcular_desempenho(pesos, resumo_ativos, covariancia_semanal):
    """Avalia a solução sem arredondar pesos e sem usar semanas do conjunto de teste."""
    if not pesos.index.equals(resumo_ativos.index) or not pesos.index.equals(covariancia_semanal.index):
        raise ValueError("Ordem dos ativos inconsistente na avaliação da carteira.")
    vetor = pesos.to_numpy()
    variancia = float(vetor @ covariancia_semanal.to_numpy() @ vetor)
    if not np.isfinite(variancia) or variancia < -1e-14:
        raise ValueError("Variância inválida ao avaliar a carteira.")
    volatilidade = np.sqrt(max(variancia, 0.0))
    media = float(pesos @ resumo_ativos[COLUNAS_RESUMO[0]])
    return pd.Series({
        COLUNAS_RESUMO[0]: media,
        COLUNAS_RESUMO[1]: media * SEMANAS_ANO,
        COLUNAS_RESUMO[2]: volatilidade,
        COLUNAS_RESUMO[3]: volatilidade * np.sqrt(SEMANAS_ANO),
    })


def otimizar_resultados(resultados_etapa_2, minimo=PESO_MINIMO, maximo=PESO_MAXIMO, solver=SOLVER):
    """Usa a mesma referência e covariância semanal validadas na Etapa 02."""
    resumo_ativos = resultados_etapa_2["resumo_risco.csv"]
    covariancia_semanal = resultados_etapa_2["matriz_covariancia.csv"]
    referencia = resultados_etapa_2["resumo_carteira_igualitaria.csv"].iloc[0]
    validar_covariancia(covariancia_semanal)
    if not resumo_ativos.index.equals(covariancia_semanal.index):
        raise ValueError("O resumo individual e a covariância têm ativos ou ordens diferentes.")
    ativos = covariancia_semanal.index
    validar_limites(len(ativos), minimo, maximo)
    iguais = pd.Series(1.0 / len(ativos), index=ativos, name="peso_igualitario_decimal")
    validar_pesos(iguais, ativos, minimo, maximo)

    # Único ponto de anualização da matriz: o arquivo original é SEMANAL em decimal².
    covariancia_anual = covariancia_semanal * SEMANAS_ANO
    otimizados = otimizar_pesos(covariancia_anual, minimo, maximo, solver)
    desempenho = calcular_desempenho(otimizados, resumo_ativos, covariancia_semanal)
    volatilidade_direta = np.sqrt(max(float(otimizados.to_numpy() @ covariancia_anual.to_numpy()
                                           @ otimizados.to_numpy()), 0.0))
    if not np.isclose(desempenho[COLUNAS_RESUMO[3]], volatilidade_direta, rtol=1e-10, atol=1e-12):
        raise ValueError("Inconsistência entre o risco semanal e o risco anualizado.")
    if desempenho[COLUNAS_RESUMO[3]] > referencia[COLUNAS_RESUMO[3]] + TOLERANCIA_RISCO:
        raise ValueError("A volatilidade otimizada excede a igualitária além da tolerância; nenhum resultado será salvo.")
    pesos = pd.concat([iguais, otimizados], axis=1).rename_axis("Ativo")
    resumo = pd.DataFrame([referencia[COLUNAS_RESUMO].astype(float), desempenho],
                         index=pd.Index(["igualitaria", "minima_volatilidade"], name="Estrategia"))
    if not np.isfinite(resumo.to_numpy()).all() or (resumo[COLUNAS_RESUMO[2:]] < 0).any().any():
        raise ValueError("A comparação contém valores inválidos.")
    return {"pesos_otimizados.csv": pesos, "resumo_otimizacao.csv": resumo}


def apresentar_resultados(resultados, resultados_etapa_2, minimo=PESO_MINIMO, maximo=PESO_MAXIMO):
    referencia = resultados_etapa_2["resumo_carteira_igualitaria.csv"].iloc[0]
    pesos = resultados["pesos_otimizados.csv"]
    resumo = resultados["resumo_otimizacao.csv"]
    print(f"Treinamento: {referencia['inicio_treinamento']} a {referencia['fim_treinamento']} "
          f"— {referencia['semanas_treinamento']} observações semanais.")
    print(f"Teste reservado: {referencia['semanas_teste']} semanas; não utilizado na escolha dos pesos.")
    print("Ativos: " + ", ".join(pesos.index))
    print(f"Limites por ativo: {minimo:.2%} a {maximo:.2%}; soma: 100%; solver: {SOLVER}.")
    print("Objetivo: minimizar a variância, não maximizar o retorno.")
    print("\nPesos em porcentagem (arredondados apenas nesta exibição):")
    print((pesos * 100).rename(columns={
        "peso_igualitario_decimal": "Peso igualitário (%)",
        "peso_otimizado_decimal": "Peso otimizado (%)",
    }).to_string(float_format=lambda valor: f"{valor:.6f}"))
    exibicao = resumo * 100
    exibicao.columns = ["Média semanal estimada (%)", "Média anual aritmética estimada (%)",
                       "Volatilidade semanal (%)", "Volatilidade anualizada (%)"]
    print("\nComparação no mesmo treinamento:")
    print(exibicao.to_string(float_format=lambda valor: f"{valor:.6f}"))
    igual = resumo.loc["igualitaria", COLUNAS_RESUMO[3]]
    otimizada = resumo.loc["minima_volatilidade", COLUNAS_RESUMO[3]]
    print(f"Redução da volatilidade anualizada: {(igual - otimizada) * 100:.6f} pontos percentuais.")
    if igual > 0:
        print(f"Redução relativa da volatilidade: {(1 - otimizada / igual) * 100:.6f}%.")
    else:
        print("Redução relativa indefinida: a volatilidade da referência é zero.")
    print("Estimativas históricas do treinamento; não representam retornos futuros garantidos.")
    print("Média anualizada aritmética = média semanal × 52; volatilidade anualizada = semanal × raiz de 52.")
    print("CSVs em escala DECIMAL, sem arredondamento de apresentação. Matrizes originais preservadas.")


def executar_testes():
    """Matrizes artificiais somente em testes; nenhum resultado real é substituído."""
    from contextlib import redirect_stdout
    from io import StringIO
    import tempfile
    from unittest.mock import patch
    import cvxpy as cp
    from pypfopt import EfficientFrontier

    def conferir(valor, esperado, tolerancia=1e-6):
        np.testing.assert_allclose(valor, esperado, rtol=1e-6, atol=tolerancia)

    def esperar_erro(funcao, *argumentos, **opcoes):
        try:
            funcao(*argumentos, **opcoes)
        except (ValueError, RuntimeError):
            return
        raise AssertionError("Era esperada a interrupção com mensagem de erro.")

    ativos = pd.Index(["A", "B", "C"])
    identidade = pd.DataFrame(np.eye(3) * 0.04, index=ativos, columns=ativos)
    with patch.object(EfficientFrontier, "clean_weights", side_effect=AssertionError("Pesos não podem ser arredondados")):
        pesos = otimizar_pesos(identidade)
    conferir(pesos, [1/3, 1/3, 1/3])
    conferir(np.sqrt(pesos.to_numpy() @ identidade.to_numpy() @ pesos.to_numpy()), np.sqrt(0.04 / 3))
    diagonal = pd.DataFrame(np.diag([0.01, 0.02, 0.03]), index=ativos, columns=ativos)
    pesos = otimizar_pesos(diagonal)
    # O primeiro limite fica ativo; os 60% restantes se distribuem em razão inversa 2:3.
    conferir(pesos, [0.4, 0.36, 0.24])
    variancia = sum(pesos.iloc[i] * pesos.iloc[j] * diagonal.iloc[i, j]
                    for i in range(3) for j in range(3))
    conferir(variancia, 0.00592)
    conferir(otimizar_pesos(diagonal / 52), pesos)
    singular = pd.DataFrame(np.full((3, 3), 0.01), index=ativos, columns=ativos)
    validar_pesos(otimizar_pesos(singular), ativos)
    print("OK: solução simétrica conhecida, limite de 40%, somatórios, escala anual e matriz singular PSD.")

    esperar_erro(validar_limites, 2, 0, 0.4)
    esperar_erro(validar_limites, 3, 0.4, 0.5)
    esperar_erro(validar_limites, 3, -0.1, 0.4)
    esperar_erro(validar_limites, 3, 0, float("nan"))
    esperar_erro(otimizar_pesos, diagonal.iloc[::-1])
    invalida = diagonal.copy()
    invalida.iloc[0, 0] = -0.01
    esperar_erro(otimizar_pesos, invalida)
    invalida = diagonal.copy()
    invalida.iloc[0, 1] = 0.01
    esperar_erro(otimizar_pesos, invalida)
    for valores in [[0.5, 0.3, 0.2], [-0.1, 0.4, 0.7], [0.2, 0.2, 0.2], [np.nan, 0.4, 0.6]]:
        esperar_erro(validar_pesos, pd.Series(valores, index=ativos), ativos)
    esperar_erro(otimizar_pesos, diagonal, solver="SOLVER_INEXISTENTE")
    with patch.object(EfficientFrontier, "min_volatility", side_effect=cp.error.SolverError("Falha simulada")):
        esperar_erro(otimizar_pesos, diagonal)
    with patch.dict(sys.modules, {"pypfopt": None}):
        esperar_erro(otimizar_pesos, diagonal)
    print("OK: limites inviáveis, pesos inválidos, ordem, PSD, dependência ausente e falhas do solver interrompem.")

    base = np.array([[1, 2, 0, -1, 3], [-2, 1, 3, 2, 0], [2, -1, 1, 4, -2], [0, 3, -2, 1, 2],
                     [3, 0, 2, -1, 1], [1, -2, 0, 3, 4], [-1, 2, 4, 0, 1], [2, 1, -1, 2, -3]], dtype=float)
    retornos = pd.DataFrame(np.tile(base, (9, 1))[:65], columns=list("ABCDE"),
                            index=pd.date_range("2020-01-10", periods=65, freq="W-FRI"))
    etapa_2 = analisar_retornos(retornos)
    resultados = otimizar_resultados(etapa_2)
    alterados = retornos.copy()
    alterados.iloc[52:] = 9000
    sem_vazamento = otimizar_resultados(analisar_retornos(alterados))
    for nome in resultados:
        pd.testing.assert_frame_equal(resultados[nome], sem_vazamento[nome], check_exact=True)
    tabela_pesos = resultados["pesos_otimizados.csv"]
    tabela_resumo = resultados["resumo_otimizacao.csv"]
    assert (tabela_resumo[COLUNAS_RESUMO[2:]] >= 0).all().all()
    assert tabela_resumo.loc["minima_volatilidade", COLUNAS_RESUMO[3]] <= tabela_resumo.loc["igualitaria", COLUNAS_RESUMO[3]] + TOLERANCIA_RISCO
    for coluna in tabela_pesos:
        validar_pesos(tabela_pesos[coluna], tabela_pesos.index)
    decimais_treino = retornos.iloc[:52] / 100
    for estrategia, coluna in [("igualitaria", "peso_igualitario_decimal"), ("minima_volatilidade", "peso_otimizado_decimal")]:
        serie_ponderada = decimais_treino @ tabela_pesos[coluna]
        conferir(tabela_resumo.loc[estrategia, COLUNAS_RESUMO[0]], serie_ponderada.mean(), 1e-12)
        conferir(tabela_resumo.loc[estrategia, COLUNAS_RESUMO[2]], serie_ponderada.std(ddof=1), 1e-12)
        conferir(tabela_resumo.loc[estrategia, COLUNAS_RESUMO[3]], serie_ponderada.std(ddof=1) * np.sqrt(52), 1e-12)
    reordenados = otimizar_resultados(analisar_retornos(retornos[list("EDCBA")]))
    conferir(reordenados["pesos_otimizados.csv"].loc[list("ABCDE")], tabela_pesos)
    with tempfile.TemporaryDirectory() as pasta:
        salvar_resultados(resultados, pasta)
        assert len(list(Path(pasta).iterdir())) == 2
        terminal = StringIO()
        with redirect_stdout(terminal):
            apresentar_resultados(resultados, etapa_2)
        for nome, esperado in resultados.items():
            salvo = pd.read_csv(Path(pasta) / nome, index_col=0,
                                converters={coluna: float for coluna in esperado.columns})
            pd.testing.assert_frame_equal(salvo, esperado, check_exact=True)
            for valor in salvo.to_numpy().ravel():
                assert f"{valor * 100:.6f}" in terminal.getvalue()
    print("OK: teste sem influência, ordem dos ativos, risco das séries ponderadas, comparação e precisão dos CSVs.")
    print("Todos os testes passaram, sem gravar dados artificiais na pasta dos resultados reais.")


def principal():
    argumentos = argparse.ArgumentParser(description=__doc__)
    argumentos.add_argument("--testar", action="store_true", help="Executa apenas os testes locais.")
    opcoes = argumentos.parse_args()
    try:
        if opcoes.testar:
            executar_testes()
        else:
            etapa_2 = carregar_resultados()
            resultados = otimizar_resultados(etapa_2)
            salvar_resultados(resultados, PASTA_DADOS)
            apresentar_resultados(resultados, etapa_2)
            print("\nValidações aprovadas. Dois CSVs salvos em: " + str(PASTA_DADOS))
        return 0
    except Exception as erro:
        print(f"Falha na otimização: {erro}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(principal())
