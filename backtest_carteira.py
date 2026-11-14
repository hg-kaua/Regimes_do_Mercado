"""Etapa 04: backtesting buy and hold das carteiras fora da amostra.

Execute: python backtest_carteira.py
Testes locais: python backtest_carteira.py --testar

Reutiliza as Etapas 01 a 03 sem alterá-las. Antes da simulação, confere que
pesos_otimizados.csv e resumo_otimizacao.csv são reproduzidos pelo otimizador
usando SOMENTE o treinamento atual (80% iniciais); o teste (20% finais) não
participa da escolha dos pesos, que ficam congelados na data de formação.

Metodologia (buy and hold):
  - Data de formação = última semana do treinamento; o patrimônio nessa data é
    o capital inicial. A primeira variação usa o retorno da primeira semana de
    teste sobre o fechamento da última semana de treinamento.
  - Cada posição recebe capital × peso e evolui pelo produto acumulado dos
    retornos semanais do próprio ativo. Patrimônio = soma das posições.
  - Não há rebalanceamento: os pesos efetivos derivam com os preços. Uma média
    ponderada fixa dos retornos semanais seria outra estratégia (rebalanceada).

Premissas simplificadoras (resultado BRUTO e hipotético, não uma operação real):
  sem corretagem, impostos, spread ou slippage; sem aportes, retiradas ou
  rebalanceamentos; posições fracionárias; preços ajustados (auto_adjust do
  yfinance, Etapa 01), que incorporam proventos como se fossem reinvestidos no
  próprio ativo.

Saídas em dados/carteiras/ (escala DECIMAL, sem arredondamento):
  curva_patrimonial_backtest.csv: patrimônio e drawdown semanais das estratégias.
  resumo_backtest.csv: indicadores realizados no teste, a volatilidade estimada
    no treinamento e a diferença mínima volatilidade − igualitária.
"""

import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

from analisar_carteira import COLUNAS_RESUMO, SEMANAS_ANO, ler_dados, ler_tabela, separar_conjuntos
from otimizar_carteira import carregar_resultados, otimizar_resultados, validar_pesos


PASTA_DADOS = Path(__file__).resolve().parent / "dados" / "carteiras"
# Capital hipotético, apenas ilustrativo: os indicadores percentuais não dependem dele.
CAPITAL_INICIAL = 100_000.00
ESTRATEGIAS = {
    "igualitaria": "peso_igualitario_decimal",
    "minima_volatilidade": "peso_otimizado_decimal",
}
NOME_DIFERENCA = "diferenca_minima_volatilidade_menos_igualitaria"
COLUNAS_METRICAS = [
    "patrimonio_final",
    "retorno_acumulado_decimal",
    "retorno_anualizado_realizado_decimal",
    "volatilidade_anualizada_realizada_decimal",
    "drawdown_maximo_decimal",
]
COLUNA_ESTIMADA = "volatilidade_anualizada_estimada_treino_decimal"
TOLERANCIA_REPRODUCAO = 1e-9


def carregar_pesos(pasta=PASTA_DADOS):
    """Lê os pesos salvos e confirma que vêm apenas do treinamento atual.

    O otimizador é executado novamente sobre as estatísticas de treino (já
    conferidas contra os CSVs da Etapa 02). Se os arquivos da Etapa 03 não
    coincidirem, o backtest é interrompido em vez de usar pesos desatualizados.
    """
    pasta = Path(pasta)
    etapa_2 = carregar_resultados(pasta)
    reproduzidos = otimizar_resultados(etapa_2)
    salvos = {}
    for nome, esperado in reproduzidos.items():
        caminho = pasta / nome
        if not caminho.is_file():
            raise ValueError(f"Arquivo ausente: {nome}. Execute otimizar_carteira.py antes do backtest.")
        salvo = pd.read_csv(caminho, index_col=0, converters={coluna: float for coluna in esperado.columns})
        try:
            pd.testing.assert_frame_equal(salvo, esperado, check_dtype=False, check_exact=False,
                                          rtol=TOLERANCIA_REPRODUCAO, atol=TOLERANCIA_REPRODUCAO)
        except AssertionError as erro:
            raise ValueError(f"{nome} não é reproduzido pelo treinamento atual. "
                             "Execute analisar_carteira.py e otimizar_carteira.py novamente.") from erro
        salvos[nome] = salvo
    pesos = salvos["pesos_otimizados.csv"]
    if list(pesos.columns) != list(ESTRATEGIAS.values()):
        raise ValueError("pesos_otimizados.csv não contém as duas estratégias esperadas.")
    for coluna in pesos:
        validar_pesos(pesos[coluna], pesos.index)
    referencia = etapa_2["resumo_carteira_igualitaria.csv"].iloc[0]
    return pesos, salvos["resumo_otimizacao.csv"], referencia


def separar_teste(retornos_percentuais, referencia):
    """Usa a mesma divisão 80/20 da Etapa 02 e confere as datas registradas."""
    treinamento, teste = separar_conjuntos(retornos_percentuais)
    datas = {
        "fim_treinamento": treinamento.index[-1], "inicio_teste": teste.index[0],
        "fim_teste": teste.index[-1],
    }
    for chave, data in datas.items():
        if data.strftime("%Y-%m-%d") != str(referencia[chave]):
            raise ValueError(f"Divisão cronológica divergente em {chave}; refaça as Etapas 02 e 03.")
    if len(teste) != int(referencia["semanas_teste"]) or not treinamento.index[-1] < teste.index[0]:
        raise ValueError("O teste não corresponde ao período reservado na Etapa 02.")
    return treinamento.index[-1], teste


def simular_buy_and_hold(retornos_teste, pesos, data_formacao, capital=CAPITAL_INICIAL):
    """Valor de cada posição: capital × peso × produto acumulado de (1 + retorno).

    retornos_teste está em DECIMAL. A primeira linha do resultado é a data de
    formação, com as alocações iniciais; nenhum peso é recalculado depois dela.
    """
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("O capital inicial deve ser positivo e finito.")
    if retornos_teste.empty or not retornos_teste.columns.equals(pesos.index):
        raise ValueError("Os retornos do teste e os pesos devem ter os mesmos ativos na mesma ordem.")
    if not np.isfinite(retornos_teste.to_numpy(dtype=float)).all() or (retornos_teste <= -1).any().any():
        raise ValueError("Há retornos de teste ausentes, infinitos ou que zerariam o preço.")
    if not pd.Timestamp(data_formacao) < retornos_teste.index[0]:
        raise ValueError("A data de formação deve ser anterior à primeira semana de teste.")
    alocacao = capital * pesos.astype(float)
    fatores = (1 + retornos_teste).cumprod()
    posicoes = fatores * alocacao
    inicial = pd.DataFrame([alocacao.to_numpy()], columns=pesos.index, index=[pd.Timestamp(data_formacao)])
    return pd.concat([inicial, posicoes]).rename_axis("Data")


def calcular_drawdown(patrimonio):
    return patrimonio / patrimonio.cummax() - 1


def calcular_indicadores(patrimonio):
    """Indicadores realizados a partir da curva patrimonial (primeiro ponto = formação)."""
    if len(patrimonio) < 3 or not np.isfinite(patrimonio.to_numpy()).all() or (patrimonio <= 0).any():
        raise ValueError("A curva patrimonial precisa de ao menos duas semanas de teste e valores positivos.")
    semanas = len(patrimonio) - 1
    inicial, final = float(patrimonio.iloc[0]), float(patrimonio.iloc[-1])
    acumulado = final / inicial - 1
    # Taxa composta no período efetivo: semanas / 52 anos (não acumulado × 52).
    anualizado = (final / inicial) ** (SEMANAS_ANO / semanas) - 1
    retornos_semanais = (patrimonio / patrimonio.shift(1) - 1).iloc[1:]
    volatilidade_semanal = float(retornos_semanais.std(ddof=1))
    drawdown = calcular_drawdown(patrimonio)
    return pd.Series({
        "data_formacao": patrimonio.index[0].strftime("%Y-%m-%d"),
        "data_final": patrimonio.index[-1].strftime("%Y-%m-%d"),
        "semanas_teste": semanas,
        "dias_corridos": int((patrimonio.index[-1] - patrimonio.index[0]).days),
        "anos_equivalentes": semanas / SEMANAS_ANO,
        "patrimonio_inicial": inicial,
        "patrimonio_final": final,
        "retorno_acumulado_decimal": acumulado,
        "retorno_anualizado_realizado_decimal": anualizado,
        "volatilidade_semanal_realizada_decimal": volatilidade_semanal,
        "volatilidade_anualizada_realizada_decimal": volatilidade_semanal * np.sqrt(SEMANAS_ANO),
        "drawdown_maximo_decimal": float(drawdown.min()),
        "data_drawdown_maximo": drawdown.idxmin().strftime("%Y-%m-%d"),
    })


def executar_backtest(retornos_percentuais, pesos, resumo_otimizacao, referencia, capital=CAPITAL_INICIAL):
    """Simula as duas estratégias nas mesmas datas e com o mesmo capital."""
    data_formacao, teste = separar_teste(retornos_percentuais, referencia)
    curvas, posicoes, indicadores = {}, {}, {}
    for estrategia, coluna in ESTRATEGIAS.items():
        posicoes[estrategia] = simular_buy_and_hold(teste, pesos[coluna], data_formacao, capital)
        curvas[estrategia] = posicoes[estrategia].sum(axis=1)
        indicadores[estrategia] = calcular_indicadores(curvas[estrategia])
        indicadores[estrategia][COLUNA_ESTIMADA] = float(resumo_otimizacao.loc[estrategia, COLUNAS_RESUMO[3]])
    resumo = pd.DataFrame(indicadores).T.rename_axis("Estrategia")
    # Datas e duração são comuns às duas estratégias; os demais valores viram diferenças.
    diferenca = resumo.loc["minima_volatilidade"].copy()
    for coluna in ["patrimonio_inicial", "volatilidade_semanal_realizada_decimal"] + COLUNAS_METRICAS + [COLUNA_ESTIMADA]:
        diferenca[coluna] = resumo.loc["minima_volatilidade", coluna] - resumo.loc["igualitaria", coluna]
    diferenca["data_drawdown_maximo"] = ""
    resumo.loc[NOME_DIFERENCA] = diferenca
    curva = pd.DataFrame({f"patrimonio_{nome}": serie for nome, serie in curvas.items()})
    for nome, serie in curvas.items():
        curva[f"drawdown_{nome}_decimal"] = calcular_drawdown(serie)
    if not curva.notna().all().all() or not np.isfinite(curva.to_numpy(dtype=float)).all():
        raise ValueError("A curva patrimonial contém valores inválidos.")
    return {"curva_patrimonial_backtest.csv": curva.rename_axis("Data"),
            "resumo_backtest.csv": resumo}, posicoes


def conferir_com_precos(posicoes, pesos, capital, pasta=PASTA_DADOS):
    """Confere o buy and hold com os preços salvos: posição = alocação × P_t / P_formação."""
    precos = ler_tabela(Path(pasta) / "precos_semanais.csv")
    for estrategia, coluna in ESTRATEGIAS.items():
        tabela = posicoes[estrategia]
        relativos = precos.loc[tabela.index, tabela.columns] / precos.loc[tabela.index[0], tabela.columns]
        esperado = relativos * (capital * pesos[coluna])
        np.testing.assert_allclose(tabela.to_numpy(), esperado.to_numpy(), rtol=1e-10, atol=1e-6)


def salvar_resultados(resultados, pasta=PASTA_DADOS):
    """Grava somente depois de confirmar que os CSVs preservam a precisão."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="backtest_", dir=pasta) as temporaria:
        for nome, tabela in resultados.items():
            caminho = Path(temporaria) / nome
            tabela.to_csv(caminho, encoding="utf-8", index=True, date_format="%Y-%m-%d")
            numericas = [c for c in tabela.columns if pd.api.types.is_numeric_dtype(pd.to_numeric(tabela[c], errors="coerce"))
                         and pd.to_numeric(tabela[c], errors="coerce").notna().all()]
            salva = pd.read_csv(caminho, index_col=0, converters={c: float for c in numericas},
                                keep_default_na=False)
            for coluna in numericas:
                np.testing.assert_array_equal(salva[coluna].to_numpy(), tabela[coluna].astype(float).to_numpy())
        for nome in resultados:
            (Path(temporaria) / nome).replace(pasta / nome)


def apresentar_resultados(resultados, capital=CAPITAL_INICIAL):
    resumo = resultados["resumo_backtest.csv"]
    linha = resumo.loc["igualitaria"]
    print(f"Formação das carteiras: {linha['data_formacao']} (última semana do treinamento).")
    print(f"Teste: {linha['semanas_teste']} semanas até {linha['data_final']} — "
          f"{linha['dias_corridos']} dias corridos, {linha['anos_equivalentes']:.6f} anos (semanas / 52).")
    print(f"Capital inicial hipotético: R$ {capital:,.2f} por estratégia; buy and hold, pesos congelados.")
    nomes = {"igualitaria": "Igualitária", "minima_volatilidade": "Mínima volatilidade",
             NOME_DIFERENCA: "Diferença (mín. vol. − igual.)"}
    tabela = pd.DataFrame({
        "Patrimônio final (R$)": resumo["patrimonio_final"].astype(float),
        "Retorno acumulado (%)": resumo["retorno_acumulado_decimal"].astype(float) * 100,
        "Retorno anualizado (%)": resumo["retorno_anualizado_realizado_decimal"].astype(float) * 100,
        "Volatilidade anual (%)": resumo["volatilidade_anualizada_realizada_decimal"].astype(float) * 100,
        "Drawdown máximo (%)": resumo["drawdown_maximo_decimal"].astype(float) * 100,
    }).rename(index=nomes)
    print("\nDesempenho REALIZADO no período de teste (diferenças em R$ e pontos percentuais):")
    print(tabela.to_string(float_format=lambda valor: f"{valor:,.4f}"))
    for estrategia in ESTRATEGIAS:
        print(f"  Pior drawdown de {nomes[estrategia]}: semana de {resumo.loc[estrategia, 'data_drawdown_maximo']}.")
    risco = pd.DataFrame({
        "Estimada no treinamento (%)": resumo[COLUNA_ESTIMADA].astype(float) * 100,
        "Realizada no teste (%)": resumo["volatilidade_anualizada_realizada_decimal"].astype(float) * 100,
    }).rename(index=nomes)
    risco["Realizada − estimada (p.p.)"] = risco.iloc[:, 1] - risco.iloc[:, 0]
    print("\nVolatilidade anualizada: ESTIMATIVA do treinamento × RESULTADO observado no teste:")
    print(risco.to_string(float_format=lambda valor: f"{valor:,.4f}"))
    menor = resumo.loc[NOME_DIFERENCA, "volatilidade_anualizada_realizada_decimal"] < 0
    print("No teste, a carteira de mínima volatilidade apresentou volatilidade realizada "
          + ("MENOR" if menor else "MAIOR OU IGUAL") + " que a igualitária.")
    print("A otimização minimiza o risco ESTIMADO no treinamento; ela não garante menor risco fora da amostra.")
    print("A estimativa supõe pesos fixos (rebalanceados); o teste é buy and hold, com pesos que derivam.")
    print("\nPremissas: sem corretagem, impostos, spread, slippage, aportes, retiradas ou rebalanceamentos;")
    print("posições fracionárias; preços ajustados por proventos. Resultado bruto e hipotético,")
    print("não representa a rentabilidade líquida de uma operação real nem recomendação de investimento.")


def executar_testes():
    """Dados artificiais em memória ou pastas temporárias; nada é gravado nos resultados reais."""

    def conferir(valor, esperado, tolerancia=1e-12):
        np.testing.assert_allclose(valor, esperado, rtol=1e-10, atol=tolerancia)

    def esperar_erro(funcao, *argumentos):
        try:
            funcao(*argumentos)
        except ValueError:
            return
        raise AssertionError("Era esperado um erro de validação.")

    # A: +10%, +10%; B: −50%, +100%. Buy and hold: 100 → 80 → 110,5.
    datas = pd.date_range("2024-01-12", periods=2, freq="W-FRI")
    retornos = pd.DataFrame({"A": [0.10, 0.10], "B": [-0.50, 1.00]}, index=datas)
    pesos = pd.Series([0.5, 0.5], index=["A", "B"])
    formacao = pd.Timestamp("2024-01-05")
    posicoes = simular_buy_and_hold(retornos, pesos, formacao, 100.0)
    conferir(posicoes["A"], [50, 55, 60.5])
    conferir(posicoes["B"], [50, 25, 50])
    patrimonio = posicoes.sum(axis=1)
    conferir(patrimonio, [100, 80, 110.5])
    assert patrimonio.index[0] == formacao
    # Uma média ponderada fixa (rebalanceada) daria 100 → 80 → 124: estratégia diferente.
    rebalanceado = 100 * (1 + retornos @ pesos).cumprod()
    assert not np.isclose(rebalanceado.iloc[-1], patrimonio.iloc[-1])
    conferir(rebalanceado.iloc[-1], 124)
    indicadores = calcular_indicadores(patrimonio)
    conferir(indicadores["retorno_acumulado_decimal"], 0.105)
    conferir(indicadores["retorno_anualizado_realizado_decimal"], 1.105 ** 26 - 1)
    conferir(indicadores["volatilidade_semanal_realizada_decimal"], np.std([-0.2, 0.38125], ddof=1))
    conferir(indicadores["volatilidade_anualizada_realizada_decimal"],
             np.std([-0.2, 0.38125], ddof=1) * np.sqrt(52))
    conferir(indicadores["drawdown_maximo_decimal"], -0.2)
    assert indicadores["semanas_teste"] == 2 and indicadores["dias_corridos"] == 14
    print("OK: buy and hold difere do rebalanceamento, formação no capital inicial, retornos, taxa composta e volatilidade.")

    # Drawdown mede a queda a partir do maior valor anterior, não do valor inicial.
    serie = pd.Series([100., 120., 90., 130., 117.], index=pd.date_range("2024-01-05", periods=5, freq="W-FRI"))
    conferir(calcular_drawdown(serie), [0, 0, -0.25, 0, -0.1])
    conferir(calcular_indicadores(serie)["drawdown_maximo_decimal"], -0.25)
    # Percentuais independem do capital.
    for capital in [1.0, 1e6]:
        outro = calcular_indicadores(simular_buy_and_hold(retornos, pesos, formacao, capital).sum(axis=1))
        for coluna in COLUNAS_METRICAS[1:]:
            conferir(outro[coluna], indicadores[coluna])
        conferir(outro["patrimonio_final"], 1.105 * capital, 1e-6)
    # Sem olhar adiante: alterar a última semana não muda os valores anteriores.
    alterados = retornos.copy()
    alterados.iloc[-1] = [0.9, -0.9]
    conferir(simular_buy_and_hold(alterados, pesos, formacao, 100.0).iloc[:-1], posicoes.iloc[:-1])
    esperar_erro(simular_buy_and_hold, retornos, pesos, datas[0], 100.0)
    esperar_erro(simular_buy_and_hold, retornos, pesos.iloc[::-1], formacao, 100.0)
    esperar_erro(simular_buy_and_hold, retornos, pesos, formacao, 0.0)
    esperar_erro(simular_buy_and_hold, retornos.assign(B=np.nan), pesos, formacao, 100.0)
    esperar_erro(calcular_indicadores, patrimonio.iloc[:2])
    print("OK: drawdown pelo pico anterior, invariância ao capital, ausência de look-ahead e entradas inválidas.")

    # Fluxo completo com 65 semanas artificiais: 52 de treino, 13 de teste.
    base = np.array([[1, 2, 0, -1, 3], [-2, 1, 3, 2, 0], [2, -1, 1, 4, -2], [0, 3, -2, 1, 2],
                     [3, 0, 2, -1, 1], [1, -2, 0, 3, 4], [-1, 2, 4, 0, 1], [2, 1, -1, 2, -3]], dtype=float)
    percentuais = pd.DataFrame(np.tile(base, (9, 1))[:65], columns=list("ABCDE"),
                               index=pd.date_range("2020-01-10", periods=65, freq="W-FRI"))
    from analisar_carteira import analisar_retornos
    etapa_2 = analisar_retornos(percentuais)
    etapa_3 = otimizar_resultados(etapa_2)
    referencia = etapa_2["resumo_carteira_igualitaria.csv"].iloc[0]
    resultados, posicoes = executar_backtest(percentuais, etapa_3["pesos_otimizados.csv"],
                                             etapa_3["resumo_otimizacao.csv"], referencia, 1000.0)
    curva = resultados["curva_patrimonial_backtest.csv"]
    resumo = resultados["resumo_backtest.csv"]
    assert len(curva) == 14 and curva.index[0] == percentuais.index[51] and curva.index[1] == percentuais.index[52]
    conferir(curva.iloc[0, :2], [1000, 1000])
    teste = percentuais.iloc[52:] / 100
    for estrategia, coluna in ESTRATEGIAS.items():
        fatores = (1 + teste).prod()
        conferir(curva[f"patrimonio_{estrategia}"].iloc[-1],
                 float(1000 * etapa_3["pesos_otimizados.csv"][coluna] @ fatores), 1e-9)
        conferir(resumo.loc[estrategia, COLUNA_ESTIMADA],
                 etapa_3["resumo_otimizacao.csv"].loc[estrategia, COLUNAS_RESUMO[3]])
    for coluna in COLUNAS_METRICAS:
        conferir(resumo.loc[NOME_DIFERENCA, coluna],
                 resumo.loc["minima_volatilidade", coluna] - resumo.loc["igualitaria", coluna], 1e-9)
    # Pesos não dependem do teste: alterar o teste muda a curva, mas não os pesos.
    alterados = percentuais.copy()
    alterados.iloc[52:] = 5.0
    etapa_3_alterada = otimizar_resultados(analisar_retornos(alterados))
    pd.testing.assert_frame_equal(etapa_3_alterada["pesos_otimizados.csv"], etapa_3["pesos_otimizados.csv"],
                                  check_exact=True)
    outra_referencia = referencia.copy()
    outra_referencia["fim_teste"] = "2099-01-01"
    esperar_erro(executar_backtest, percentuais, etapa_3["pesos_otimizados.csv"],
                 etapa_3["resumo_otimizacao.csv"], outra_referencia)
    with tempfile.TemporaryDirectory() as pasta:
        salvar_resultados(resultados, pasta)
        assert sorted(p.name for p in Path(pasta).iterdir()) == sorted(resultados)
        salvo = pd.read_csv(Path(pasta) / "curva_patrimonial_backtest.csv", index_col=0, parse_dates=True,
                            converters={c: float for c in curva.columns})
        pd.testing.assert_frame_equal(salvo, curva, check_exact=True, check_freq=False)
    print("OK: fluxo completo, mesmas datas e capital, divisão 80/20, pesos independentes do teste e precisão dos CSVs.")
    print("Todos os testes passaram, sem gravar dados artificiais na pasta dos resultados reais.")


def principal():
    argumentos = argparse.ArgumentParser(description=__doc__)
    argumentos.add_argument("--testar", action="store_true", help="Executa apenas os testes locais.")
    opcoes = argumentos.parse_args()
    try:
        if opcoes.testar:
            executar_testes()
        else:
            pesos, resumo_otimizacao, referencia = carregar_pesos()
            resultados, posicoes = executar_backtest(ler_dados(), pesos, resumo_otimizacao, referencia)
            conferir_com_precos(posicoes, pesos, CAPITAL_INICIAL)
            salvar_resultados(resultados)
            apresentar_resultados(resultados)
            print("\nValidações aprovadas. Dois CSVs salvos em: " + str(PASTA_DADOS))
        return 0
    except Exception as erro:
        print(f"Falha no backtesting: {erro}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(principal())
