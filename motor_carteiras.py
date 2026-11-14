"""Etapa 06A: análise de carteiras parametrizada, inteiramente em memória.

Exemplo:
    from motor_carteiras import analisar_carteiras
    resultado = analisar_carteiras(
        ["PETR4", "VALE3", "ITUB4"], "2020-01-01", "2026-11-13", 0.50)
    print(resultado["pesos"])
    print(resultado["resumo"])

As datas são inclusivas, em YYYY-MM-DD ou datetime.date. O final respeita o
corte de pregão encerrado do projeto; semanas em andamento são descartadas.
A primeira semana fornece apenas o fechamento de referência dos retornos.
Aceita de 3 a 12 códigos, com ou sem .SA; peso_maximo é decimal (0.40 = 40%).
São necessárias pelo menos 52 semanas de treino na divisão cronológica 80/20.

Frequência semanal, preços ajustados, posições compradas, mínima volatilidade
e referência igualitária. Backtest buy and hold com R$ 100.000 por estratégia,
sem custos ou rebalanceamento, conforme backtest_carteira.py. O teto de peso
vale na formação; os pesos efetivos podem variar durante o buy and hold.

Cada chamada coleta seus próprios dados, sem ler/gravar CSVs de resultados ou
usar cache de carteiras. Erros de entrada/coleta geram ValueError; erros do
otimizador preservam RuntimeError. Nunca retorna resultados parciais ou pesos
substitutos. Não altera os comandos tradicionais nem a interface.
"""

from datetime import date
from copy import deepcopy
from math import fsum
from numbers import Real
import re

import numpy as np
import pandas as pd

from baixar_dados import calcular_data_limite
from preparar_carteira import (
    coletar_historicos, agrupar_fechamentos, consolidar_precos, alinhar_datas,
    calcular_retornos, validar_resultados,
)
from analisar_carteira import analisar_retornos, separar_conjuntos, SEMANAS_ANO
from maximo_sharpe import otimizar_maximo_sharpe, validar_taxa_livre_risco
from otimizar_carteira import otimizar_resultados, validar_limites, calcular_desempenho, PESO_MAXIMO
from backtest_carteira import (
    executar_backtest, conferir_com_precos, simular_buy_and_hold, calcular_indicadores,
    calcular_drawdown, COLUNA_ESTIMADA, CAPITAL_INICIAL,
)


def validar_parametros(ativos, data_inicial, data_final, peso_maximo):
    """Normaliza entradas e rejeita solicitações inválidas antes de acessar a rede."""
    if not isinstance(ativos, (list, tuple)) or not 3 <= len(ativos) <= 12:
        raise ValueError("Informe uma lista de 3 a 12 códigos da B3.")
    codigos = []
    for ativo in ativos:
        if not isinstance(ativo, str):
            raise ValueError("Cada código de ativo precisa ser um texto, por exemplo PETR4.")
        codigo = ativo.strip().upper().removesuffix(".SA")
        if not re.fullmatch(r"[A-Z][A-Z0-9]{3}[0-9]{1,2}", codigo):
            raise ValueError(f"Código inválido: {ativo!r}. Use o código da B3, com ou sem .SA.")
        codigos.append(codigo + ".SA")
    if len(set(codigos)) != len(codigos):
        raise ValueError("Há ativos duplicados após normalizar os códigos (ex.: PETR4 e PETR4.SA).")

    datas = []
    for nome, valor in [("inicial", data_inicial), ("final", data_final)]:
        try:
            # Evita interpretações ambíguas e descarta implicitamente nenhum horário.
            if type(valor) is date:
                data = valor
            elif isinstance(valor, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", valor):
                data = date.fromisoformat(valor)
            else:
                raise ValueError()
        except ValueError as erro:
            raise ValueError(f"Data {nome} inválida: use YYYY-MM-DD ou datetime.date.") from erro
        datas.append(data)
    inicio, fim = datas
    if inicio >= fim:
        raise ValueError("A data inicial deve ser anterior à data final.")
    if fim > pd.Timestamp.now(tz="America/Sao_Paulo").date():
        raise ValueError("A data final não pode estar no futuro.")
    if isinstance(peso_maximo, bool) or not isinstance(peso_maximo, Real):
        raise ValueError("O limite máximo deve ser numérico e decimal, por exemplo 0.40.")
    maximo = float(peso_maximo)
    validar_limites(len(codigos), 0.0, maximo)
    return codigos, inicio, fim, maximo


def analisar_carteiras(ativos, data_inicial, data_final, peso_maximo=PESO_MAXIMO):
    """Executa coleta → estatísticas → otimização → backtest e retorna um dict.

    Chaves: parametros; precos (ajustados); retornos_percentuais; treinamento e
    teste (retornos decimais); resumo_risco; correlacao; covariancia_semanal
    (decimal²); pesos (decimais); resumo_otimizacao; periodo; resumo (backtest);
    curva (patrimônio/drawdown); posicoes por estratégia; diagnostico_coleta.
    pesos/periodo/resumo/curva também atendem ajustar_capital de dados_carteiras.
    Todos os objetos pertencem à chamada atual, sem estado global de resultados.
    """
    codigos, inicio, fim, maximo = validar_parametros(ativos, data_inicial, data_final, peso_maximo)
    limite = calcular_data_limite(fim)
    historicos, repeticoes = coletar_historicos(codigos, inicio, limite)
    series, informacoes = [], {}
    for codigo in codigos:
        serie, informacoes[codigo] = agrupar_fechamentos(historicos[codigo], inicio, limite)
        series.append(serie)
    precos = alinhar_datas(consolidar_precos(series), informacoes, repeticoes)
    retornos = calcular_retornos(precos)
    validar_resultados(precos, retornos, codigos, inicio, limite)
    treinamento, teste = separar_conjuntos(retornos)
    estatisticas = analisar_retornos(retornos)
    otimizacao = otimizar_resultados(estatisticas, minimo=0.0, maximo=maximo)
    pesos = otimizacao["pesos_otimizados.csv"]
    periodo = estatisticas["resumo_carteira_igualitaria.csv"].iloc[0].copy()
    backtest, posicoes = executar_backtest(
        retornos, pesos, otimizacao["resumo_otimizacao.csv"], periodo)
    conferir_com_precos(posicoes, pesos, CAPITAL_INICIAL, precos=precos)
    return {
        "parametros": {"ativos": tuple(codigos), "data_inicial": inicio,
                       "data_final": fim, "limite_coleta_exclusivo": limite,
                       "peso_minimo": 0.0, "peso_maximo": maximo,
                       "frequencia": "W-FRI", "fracao_treinamento": 0.8,
                       "capital_inicial": CAPITAL_INICIAL},
        "precos": precos, "retornos_percentuais": retornos,
        "treinamento": treinamento, "teste": teste,
        "resumo_risco": estatisticas["resumo_risco.csv"],
        "correlacao": estatisticas["matriz_correlacao.csv"],
        "covariancia_semanal": estatisticas["matriz_covariancia.csv"],
        "pesos": pesos, "resumo_otimizacao": otimizacao["resumo_otimizacao.csv"],
        "periodo": periodo, "resumo": backtest["resumo_backtest.csv"],
        "curva": backtest["curva_patrimonial_backtest.csv"], "posicoes": posicoes,
        "diagnostico_coleta": {"semanas_por_ativo": informacoes, "duplicatas_removidas": repeticoes},
    }


def validar_pesos_personalizados(pesos_percentuais, ativos):
    """Recebe dict/Series em porcentagem e retorna decimais sem normalizar.

    Cada ativo deve estar presente uma vez, inclusive os de peso zero.
    A tolerância de 1e-8 ponto percentual cobre somente aritmética de floats.
    O teto do otimizador não se aplica à carteira manual.
    """
    if not isinstance(pesos_percentuais, (dict, pd.Series)):
        raise ValueError("Informe os pesos percentuais associados aos nomes dos ativos.")
    pesos = pd.Series(pesos_percentuais)
    if pesos.index.has_duplicates or len(pesos) != len(ativos) or set(pesos.index) != set(ativos):
        raise ValueError("Informe exatamente um peso para cada ativo da análise, inclusive os de peso zero.")
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) for x in pesos):
        raise ValueError("Os pesos devem ser números finitos em porcentagem.")
    pesos = pesos.reindex(ativos).astype(float)
    if not np.isfinite(pesos).all():
        raise ValueError("Os pesos devem ser números finitos em porcentagem.")
    if (pesos < 0).any() or (pesos > 100).any():
        raise ValueError("Cada peso deve estar entre 0% e 100%, sem posições vendidas ou alavancagem.")
    total = fsum(pesos)
    if abs(total - 100) > 1e-8:
        # Até 6 casas, sem zeros à direita e com vírgula decimal: 99,999999% não aparece como 100%.
        soma = f"{total:.6f}".rstrip("0").rstrip(".").replace(".", ",")
        raise ValueError(f"Os pesos devem somar 100%. Soma informada: {soma}%. Nenhum peso foi normalizado.")
    return (pesos / 100).rename("peso_personalizado_decimal")


def adicionar_personalizada(carteiras, pesos_percentuais):
    """Retorna uma cópia com a estratégia manual; não altera a análise nem o cache.

    Use após analisar_carteiras: adicionar_personalizada(resultado,
    {"PETR4": 60, "VALE3": 40, "ITUB4": 0}). Mantém capital, datas e premissas.
    As estimativas usam exclusivamente o treinamento já validado; somente o
    backtest recebe os retornos de teste. Não coleta dados nem chama o solver.
    """
    pesos = validar_pesos_personalizados(pesos_percentuais, carteiras["pesos"].index)
    return adicionar_estrategia(carteiras, pesos, "personalizada")


def adicionar_maximo_sharpe(carteiras, taxa_livre_risco):
    """Taxa anual decimal e estatísticas exclusivamente do treinamento já validado."""
    taxa = validar_taxa_livre_risco(taxa_livre_risco)
    retornos_anuais = carteiras["treinamento"].mean() * SEMANAS_ANO
    covariancia_anual = carteiras["covariancia_semanal"] * SEMANAS_ANO
    pesos = otimizar_maximo_sharpe(retornos_anuais, covariancia_anual, taxa,
                                  maximo=carteiras["parametros"]["peso_maximo"])
    resultado = adicionar_estrategia(carteiras, pesos, "maximo_sharpe")
    resultado["parametros"]["taxa_livre_risco_anual_decimal"] = taxa
    return resultado


def adicionar_estrategia(carteiras, pesos, nome):
    """Avaliação comum, sem alterar os objetos originais ou recalcular as outras estratégias."""
    desempenho = calcular_desempenho(pesos, carteiras["resumo_risco"], carteiras["covariancia_semanal"])
    capital = carteiras["parametros"]["capital_inicial"]
    posicoes = simular_buy_and_hold(carteiras["teste"], pesos, carteiras["periodo"]["fim_treinamento"], capital)
    patrimonio = posicoes.sum(axis=1)
    if not patrimonio.index.equals(carteiras["curva"].index):
        raise ValueError("O período da nova carteira não corresponde às demais estratégias.")
    relativos = carteiras["precos"].loc[posicoes.index, pesos.index] / carteiras["precos"].loc[posicoes.index[0], pesos.index]
    np.testing.assert_allclose(posicoes, relativos * (capital * pesos), rtol=1e-10, atol=1e-6)
    indicadores = calcular_indicadores(patrimonio)
    indicadores[COLUNA_ESTIMADA] = desempenho["volatilidade_anualizada_decimal"]
    resultado = deepcopy(carteiras)
    resultado["pesos"][pesos.name] = pesos
    resultado["resumo_otimizacao"].loc[nome] = desempenho
    resultado["resumo"].loc[nome] = indicadores
    resultado["curva"][f"patrimonio_{nome}"] = patrimonio
    resultado["curva"][f"drawdown_{nome}_decimal"] = calcular_drawdown(patrimonio)
    resultado["posicoes"][nome] = posicoes
    return resultado
