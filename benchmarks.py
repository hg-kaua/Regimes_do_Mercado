"""Etapa 10: benchmarks Ibovespa e CDI no mesmo período de teste das carteiras.

Fontes (conferidas antes da implementação):
  Ibovespa — Yahoo Finance, código ^BVSP: fechamentos diários em pontos do índice.
    Pela metodologia da B3, o Ibovespa é um índice de RETORNO TOTAL: os proventos
    são incorporados à carteira teórica. Limitações da comparação com as carteiras,
    que usam preços ajustados do Yahoo (auto_adjust): a carteira teórica da B3 é
    rebalanceada periodicamente e ponderada por free float, o tratamento de
    proventos da B3 difere do ajuste do Yahoo, e o índice não é investível
    diretamente (um ETF teria taxas e erro de acompanhamento).
  CDI — Banco Central do Brasil, SGS série 12 ("Taxa de juros - CDI"), em
    PORCENTAGEM AO DIA ÚTIL (ex.: 0.055131 = 0,055131% a.d. ≈ 14,90% a.a. em 252
    dias). Não é a Selic (SGS 11): as duas costumam ser próximas, mas são taxas
    distintas. A API aceita janelas de no máximo 10 anos para séries diárias.

Capitalização do CDI: a taxa publicada para o dia útil d remunera o overnight de
d até o próximo dia útil. O patrimônio na semana t, a partir da formação f, é
capital × Π(1 + taxa_d / 100) para os dias úteis d com f ≤ d < t. Assim a semana t
usa só taxas conhecidas até t. Se a sexta-feira de referência for feriado, a taxa
do último dia útil anterior entra na semana seguinte (diferença de no máximo um dia).

Ibovespa: fechamento semanal (último pregão da semana W-FRI, sem preencher dias
ausentes), mesmo agrupamento das ações; patrimônio = capital × P_t / P_formação.

Cada benchmark é calculado de forma independente: uma falha marca apenas aquele
benchmark como indisponível, sem interromper a análise das carteiras.
"""

from datetime import timedelta
import json
import urllib.request

import numpy as np
import pandas as pd

from backtest_carteira import CAPITAL_INICIAL, calcular_drawdown, calcular_indicadores
from preparar_carteira import agrupar_fechamentos, coletar_historicos


CODIGO_IBOVESPA = "^BVSP"
SERIE_CDI = 12
URL_SGS = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados?formato=json&dataInicial={inicio}&dataFinal={fim}"
JANELA_MAXIMA_DIAS = 3650  # A API do BCB recusa consultas diárias acima de 10 anos.
# Maior intervalo aceito entre taxas consecutivas: cobre feriados emendados (ex.: Carnaval).
LACUNA_MAXIMA_DIAS = 6
BENCHMARKS = {"ibovespa": "Ibovespa", "cdi": "CDI"}


def baixar_ibovespa(inicio, limite):
    """Fechamentos semanais do ^BVSP entre inicio e limite (exclusivo), sem preenchimento."""
    historicos, _ = coletar_historicos([CODIGO_IBOVESPA], inicio, limite)
    semanais, _ = agrupar_fechamentos(historicos[CODIGO_IBOVESPA], inicio, limite)
    return semanais


def baixar_cdi(inicio, fim):
    """Taxas diárias do CDI (SGS 12) em % a.d., consultadas em janelas de até 10 anos."""
    partes, atual = [], inicio
    while atual <= fim:
        final = min(fim, atual + timedelta(days=JANELA_MAXIMA_DIAS - 1))
        url = URL_SGS.format(serie=SERIE_CDI, inicio=f"{atual:%d/%m/%Y}", fim=f"{final:%d/%m/%Y}")
        requisicao = urllib.request.Request(url, headers={"Accept": "application/json",
                                                          "User-Agent": "regimes-do-mercado"})
        with urllib.request.urlopen(requisicao, timeout=30) as resposta:
            registros = json.load(resposta)
        if not isinstance(registros, list):
            raise ValueError("Resposta inesperada do Banco Central.")
        partes.extend(registros)
        atual = final + timedelta(days=1)
    if not partes:
        raise ValueError("O Banco Central não retornou taxas do CDI para o período.")
    taxas = pd.Series([float(r["valor"]) for r in partes],
                      index=pd.to_datetime([r["data"] for r in partes], format="%d/%m/%Y"), name="cdi")
    return taxas.sort_index()


def curva_ibovespa(semanais, datas, capital=CAPITAL_INICIAL):
    """Patrimônio = capital × P_t / P_formação nas mesmas datas da curva das carteiras."""
    ausentes = datas.difference(semanais.dropna().index)
    if len(ausentes):
        raise ValueError("semanas sem fechamento do Ibovespa: "
                         + ", ".join(pd.DatetimeIndex(ausentes).strftime("%d/%m/%Y")))
    precos = semanais.loc[datas].astype(float)
    if not np.isfinite(precos).all() or (precos <= 0).any():
        raise ValueError("pontos do Ibovespa inválidos.")
    return capital * precos / precos.iloc[0]


def curva_cdi(taxas, datas, capital=CAPITAL_INICIAL):
    """Capitaliza as taxas diárias (% a.d.) dos dias úteis f ≤ d < t de cada semana t."""
    taxas = taxas.sort_index()
    if taxas.index.has_duplicates or not np.isfinite(taxas).all() or ((taxas < 0) | (taxas >= 1)).any():
        raise ValueError("taxas do CDI duplicadas, inválidas ou fora da unidade esperada (% ao dia).")
    janela = taxas[(taxas.index >= datas[0]) & (taxas.index < datas[-1])]
    marcos = pd.DatetimeIndex([datas[0], *janela.index, datas[-1]])
    if janela.empty or (marcos.to_series().diff().dt.days > LACUNA_MAXIMA_DIAS).any():
        raise ValueError("série do CDI com lacunas no período de teste; nenhum valor foi preenchido.")
    fatores = (1 + janela / 100).cumprod()
    curva = []
    for data in datas:
        anteriores = fatores[fatores.index < data]
        curva.append(capital * (anteriores.iloc[-1] if len(anteriores) else 1.0))
    return pd.Series(curva, index=datas, name="cdi")


def calcular_benchmarks(datas, limite, capital=CAPITAL_INICIAL, *, ibovespa=baixar_ibovespa, cdi=baixar_cdi):
    """Curvas e indicadores dos benchmarks nas datas da curva das carteiras.

    datas: índice semanal da curva (primeira = formação). limite: corte exclusivo da
    coleta das carteiras. As funções de coleta podem ser substituídas (cache, testes).
    Retorna {"curva", "resumo", "indisponiveis"}; nunca levanta por falha de fonte.
    """
    datas = pd.DatetimeIndex(datas)
    inicio = (datas[0] - pd.Timedelta(days=6)).date()  # Sábado anterior: semana da formação completa.
    fim_coleta = min(pd.Timestamp(limite), datas[-1] + pd.Timedelta(days=1)).date()
    curvas, indisponiveis = {}, {}
    for nome, montar in [
        ("ibovespa", lambda: curva_ibovespa(ibovespa(inicio, fim_coleta), datas, capital)),
        ("cdi", lambda: curva_cdi(cdi(datas[0].date(), datas[-1].date()), datas, capital)),
    ]:
        try:
            curvas[nome] = montar()
        except Exception as erro:
            indisponiveis[nome] = str(erro) or erro.__class__.__name__
    curva = pd.DataFrame(curvas, index=datas)
    resumo = pd.DataFrame({nome: calcular_indicadores(serie) for nome, serie in curva.items()}).T
    for nome, serie in curva.items():
        curva[f"drawdown_{nome}_decimal"] = calcular_drawdown(serie)
    return {"curva": curva, "resumo": resumo, "indisponiveis": indisponiveis, "capital": capital}
