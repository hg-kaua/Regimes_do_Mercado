"""Leitura, para a interface, dos resultados já calculados das Etapas 02 a 04.

Não baixa dados nem recalcula otimização ou backtesting: apenas lê os CSVs de
dados/carteiras/, confere a coerência entre eles e ajusta a escala monetária
da curva patrimonial ao capital escolhido (os percentuais não mudam).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from analisar_carteira import ler_tabela
from backtest_carteira import COLUNA_ESTIMADA, COLUNAS_METRICAS, ESTRATEGIAS
from otimizar_carteira import validar_pesos


PASTA_DADOS = Path(__file__).resolve().parent / "dados" / "carteiras"
ARQUIVOS = [
    "precos_semanais.csv",
    "resumo_carteira_igualitaria.csv",
    "pesos_otimizados.csv",
    "resumo_backtest.csv",
    "curva_patrimonial_backtest.csv",
]
COMANDOS = [
    "python preparar_carteira.py",
    "python analisar_carteira.py",
    "python otimizar_carteira.py",
    "python backtest_carteira.py",
]


def resolver_pasta(pasta):
    return Path(PASTA_DADOS if pasta is None else pasta)


def listar_ausentes(pasta=None):
    return [nome for nome in ARQUIVOS if not (resolver_pasta(pasta) / nome).is_file()]


def assinatura_arquivos(pasta=None):
    """Muda sempre que um CSV é regerado; serve de chave para o cache da interface."""
    return tuple((nome, (resolver_pasta(pasta) / nome).stat().st_mtime_ns) for nome in ARQUIVOS)


def ler_csv(pasta, nome, numericas):
    return pd.read_csv(Path(pasta) / nome, index_col=0, keep_default_na=False,
                       converters={coluna: float for coluna in numericas})


def carregar_carteiras(pasta=None):
    """Lê os quatro CSVs com precisão completa e interrompe se forem incoerentes."""
    pasta = resolver_pasta(pasta)
    ausentes = listar_ausentes(pasta)
    if ausentes:
        raise FileNotFoundError("Arquivos ausentes: " + ", ".join(ausentes))
    pesos = ler_csv(pasta, "pesos_otimizados.csv", ESTRATEGIAS.values())
    periodo = ler_csv(pasta, "resumo_carteira_igualitaria.csv", []).loc["igualitaria"]
    numericas = ["patrimonio_inicial", "semanas_teste", COLUNA_ESTIMADA,
                 "volatilidade_semanal_realizada_decimal"] + COLUNAS_METRICAS
    resumo = ler_csv(pasta, "resumo_backtest.csv", numericas)
    colunas_curva = [f"patrimonio_{nome}" for nome in ESTRATEGIAS] + [f"drawdown_{nome}_decimal" for nome in ESTRATEGIAS]
    curva = ler_csv(pasta, "curva_patrimonial_backtest.csv", colunas_curva)
    curva.index = pd.to_datetime(curva.index, format="%Y-%m-%d")

    if list(pesos.columns) != list(ESTRATEGIAS.values()):
        raise ValueError("pesos_otimizados.csv não contém as duas estratégias esperadas.")
    for coluna in pesos:
        validar_pesos(pesos[coluna], pesos.index)
    if list(curva.columns) != colunas_curva or curva.empty or not np.isfinite(curva.to_numpy()).all():
        raise ValueError("curva_patrimonial_backtest.csv está incompleto ou contém valores inválidos.")
    if not set(ESTRATEGIAS) <= set(resumo.index):
        raise ValueError("resumo_backtest.csv não contém as duas estratégias.")
    for nome in ESTRATEGIAS:
        linha = resumo.loc[nome]
        serie = curva[f"patrimonio_{nome}"]
        coerente = (
            np.isclose(serie.iloc[0], linha["patrimonio_inicial"], rtol=1e-12)
            and np.isclose(serie.iloc[-1], linha["patrimonio_final"], rtol=1e-12)
            and np.isclose(curva[f"drawdown_{nome}_decimal"].min(), linha["drawdown_maximo_decimal"], rtol=1e-12)
            and linha["data_formacao"] == periodo["fim_treinamento"] == f"{curva.index[0]:%Y-%m-%d}"
            and linha["data_final"] == periodo["fim_teste"] == f"{curva.index[-1]:%Y-%m-%d}"
            and int(linha["semanas_teste"]) == len(curva) - 1 == int(periodo["semanas_teste"])
        )
        if not coerente:
            raise ValueError("Os resultados do backtesting não correspondem à curva patrimonial ou ao "
                             "período da análise. Execute novamente otimizar_carteira.py e backtest_carteira.py.")
    # Confirma que a curva foi gerada com ESTES pesos: buy and hold = capital × soma(w × P_t / P_formação).
    precos = ler_tabela(pasta / "precos_semanais.csv")
    if not curva.index.isin(precos.index).all() or not pesos.index.isin(precos.columns).all():
        raise ValueError("A curva patrimonial não corresponde às datas e ativos de precos_semanais.csv.")
    relativos = precos.loc[curva.index, pesos.index] / precos.loc[curva.index[0], pesos.index]
    for nome, coluna in ESTRATEGIAS.items():
        esperado = (relativos @ pesos[coluna]) * resumo.loc[nome, "patrimonio_inicial"]
        if not np.allclose(curva[f"patrimonio_{nome}"], esperado, rtol=1e-9, atol=1e-6):
            raise ValueError("pesos_otimizados.csv não corresponde aos pesos usados no backtesting. "
                             "Execute novamente backtest_carteira.py.")
    return {"pesos": pesos, "periodo": periodo, "resumo": resumo, "curva": curva}


def ajustar_capital(carteiras, capital):
    """Reescala a curva em R$ para o capital escolhido; retornos e drawdowns não mudam."""
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("O capital inicial deve ser positivo.")
    curva = carteiras["curva"]
    patrimonio = pd.DataFrame({
        nome: curva[f"patrimonio_{nome}"] * (capital / carteiras["resumo"].loc[nome, "patrimonio_inicial"])
        for nome in (coluna.removeprefix("patrimonio_") for coluna in curva if coluna.startswith("patrimonio_"))
    })
    return patrimonio
