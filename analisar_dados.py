"""Agrupa semanas por retorno e amplitude e gera dois gráficos."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")  # Salva os gráficos sem abrir janelas.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Abaixo deste número de períodos, K-Means e Markov não são executados.
MINIMO_OBSERVACOES = 30
VARIAVEIS = ["variacao_percentual", "amplitude_percentual"]
NORMALIZADAS = ["variacao_normalizada", "amplitude_normalizada"]


def normalizar_e_agrupar(dados, mostrar=False):
    """Divide 80/20, ajusta Min-Max e K-Means só no treino e classifica todos os períodos.

    Retorna os dados com colunas novas, o resumo por grupo, os centroides
    (em %, ordenados do grupo 1 ao 3) e a posição do corte treino/teste.
    """
    dados = dados.copy()
    corte = int(len(dados) * 0.8)
    if corte < 3 or corte == len(dados):
        raise ValueError("São necessários ao menos três períodos de treino e um de teste.")
    treinamento = dados.iloc[:corte][VARIAVEIS]
    minimos, maximos = treinamento.min(), treinamento.max()
    escalas = maximos - minimos
    if mostrar:
        print("Min-Max por variável, com limites calculados somente no treinamento:")
    for variavel, normalizada in zip(VARIAVEIS, NORMALIZADAS):
        if escalas[variavel] == 0:
            # Variável constante: treino vira zero; teste mantém sua diferença ao mínimo.
            escalas[variavel] = 1.0
            if mostrar:
                print(f"{variavel}: constante no treino; denominador 1 para evitar divisão por zero.")
        dados[normalizada] = (dados[variavel] - minimos[variavel]) / escalas[variavel]
        if mostrar:
            print(f"{variavel}: mínimo={minimos[variavel]:.6f}%; máximo={maximos[variavel]:.6f}%.")
    if mostrar:
        print("O teste usa os mesmos parâmetros, sem limitar seus valores a [0, 1].")
    if len(dados.iloc[:corte][NORMALIZADAS].drop_duplicates()) < 3:
        raise ValueError("O treinamento precisa de pelo menos três pares distintos de variação e amplitude.")
    dados["conjunto"] = "treinamento"
    dados.loc[dados.index[corte:], "conjunto"] = "teste"

    modelo = KMeans(n_clusters=3, random_state=42, n_init=10)
    modelo.fit(dados.iloc[:corte][NORMALIZADAS])
    if len(np.unique(modelo.labels_)) != 3:
        raise ValueError("O K-Means não encontrou três grupos distintos no treino.")
    centroides = modelo.cluster_centers_ * escalas.to_numpy() + minimos.to_numpy()
    # Grupos numerados pelo centroide do retorno, do menor para o maior.
    ordem = np.argsort(centroides[:, 0], kind="stable")
    nomes_grupos = {int(rotulo): grupo for grupo, rotulo in enumerate(ordem, start=1)}
    # Aplica os mesmos centroides e a mesma numeração ao treino e ao teste.
    rotulos = modelo.predict(dados[NORMALIZADAS])
    dados["grupo"] = [nomes_grupos[int(rotulo)] for rotulo in rotulos]

    resumo = dados.groupby("grupo").agg(
        semanas=("grupo", "count"), retorno_medio=("variacao_percentual", "mean"),
        amplitude_media=("amplitude_percentual", "mean"),
        variacao_minima=("variacao_percentual", "min"), variacao_maxima=("variacao_percentual", "max"),
    )
    resumo["centroide_variacao"] = centroides[ordem, 0]
    resumo["centroide_amplitude"] = centroides[ordem, 1]
    return dados, resumo, centroides[ordem], corte


def principal():
    """Usa somente o treinamento para ajustar a escala e o K-Means."""
    try:
        pasta_dados = Path(__file__).resolve().parent / "dados"
        entrada = pasta_dados / "petrobras.csv"
        if not entrada.exists():
            raise ValueError("Execute baixar_dados.py antes da análise.")
        colunas_originais = ["data", "fechamento", "maxima", "minima", "variacao_percentual", "amplitude_percentual"]
        dados = pd.read_csv(entrada, usecols=colunas_originais)
        dados["data"] = pd.to_datetime(dados["data"], errors="raise")
        dados = dados.sort_values("data").reset_index(drop=True)
        if dados["data"].isna().any() or dados["data"].duplicated().any():
            raise ValueError("Há datas ausentes ou duplicadas no CSV.")
        if not dados["data"].between("2020-01-01", "2026-12-31").all():
            raise ValueError("O CSV contém datas fora de 2020–2026.")
        if not (dados["data"].dt.day_of_week == 4).all():
            raise ValueError("Reexecute a coleta semanal: todas as datas devem ser sextas-feiras.")
        for coluna in colunas_originais[1:]:
            dados[coluna] = pd.to_numeric(dados[coluna], errors="raise")
            if not np.isfinite(dados[coluna]).all():
                raise ValueError(f"Há valores ausentes ou infinitos em {coluna}.")
        if (dados[["fechamento", "maxima", "minima"]] <= 0).any().any():
            raise ValueError("Os preços devem ser positivos.")
        tolerancia = dados["fechamento"].abs() * 1e-10
        fora_intervalo = (dados["fechamento"] < dados["minima"] - tolerancia) | (dados["fechamento"] > dados["maxima"] + tolerancia)
        if fora_intervalo.any() or (dados["maxima"] < dados["minima"]).any() or (dados["amplitude_percentual"] < 0).any():
            raise ValueError("Há preços ou amplitudes inconsistentes no CSV.")

        dados, resumo, centroides, corte = normalizar_e_agrupar(dados, mostrar=True)
        print(f"Semanas: {len(dados)} | Treinamento: {corte} | Teste: {len(dados) - corte}")
        print(f"Período: {dados['data'].iloc[0]:%Y-%m-%d} a {dados['data'].iloc[-1]:%Y-%m-%d}")
        print(f"Início do teste: {dados['data'].iloc[corte]:%Y-%m-%d}")
        print("Resumo descritivo de toda a base (treino + teste); centroides ajustados apenas no treino:")
        print(resumo.to_string(float_format=lambda valor: f"{valor:.6f}"))
        for grupo, linha in resumo.iterrows():
            amplitude = "intermediária"
            if np.isclose(linha["amplitude_media"], resumo["amplitude_media"].min()):
                amplitude = "menor"
            elif np.isclose(linha["amplitude_media"], resumo["amplitude_media"].max()):
                amplitude = "maior"
            print(f"Grupo {grupo}: retorno semanal médio {linha['retorno_medio']:+.2f}% e amplitude semanal média {linha['amplitude_media']:.2f}%, {amplitude} entre os grupos.")

        figura, eixo = plt.subplots(figsize=(11, 4.5), layout="constrained")
        eixo.plot(dados["data"], dados["fechamento"], color="#245b8a", linewidth=1, label="Fechamento semanal ajustado")
        eixo.axvline(dados["data"].iloc[corte], color="#555555", linestyle="--", label="Início do teste")
        eixo.set(title="Fechamento semanal da Petrobras (PETR4) — 2020 a 2026", xlabel="Semana encerrada na sexta-feira", ylabel="Fechamento ajustado (R$)")
        eixo.legend()
        eixo.grid(alpha=0.2)
        figura.savefig(pasta_dados / "evolucao_petrobras.png", dpi=150)
        plt.close(figura)

        figura, eixo = plt.subplots(figsize=(9, 6), layout="constrained")
        for grupo, cor in zip([1, 2, 3], ["#d97706", "#245b8a", "#16856b"]):
            selecao = dados.loc[dados["grupo"] == grupo]
            eixo.scatter(selecao["variacao_percentual"], selecao["amplitude_percentual"], s=14, alpha=0.6, color=cor, label=f"Grupo {grupo}")
            centroide = centroides[grupo - 1]
            eixo.scatter(*centroide, marker="X", s=220, color=cor, edgecolors="black", linewidths=1.2, label=f"Centroide {grupo} (treino)", zorder=5)
        eixo.set(title="Agrupamentos semanais da Petrobras — K-Means", xlabel="Retorno semanal (%)", ylabel="Amplitude semanal (%)")
        eixo.legend()
        eixo.grid(alpha=0.2)
        figura.savefig(pasta_dados / "grupos_petrobras.png", dpi=150)
        plt.close(figura)

        dados = dados[colunas_originais + NORMALIZADAS + ["grupo", "conjunto"]]
        dados.to_csv(pasta_dados / "petrobras_grupos.csv", index=False, date_format="%Y-%m-%d")
        # Resumo com centroides do treino, lido pela interface app.py.
        resumo.to_csv(pasta_dados / "resumo_grupos.csv")
        print(f"CSVs e dois gráficos salvos em: {pasta_dados}")
        return 0
    except Exception as erro:
        plt.close("all")
        print(f"Falha na análise da Petrobras: {erro}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(principal())
