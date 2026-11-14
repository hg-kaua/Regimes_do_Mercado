"""Conta transições entre semanas consecutivas da Petrobras."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calcular_transicoes(grupos, datas, regra="W-FRI"):
    """Retorna contagens e frequências; linhas sem transições ficam indefinidas.

    Só conta pares de períodos consecutivos (ex.: sexta seguinte, fim do mês seguinte).
    """
    contagens = np.zeros((3, 3), dtype=int)
    datas = pd.DatetimeIndex(datas)
    proximo_periodo = pd.tseries.frequencies.to_offset(regra)
    for posicao in range(len(grupos) - 1):
        if datas[posicao + 1] == datas[posicao] + proximo_periodo:
            atual, seguinte = int(grupos[posicao]), int(grupos[posicao + 1])
            contagens[atual - 1, seguinte - 1] += 1
    probabilidades = np.full((3, 3), np.nan)
    for linha in range(3):
        total = contagens[linha].sum()
        if total > 0:
            probabilidades[linha] = contagens[linha] / total
    return contagens, probabilidades


def estimar_markov(dados, regra="W-FRI"):
    """Conta transições separadamente no treino e no teste, sem cruzar a fronteira."""
    treino = dados[dados["conjunto"] == "treinamento"]
    teste = dados[dados["conjunto"] == "teste"]
    contagens_treino, probabilidades_treino = calcular_transicoes(treino["grupo"].to_numpy(), treino["data"], regra)
    contagens_teste, probabilidades_teste = calcular_transicoes(teste["grupo"].to_numpy(), teste["data"], regra)
    return contagens_treino, probabilidades_treino, contagens_teste, probabilidades_teste


def principal():
    """Estima Markov no treinamento e compara com frequências separadas do teste."""
    try:
        pasta_dados = Path(__file__).resolve().parent / "dados"
        entrada = pasta_dados / "petrobras_grupos.csv"
        if not entrada.exists():
            raise ValueError("Execute analisar_dados.py antes de calcular Markov.")
        dados = pd.read_csv(entrada, usecols=["data", "grupo", "conjunto", "variacao_normalizada", "amplitude_normalizada"])
        if not np.isfinite(dados[["variacao_normalizada", "amplitude_normalizada"]].to_numpy(dtype=float)).all():
            raise ValueError("Reexecute a análise: as duas variáveis normalizadas devem ser válidas.")
        dados["data"] = pd.to_datetime(dados["data"], errors="raise")
        dados = dados.sort_values("data").reset_index(drop=True)
        if dados["data"].isna().any() or dados["data"].duplicated().any():
            raise ValueError("Há datas ausentes ou duplicadas.")
        if not dados["data"].between("2020-01-01", "2026-12-31").all():
            raise ValueError("Há datas fora de 2020–2026.")
        if not (dados["data"].dt.day_of_week == 4).all():
            raise ValueError("Reexecute a análise semanal: as datas devem ser sextas-feiras.")
        dados["grupo"] = pd.to_numeric(dados["grupo"], errors="raise")
        if not dados["grupo"].isin([1, 2, 3]).all():
            raise ValueError("Todos os grupos devem ser 1, 2 ou 3.")
        corte = int(len(dados) * 0.8)
        conjuntos_esperados = ["treinamento"] * corte + ["teste"] * (len(dados) - corte)
        if corte < 2 or len(dados) == corte or dados["conjunto"].tolist() != conjuntos_esperados:
            raise ValueError("O CSV deve manter a divisão cronológica 80/20 do K-Means.")

        # Cada conjunto é contado separadamente: não inclui a fronteira treino/teste.
        contagens_treino, probabilidades_treino, contagens_teste, probabilidades_teste = estimar_markov(dados)
        print("Transições semanais: somente pares separados por sete dias, dentro do mesmo conjunto.")
        nomes = ["Grupo 1", "Grupo 2", "Grupo 3"]
        for nome, contagens, probabilidades in [
            ("Treinamento", contagens_treino, probabilidades_treino),
            ("Teste", contagens_teste, probabilidades_teste),
        ]:
            totais = contagens.sum(axis=1)
            observadas = totais > 0
            if not np.allclose(probabilidades[observadas].sum(axis=1), 1):
                raise ValueError("Uma linha observada da matriz não soma 100%.")
            print(f"\n{nome}: {contagens.sum()} transições (linha = atual; coluna = próximo)")
            print("Contagens:")
            print(pd.DataFrame(contagens, index=nomes, columns=nomes).to_string())
            print("Probabilidades/frequências em %:")
            print(pd.DataFrame(probabilidades * 100, index=nomes, columns=nomes).to_string(
                float_format=lambda valor: f"{valor:.2f}", na_rep="sem transições",
            ))
            for grupo, total in enumerate(totais, start=1):
                print(f"{nome} — Grupo {grupo}: {total} transições de saída.")
                if total == 0:
                    print("  Sem transições observadas: probabilidades indefinidas.")
                elif total == 1:
                    print("  Apenas uma transição: base insuficiente para uma estimativa estável.")
                elif total < 10:
                    print("  Poucas transições (menos de 10); frequências sensíveis a cada observação.")

        print("\nPermanência e mudança estimadas no treinamento:")
        permanencias = np.diag(probabilidades_treino)
        for grupo, permanencia in enumerate(permanencias, start=1):
            if np.isfinite(permanencia):
                print(f"Grupo {grupo}: permanecer {permanencia:.2%}; mudar de grupo {1 - permanencia:.2%}.")
                linha = contagens_treino[grupo - 1]
                destinos = ", ".join(str(i + 1) for i in np.flatnonzero(linha == linha.max()))
                print(f"  Transição mais frequente deste grupo: {grupo} → {destinos}, {linha.max()} ocorrência(s) por destino.")
            else:
                print(f"Grupo {grupo}: permanência e mudança indefinidas.")
        if np.isfinite(permanencias).any():
            maior = np.nanmax(permanencias)
            grupos = ", ".join(str(i + 1) for i in np.flatnonzero(np.isclose(permanencias, maior)))
            print(f"Maior permanência: grupo(s) {grupos}, com {maior:.2%}.")
        maior_contagem = contagens_treino.max()
        print("Transição(ões) mais frequente(s) no treinamento, por contagem absoluta:")
        for origem, destino in np.argwhere((contagens_treino == maior_contagem) & (contagens_treino > 0)):
            print(f"Grupo {origem + 1} → Grupo {destino + 1}: {maior_contagem} ocorrências.")

        diferencas = (probabilidades_teste - probabilidades_treino) * 100
        print("\nComparação: frequência do teste menos probabilidade do treino (pontos percentuais):")
        print(pd.DataFrame(diferencas, index=nomes, columns=nomes).to_string(
            float_format=lambda valor: f"{valor:+.2f}", na_rep="sem comparação",
        ))
        if np.isfinite(diferencas).any():
            origem, destino = np.unravel_index(np.nanargmax(np.abs(diferencas)), diferencas.shape)
            print(f"Maior diferença absoluta: Grupo {origem + 1} → Grupo {destino + 1}, {diferencas[origem, destino]:+.2f} p.p.")
        print("As frequências podem mudar entre períodos; o teste não altera a matriz do treino.")
        print("Linhas com poucas transições exigem cautela. A soma exibida pode diferir de 100% por arredondamento.")

        # Matriz do treinamento (linha = atual; coluna = próximo), lida pela interface app.py.
        pd.DataFrame(probabilidades_treino, index=nomes, columns=nomes).to_csv(pasta_dados / "markov_treinamento.csv")
        print(f"Matriz do treinamento salva: {pasta_dados / 'markov_treinamento.csv'}")

        figura, eixo = plt.subplots(figsize=(7, 5.5), layout="constrained")
        cores = plt.get_cmap("Blues").copy()
        cores.set_bad("#dddddd")
        imagem = eixo.imshow(np.ma.masked_invalid(probabilidades_treino * 100), cmap=cores, vmin=0, vmax=100)
        figura.colorbar(imagem, ax=eixo, label="Probabilidade (%)")
        eixo.set(xticks=range(3), yticks=range(3), xticklabels=nomes, yticklabels=nomes,
                 xlabel="Grupo da próxima semana", ylabel="Grupo da semana atual",
                 title="Petrobras (PETR4) — Markov semanal\nGrupos por retorno e amplitude — treinamento")
        for linha in range(3):
            for coluna in range(3):
                valor = probabilidades_treino[linha, coluna] * 100
                texto = f"{valor:.2f}%" if np.isfinite(valor) else "Sem dados"
                eixo.text(coluna, linha, texto, ha="center", va="center", color="white" if valor > 50 else "black")
        figura.savefig(pasta_dados / "markov_petrobras.png", dpi=150)
        plt.close(figura)
        print(f"\nGráfico salvo: {pasta_dados / 'markov_petrobras.png'}")
        return 0
    except Exception as erro:
        plt.close("all")
        print(f"Falha no cálculo de Markov: {erro}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(principal())
