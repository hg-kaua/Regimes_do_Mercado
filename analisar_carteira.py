"""Etapa 02: risco no treinamento e carteira de referência com pesos iguais.

Execução: python analisar_carteira.py
Testes matemáticos, sem downloads nem resultados reais: python analisar_carteira.py --testar

Entradas de dados/carteiras/: precos_semanais.csv e retornos_semanais.csv (em %).
A conversão de porcentagem para decimal ocorre uma única vez em separar_conjuntos.
Saídas na mesma pasta, calculadas exclusivamente no treinamento:
  resumo_risco.csv: médias e volatilidades em decimal, conforme os cabeçalhos.
  matriz_correlacao.csv: coeficientes de Pearson, adimensionais.
  matriz_covariancia.csv: covariância amostral SEMANAL em retorno decimal ao quadrado.
  resumo_carteira_igualitaria.csv: pesos, médias e volatilidades em decimal;
    variância semanal em decimal²; datas e tamanhos dos conjuntos como metadados.

No terminal, médias e volatilidades são apresentadas em porcentagem. A média
anualizada é aritmética (52 vezes a semanal), não um retorno anual realizado.
A volatilidade usa raiz de 52, uma aproximação que desconsidera autocorrelação.
Os pesos iguais são uma referência estatística fixa, não uma simulação de negociação.
"""

import argparse
import csv
from itertools import combinations
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

from baixar_dados import calcular_data_limite
from preparar_carteira import validar_resultados as validar_dados_etapa_1


PASTA_DADOS = Path(__file__).resolve().parent / "dados" / "carteiras"
SEMANAS_ANO = 52
# Um ano de treinamento é um mínimo operacional, não uma garantia de estabilidade.
MINIMO_SEMANAS_TREINO = 52
COLUNAS_RESUMO = [
    "retorno_medio_semanal_decimal",
    "retorno_medio_anualizado_aritmetico_decimal",
    "volatilidade_semanal_decimal",
    "volatilidade_anualizada_decimal",
]


def ler_tabela(caminho):
    """Valida o cabeçalho antes que o pandas possa renomear colunas duplicadas."""
    caminho = Path(caminho)
    if not caminho.is_file():
        raise ValueError(f"Arquivo ausente: {caminho}. Execute preparar_carteira.py primeiro.")
    with caminho.open(encoding="utf-8-sig", newline="") as arquivo:
        cabecalho = next(csv.reader(arquivo), [])
    if (len(cabecalho) < 3 or cabecalho[0] != "Data"
            or len(cabecalho) != len(set(cabecalho))
            or any(not nome or nome != nome.strip() for nome in cabecalho)):
        raise ValueError(f"Cabeçalho inválido em {caminho.name}: use Data e pelo menos dois ativos distintos.")
    tabela = pd.read_csv(caminho, encoding="utf-8-sig", converters={nome: float for nome in cabecalho[1:]})
    tabela["Data"] = pd.to_datetime(tabela["Data"], format="%Y-%m-%d", errors="raise")
    tabela = tabela.set_index("Data").sort_index()
    if tabela.index.tz is not None or not tabela.index.equals(tabela.index.normalize()):
        raise ValueError(f"Datas com horário ou fuso inesperado em {caminho.name}.")
    return tabela


def ler_dados(pasta=PASTA_DADOS):
    """Lê os dois CSVs existentes e reutiliza as verificações da Etapa 01."""
    pasta = Path(pasta)
    precos = ler_tabela(pasta / "precos_semanais.csv")
    retornos_percentuais = ler_tabela(pasta / "retornos_semanais.csv")
    if precos.empty:
        raise ValueError("A tabela de preços está vazia. Corrija a geração na Etapa 01.")
    validar_dados_etapa_1(
        precos, retornos_percentuais, list(precos.columns),
        precos.index[0].date(), calcular_data_limite(),
    )
    return retornos_percentuais


def separar_conjuntos(retornos_percentuais):
    """Converte % para decimal uma vez e divide cronologicamente em 80%/20%."""
    if not np.isfinite(retornos_percentuais.to_numpy(dtype=float)).all():
        raise ValueError("Há retornos ausentes ou infinitos.")
    if retornos_percentuais.index.has_duplicates or not retornos_percentuais.index.is_monotonic_increasing:
        raise ValueError("Os retornos precisam ter datas únicas em ordem cronológica.")
    corte = int(len(retornos_percentuais) * 0.8)
    if corte < MINIMO_SEMANAS_TREINO or corte == len(retornos_percentuais):
        raise ValueError(
            f"Amostra insuficiente: {corte} semanas no treino. São necessárias pelo menos "
            f"{MINIMO_SEMANAS_TREINO} no treino e uma no teste, mantendo a divisão 80/20."
        )
    retornos_decimais = retornos_percentuais / 100.0
    return retornos_decimais.iloc[:corte].copy(), retornos_decimais.iloc[corte:].copy()


def validar_matrizes(resumo, correlacao, covariancia):
    """Confere ordem dos ativos, simetria, diagonais, limites e valores finitos."""
    ativos = resumo.index
    for nome, matriz in [("correlação", correlacao), ("covariância", covariancia)]:
        if not matriz.index.equals(ativos) or not matriz.columns.equals(ativos):
            raise ValueError(f"Ativos ou ordem incompatíveis na matriz de {nome}.")
        valores = matriz.to_numpy(dtype=float)
        if valores.shape != (len(ativos), len(ativos)) or not np.isfinite(valores).all():
            raise ValueError(f"Matriz de {nome} inválida.")
        if not np.allclose(valores, valores.T, rtol=1e-10, atol=1e-12):
            raise ValueError(f"Matriz de {nome} não simétrica.")
    if not np.allclose(np.diag(correlacao), 1, rtol=0, atol=1e-12):
        raise ValueError("A diagonal da correlação deve ser igual a 1.")
    if (np.abs(correlacao.to_numpy()) > 1 + 1e-12).any():
        raise ValueError("Há correlações fora do intervalo [-1, 1].")
    if (np.diag(covariancia) < 0).any():
        raise ValueError("Há variâncias negativas na diagonal da covariância.")
    if not np.allclose(np.diag(covariancia), resumo["volatilidade_semanal_decimal"] ** 2,
                       rtol=1e-10, atol=1e-14):
        raise ValueError("As variâncias não correspondem ao quadrado das volatilidades.")


def calcular_estatisticas(treinamento):
    """Recebe apenas retornos DECIMAIS do treinamento; usa estatísticas amostrais."""
    if len(treinamento) < 2 or treinamento.shape[1] < 2 or treinamento.columns.has_duplicates:
        raise ValueError("São necessárias duas observações e pelo menos dois ativos distintos.")
    if not np.isfinite(treinamento.to_numpy(dtype=float)).all():
        raise ValueError("O treinamento contém valores ausentes ou infinitos.")
    volatilidades = treinamento.std(ddof=1)
    constantes = treinamento.columns[(treinamento.nunique() < 2) | (volatilidades <= 0)]
    if len(constantes):
        raise ValueError("Correlação indefinida por variância zero: " + ", ".join(constantes)
                         + ". Nenhum ativo será removido automaticamente.")
    medias = treinamento.mean()
    resumo = pd.DataFrame({
        COLUNAS_RESUMO[0]: medias,
        COLUNAS_RESUMO[1]: medias * SEMANAS_ANO,
        COLUNAS_RESUMO[2]: volatilidades,
        COLUNAS_RESUMO[3]: volatilidades * np.sqrt(SEMANAS_ANO),
    }).rename_axis("Ativo")
    correlacao = treinamento.corr(method="pearson")
    covariancia = treinamento.cov(ddof=1)
    validar_matrizes(resumo, correlacao, covariancia)
    if not np.isfinite(resumo.to_numpy()).all():
        raise ValueError("As estatísticas produziram valores inválidos.")
    return resumo, correlacao, covariancia


def identificar_pares(correlacao):
    """Considera cada par uma vez; exclui a diagonal e distingue correlação positiva."""
    pares = [(a, b, float(correlacao.loc[a, b])) for a, b in combinations(correlacao.columns, 2)]
    positivos = [par for par in pares if par[2] > 0]
    maior_positivo = max(positivos, key=lambda par: par[2]) if positivos else None
    menor = min(pares, key=lambda par: par[2])
    return maior_positivo, menor


def calcular_carteira(resumo, covariancia):
    """Pesos iguais calculados a partir dos ativos, com risco semanal por w.T @ cov @ w."""
    ativos = resumo.index
    if not covariancia.index.equals(ativos) or not covariancia.columns.equals(ativos):
        raise ValueError("A ordem dos ativos da covariância deve coincidir com o resumo.")
    pesos = pd.Series(1.0 / len(ativos), index=ativos, name="peso_decimal")
    if (pesos < 0).any() or not np.isclose(pesos.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("Os pesos precisam ser não negativos e somar 1.")
    vetor = pesos.to_numpy()
    variancia = float(vetor @ covariancia.to_numpy() @ vetor)
    if not np.isfinite(variancia) or variancia < -1e-14:
        raise ValueError("A carteira produziu variância inválida.")
    # Apenas resíduos negativos de arredondamento, próximos de zero, são neutralizados.
    variancia = max(variancia, 0.0)
    media = float(pesos @ resumo["retorno_medio_semanal_decimal"])
    volatilidade = float(np.sqrt(variancia))
    volatilidade_anual = volatilidade * np.sqrt(SEMANAS_ANO)
    media_volatilidades = float(pesos @ resumo["volatilidade_anualizada_decimal"])
    if volatilidade_anual > media_volatilidades + 1e-12:
        raise ValueError("A volatilidade da carteira excedeu a média ponderada; confira os cálculos.")
    carteira = {
        "retorno_medio_semanal_decimal": media,
        "retorno_medio_anualizado_aritmetico_decimal": media * SEMANAS_ANO,
        "variancia_semanal_decimal2": variancia,
        "volatilidade_semanal_decimal": volatilidade,
        "volatilidade_anualizada_decimal": volatilidade_anual,
        "media_ponderada_volatilidades_anualizadas_decimal": media_volatilidades,
        "reducao_volatilidade_anual_decimal": media_volatilidades - volatilidade_anual,
    }
    if not np.isfinite(list(carteira.values())).all():
        raise ValueError("A carteira produziu estatísticas inválidas.")
    return pesos, carteira


def analisar_retornos(retornos_percentuais):
    """O teste fornece apenas metadados de período; nunca entra nas estatísticas."""
    treinamento, teste = separar_conjuntos(retornos_percentuais)
    resumo, correlacao, covariancia = calcular_estatisticas(treinamento)
    pesos, carteira = calcular_carteira(resumo, covariancia)
    metadados = {
        "quantidade_ativos": len(pesos), "semanas_por_ano": SEMANAS_ANO,
        "inicio_treinamento": treinamento.index[0].strftime("%Y-%m-%d"),
        "fim_treinamento": treinamento.index[-1].strftime("%Y-%m-%d"),
        "semanas_treinamento": len(treinamento),
        "inicio_teste": teste.index[0].strftime("%Y-%m-%d"),
        "fim_teste": teste.index[-1].strftime("%Y-%m-%d"), "semanas_teste": len(teste),
    }
    resumo_carteira = pd.DataFrame([
        metadados | carteira | {f"peso_{ativo}_decimal": peso for ativo, peso in pesos.items()}
    ], index=pd.Index(["igualitaria"], name="Carteira"))
    return {
        "resumo_risco.csv": resumo,
        "matriz_correlacao.csv": correlacao.rename_axis("Ativo (Pearson, adimensional)"),
        "matriz_covariancia.csv": covariancia.rename_axis("Ativo (covariância semanal, decimal²)"),
        "resumo_carteira_igualitaria.csv": resumo_carteira,
    }


def salvar_resultados(resultados, pasta=PASTA_DADOS):
    """Grava os quatro arquivos somente após conferir a leitura das cópias temporárias."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="risco_", dir=pasta) as temporaria:
        for nome, tabela in resultados.items():
            caminho = Path(temporaria) / nome
            tabela.to_csv(caminho, encoding="utf-8", index=True)
            conversores = {coluna: float for coluna in tabela.select_dtypes(include="number").columns}
            salva = pd.read_csv(caminho, index_col=0, converters=conversores)
            # Compara números sem tolerância: o CSV deve manter a precisão completa.
            pd.testing.assert_frame_equal(tabela, salva, check_dtype=False, check_exact=True)
        for nome in resultados:
            (Path(temporaria) / nome).replace(pasta / nome)


def apresentar_resultados(resultados):
    resumo = resultados["resumo_risco.csv"]
    correlacao = resultados["matriz_correlacao.csv"]
    covariancia = resultados["matriz_covariancia.csv"]
    carteira = resultados["resumo_carteira_igualitaria.csv"].iloc[0]
    for conjunto in ["treinamento", "teste"]:
        print(f"{conjunto.capitalize()}: {carteira['inicio_' + conjunto]} a "
              f"{carteira['fim_' + conjunto]} — {carteira['semanas_' + conjunto]} semanas.")
    print("Somente o treinamento participa dos cálculos; o teste permanece reservado.")
    print("\nAções — estatísticas do treinamento em porcentagem (%):")
    exibicao = (resumo * 100).copy()
    exibicao.columns = ["Média semanal (%)", "Média anual aritmética (%)",
                       "Volatilidade semanal (%)", "Volatilidade anualizada (%)"]
    print(exibicao.to_string(float_format=lambda valor: f"{valor:.6f}"))
    print("\nCorrelação de Pearson (adimensional):")
    print(correlacao.to_string(float_format=lambda valor: f"{valor:.6f}"))
    maior, menor = identificar_pares(correlacao)
    if maior:
        print(f"Maior correlação positiva: {maior[0]} × {maior[1]} = {maior[2]:.6f}.")
    else:
        print("Não há pares com correlação estritamente positiva.")
    print(f"Menor correlação: {menor[0]} × {menor[1]} = {menor[2]:.6f}.")
    print("\nCovariância amostral SEMANAL (retorno decimal²; não anualizada):")
    print(covariancia.to_string(float_format=lambda valor: f"{valor:.10f}"))
    print(f"\nCarteira igualitária — {carteira['quantidade_ativos']} ativos:")
    for ativo in resumo.index:
        print(f"  {ativo}: {carteira[f'peso_{ativo}_decimal']:.2%}")
    print(f"Média semanal: {carteira['retorno_medio_semanal_decimal']:.6%}")
    print(f"Média anualizada aritmética: {carteira['retorno_medio_anualizado_aritmetico_decimal']:.6%}")
    print(f"Volatilidade semanal: {carteira['volatilidade_semanal_decimal']:.6%}")
    print(f"Volatilidade anualizada: {carteira['volatilidade_anualizada_decimal']:.6%}")
    comparacao = pd.DataFrame({
        "Volatilidade do ativo (%)": resumo["volatilidade_anualizada_decimal"] * 100,
        "Volatilidade da carteira (%)": carteira["volatilidade_anualizada_decimal"] * 100,
        "Carteira menos ativo (p.p.)": (carteira["volatilidade_anualizada_decimal"]
                                        - resumo["volatilidade_anualizada_decimal"]) * 100,
    })
    print("\nComparação das volatilidades anualizadas:")
    print(comparacao.to_string(float_format=lambda valor: f"{valor:.6f}"))
    print(f"Média ponderada das volatilidades individuais: "
          f"{carteira['media_ponderada_volatilidades_anualizadas_decimal']:.6%}.")
    reducao = carteira["reducao_volatilidade_anual_decimal"]
    if reducao > 1e-12:
        print(f"A combinação reduziu a volatilidade anualizada em {reducao * 100:.6f} p.p. "
              "em relação à média ponderada das volatilidades individuais no treinamento.")
    else:
        print("Não houve redução de volatilidade em relação à média ponderada, dentro da tolerância numérica.")
    print("Média anualizada aritmética não é retorno anual realizado ou garantido.")
    print("A anualização por raiz de 52 desconsidera autocorrelação. Não há recomendação de compra ou venda.")
    print("CSVs: retornos, volatilidades e pesos em decimal; covariância semanal em decimal²; correlação sem unidade.")


def executar_testes():
    """Dados artificiais somente em memória/temporários, sem gravar resultados reais."""
    from io import StringIO
    from contextlib import redirect_stdout

    def conferir(valor, esperado):
        np.testing.assert_allclose(valor, esperado, rtol=1e-10, atol=1e-12)

    def esperar_erro(funcao, *argumentos):
        try:
            funcao(*argumentos)
        except ValueError:
            return
        raise AssertionError("Era esperado um erro de validação.")

    # Médias, variâncias e covariância conhecidos: cov(A,B)=0,0002 e var(A)=0,0005/3.
    exemplo = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.04], "B": [0.04, 0.02, 0.08, 0.06]})
    resumo, correlacao, covariancia = calcular_estatisticas(exemplo)
    medias = np.array([0.025, 0.05])
    variancias = np.array([0.0005 / 3, 0.002 / 3])
    conferir(resumo[COLUNAS_RESUMO[0]], medias)
    conferir(np.diag(covariancia), variancias)
    # O somatório cruzado deste exemplo é 0,0006; corr(A,B)=0,6.
    cov_ab = 0.0006 / 3
    conferir(covariancia.loc["A", "B"], cov_ab)
    conferir(correlacao.loc["A", "B"], 0.6)
    conferir(resumo[COLUNAS_RESUMO[2]], np.sqrt(variancias))
    conferir(resumo[COLUNAS_RESUMO[1]], medias * 52)
    conferir(resumo[COLUNAS_RESUMO[3]], np.sqrt(variancias * 52))
    pesos, carteira = calcular_carteira(resumo, covariancia)
    conferir(pesos, [0.5, 0.5])
    conferir(pesos.sum(), 1)
    assert (pesos >= 0).all() and carteira["volatilidade_semanal_decimal"] >= 0
    variancia_soma = sum(pesos.iloc[i] * pesos.iloc[j] * covariancia.iloc[i, j]
                        for i in range(2) for j in range(2))
    conferir(carteira["variancia_semanal_decimal2"], variancia_soma)
    conferir(carteira["variancia_semanal_decimal2"], 0.25 * (variancias.sum() + 2 * cov_ab))
    conferir(carteira["volatilidade_semanal_decimal"], exemplo.mean(axis=1).std(ddof=1))
    conferir(carteira["retorno_medio_semanal_decimal"], 0.0375)
    conferir(carteira["volatilidade_anualizada_decimal"], np.sqrt(variancia_soma * 52))
    conferir(covariancia, covariancia.T)
    conferir(correlacao, correlacao.T)
    conferir(np.diag(correlacao), [1, 1])
    print("OK: médias, amostrais ddof=1, correlação, covariância, pesos, risco por somatórios e anualização.")

    inversos = pd.DataFrame({"A": [-0.01, 0.01, -0.01, 0.01], "B": [0.01, -0.01, 0.01, -0.01]})
    resumo_i, correlacao_i, covariancia_i = calcular_estatisticas(inversos)
    _, carteira_i = calcular_carteira(resumo_i, covariancia_i)
    conferir(carteira_i["volatilidade_semanal_decimal"], 0)
    assert identificar_pares(correlacao_i)[0] is None
    conferir(identificar_pares(correlacao_i)[1][2], -1)
    esperar_erro(calcular_estatisticas, exemplo.assign(B=0.02))
    esperar_erro(calcular_estatisticas, exemplo.assign(B=np.nan))
    esperar_erro(calcular_estatisticas, exemplo.assign(B=np.inf))
    esperar_erro(calcular_carteira, resumo, covariancia.iloc[::-1, ::-1])
    for quantidade in [4, 5]:
        dados = pd.DataFrame({f"Ativo_{i}": exemplo["A"] * (i + 1) for i in range(quantidade)})
        resumo_n, _, covariancia_n = calcular_estatisticas(dados)
        pesos_n, _ = calcular_carteira(resumo_n, covariancia_n)
        conferir(pesos_n, np.full(quantidade, 1 / quantidade))
    print("OK: correlação negativa, risco zero, variância zero explícita, valores inválidos e pesos dinâmicos.")

    # 65 semanas: 52 de treino e 13 reservadas. Mudar só o teste não muda nenhuma saída.
    datas = pd.date_range("2020-01-10", periods=65, freq="W-FRI")
    percentuais = pd.DataFrame({"A": np.resize([1., 2., 3., 4.], 65),
                               "B": np.resize([4., 2., 8., 6.], 65)}, index=datas)
    treinamento, teste = separar_conjuntos(percentuais)
    assert len(treinamento) == 52 and len(teste) == 13 and treinamento.index[-1] < teste.index[0]
    conferir(treinamento.iloc[0], [0.01, 0.04])
    resultados = analisar_retornos(percentuais)
    alterados = percentuais.copy()
    alterados.iloc[52:] = 9999.
    for nome, tabela in analisar_retornos(alterados).items():
        pd.testing.assert_frame_equal(tabela, resultados[nome], check_exact=True)
    esperar_erro(separar_conjuntos, percentuais.iloc[:64])
    reordenados = analisar_retornos(percentuais[["B", "A"]])
    for coluna in COLUNAS_RESUMO:
        conferir(reordenados["resumo_carteira_igualitaria.csv"][coluna],
                 resultados["resumo_carteira_igualitaria.csv"][coluna])
    with tempfile.TemporaryDirectory() as pasta:
        salvar_resultados(resultados, pasta)
        assert len(list(Path(pasta).iterdir())) == 4
    # Verifica o contrato dos CSVs: retornos percentuais consistentes com os preços.
    with tempfile.TemporaryDirectory() as pasta:
        pasta = Path(pasta)
        precos = pd.DataFrame({"A": [100., 102., 101., 103.], "B": [50., 51., 52., 51.]},
                              index=pd.date_range("2020-01-03", periods=4, freq="W-FRI"))
        precos.index.name = "Data"
        percentuais_csv = (precos / precos.shift(1) - 1).iloc[1:] * 100
        precos.to_csv(pasta / "precos_semanais.csv")
        percentuais_csv.to_csv(pasta / "retornos_semanais.csv")
        pd.testing.assert_frame_equal(ler_dados(pasta), percentuais_csv, check_freq=False)
        # Um arquivo em decimal, rotulado como entrada percentual, deve ser recusado.
        (percentuais_csv / 100).to_csv(pasta / "retornos_semanais.csv")
        esperar_erro(ler_dados, pasta)
        percentuais_csv[["B", "A"]].to_csv(pasta / "retornos_semanais.csv")
        esperar_erro(ler_dados, pasta)
        (pasta / "retornos_semanais.csv").write_text("Data,A,A\n2020-01-10,2,2\n", encoding="utf-8")
        esperar_erro(ler_dados, pasta)
    with redirect_stdout(StringIO()) as saida:
        apresentar_resultados(resultados)
    assert "Não há pares" not in saida.getvalue()
    print("OK: conversão única, divisão 80/20, teste sem influência, ordem dos ativos e precisão dos quatro CSVs.")
    print("OK: entradas em unidade incorreta, colunas fora de ordem e nomes duplicados são recusados.")
    print("Todos os testes matemáticos passaram; nenhum dado artificial foi salvo nos resultados reais.")


def principal():
    argumentos = argparse.ArgumentParser(description=__doc__)
    argumentos.add_argument("--testar", action="store_true", help="Executa somente testes matemáticos locais.")
    opcoes = argumentos.parse_args()
    try:
        if opcoes.testar:
            executar_testes()
        else:
            resultados = analisar_retornos(ler_dados())
            salvar_resultados(resultados)
            apresentar_resultados(resultados)
            print("\nValidações aprovadas. Quatro CSVs salvos em: " + str(PASTA_DADOS))
        return 0
    except Exception as erro:
        print(f"Falha na análise da carteira: {erro}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(principal())
