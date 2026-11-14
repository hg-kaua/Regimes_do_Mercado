"""Etapa 01: coleta diária ajustada, preços alinhados e retornos semanais.

Execute: python preparar_carteira.py
Edite ATIVOS e DATA_INICIAL para mudar o universo ou o início da coleta.
Fonte: Yahoo Finance via yfinance, com auto_adjust=True e sem arredondamento.
Os CSVs ficam em dados/carteiras/, relativos à localização deste script.
Cada data identifica uma semana encerrada na sexta-feira. Em feriados,
usa-se o último fechamento disponível da semana, sem preencher dias ausentes.
Os retornos são PORCENTAGENS: 2.5 significa 2,5%, e não 250%.
"""

from datetime import date
from math import isclose, isfinite
from pathlib import Path
import sys
import tempfile

import pandas as pd
import yfinance as yf

from baixar_dados import calcular_data_limite


ATIVOS = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBAS3.SA", "ABEV3.SA"]
DATA_INICIAL = date(2020, 1, 1)
# Mínimo técnico para conferir três retornos; não é um critério de risco.
MINIMO_SEMANAS = 4
PASTA_SAIDA = Path(__file__).resolve().parent / "dados" / "carteiras"


def extrair_fechamentos(historico, codigo, data_inicial, data_limite):
    """Aceita colunas simples ou MultiIndex nas duas ordens do yfinance."""
    if historico is None or historico.empty:
        raise ValueError(f"{codigo}: nenhum histórico retornado pelo Yahoo Finance.")
    if isinstance(historico.columns, pd.MultiIndex):
        colunas = [c for c in historico.columns if "Close" in c and codigo in c]
    else:
        colunas = [c for c in historico.columns if c == "Close"]
    if len(colunas) != 1:
        raise ValueError(f"{codigo}: coluna Close ausente ou ambígua na resposta.")
    if not isinstance(historico.index, pd.DatetimeIndex):
        raise ValueError(f"{codigo}: a resposta não contém um índice de datas válido.")

    datas = historico.index
    if datas.tz is not None:
        datas = datas.tz_convert("America/Sao_Paulo").tz_localize(None)
    datas = datas.normalize()
    if datas.hasnans:
        raise ValueError(f"{codigo}: há registros sem data.")
    if (datas < pd.Timestamp(data_inicial)).any() or (datas >= pd.Timestamp(data_limite)).any():
        raise ValueError(f"{codigo}: há datas fora do período solicitado.")
    fechamentos = pd.Series(
        pd.to_numeric(historico[colunas[0]], errors="coerce").to_numpy(dtype=float),
        index=datas, name=codigo.removesuffix(".SA"),
    ).sort_index()
    duplicadas = int(fechamentos.index.duplicated().sum())
    if duplicadas:
        contagens = fechamentos.groupby(level=0).nunique(dropna=False)
        conflitos = contagens[contagens > 1].index
        if len(conflitos):
            raise ValueError(f"{codigo}: preços conflitantes em datas duplicadas: {listar_datas(conflitos)}.")
        # Somente repetições com exatamente o mesmo preço podem ser eliminadas.
        fechamentos = fechamentos[~fechamentos.index.duplicated(keep="first")]
    return fechamentos, duplicadas


def coletar_historicos(ativos, data_inicial, data_limite):
    """Consulta cada ativo separadamente e não aceita falhas parciais da coleta."""
    nomes = [codigo.removesuffix(".SA") for codigo in ativos]
    if not ativos or len(set(nomes)) != len(nomes):
        raise ValueError("Informe uma lista não vazia de ativos sem repetições.")
    if data_inicial >= data_limite:
        raise ValueError("A data inicial precisa ser anterior ao limite da coleta.")
    historicos, repeticoes, falhas = {}, {}, []
    for codigo in ativos:
        print(f"Coletando {codigo}...", flush=True)
        try:
            historico = yf.download(
                codigo, start=data_inicial.isoformat(), end=data_limite.isoformat(),
                interval="1d", auto_adjust=True, actions=False, keepna=True,
                rounding=False, repair=False, threads=False, progress=False,
                timeout=30,
            )
            historicos[codigo], repeticoes[codigo] = extrair_fechamentos(
                historico, codigo, data_inicial, data_limite
            )
        except Exception as erro:
            falhas.append(f"{codigo}: {erro}")
    if falhas:
        raise ValueError("Coleta incompleta; nenhum resultado será salvo.\n" + "\n".join(falhas))
    return historicos, repeticoes


def agrupar_fechamentos(fechamentos, data_inicial, data_limite):
    """Usa W-FRI e descarta toda semana que contenha um preço diário inválido."""
    calendario = pd.date_range(data_inicial, pd.Timestamp(data_limite) - pd.Timedelta(days=1), freq="W-FRI")
    if calendario.empty:
        raise ValueError("O intervalo não contém nenhuma semana encerrada.")
    reamostragem = {"rule": "W-FRI", "closed": "right", "label": "right"}
    ausentes = fechamentos.isna()
    invalidos = ~fechamentos.map(lambda valor: isfinite(valor) and valor > 0) & ~ausentes
    quantidade = fechamentos.resample(**reamostragem).size()
    precos = fechamentos.resample(**reamostragem).last().reindex(calendario)
    semanas_ausentes = ausentes.resample(**reamostragem).max().reindex(calendario, fill_value=False).fillna(False)
    semanas_invalidas = invalidos.resample(**reamostragem).max().reindex(calendario, fill_value=False).fillna(False)
    sem_pregoes = quantidade.reindex(calendario, fill_value=0).eq(0)

    # O last() não pode esconder um fechamento explicitamente inválido no fim da semana.
    precos.loc[sem_pregoes | semanas_ausentes | semanas_invalidas] = float("nan")
    motivos = pd.Series("", index=calendario)
    motivos.loc[sem_pregoes] = "sem pregões retornados"
    motivos.loc[semanas_ausentes] = "fechamento ausente ou não numérico"
    motivos.loc[semanas_invalidas] = "fechamento não positivo ou infinito"
    motivos.loc[semanas_ausentes & semanas_invalidas] = "fechamento ausente e preço inválido"
    precos.index.name = "Data"
    informacoes = {
        "recebidas": int(quantidade.reindex(calendario, fill_value=0).gt(0).sum()),
        "validas": int(precos.notna().sum()),
        "em_andamento": int((quantidade.loc[quantidade.index >= pd.Timestamp(data_limite)] > 0).sum()),
        "motivos": motivos[motivos.ne("")],
    }
    return precos, informacoes


def consolidar_precos(series):
    """Reúne os ativos por data, mantendo ausências visíveis até o alinhamento."""
    return pd.concat(series, axis=1, join="outer").sort_index().rename_axis("Data")


def listar_datas(datas):
    return ", ".join(pd.DatetimeIndex(datas).strftime("%Y-%m-%d"))


def alinhar_datas(precos, informacoes, repeticoes):
    """Mantém só semanas comuns e informa cada motivo de descarte."""
    alinhados = precos.dropna(how="any").copy()
    print("\nSemanas por ativo (antes e depois do alinhamento):")
    insuficientes = []
    for codigo, info in informacoes.items():
        nome = codigo.removesuffix(".SA")
        perdidas = int((precos[nome].notna() & precos.isna().any(axis=1)).sum())
        print(f"  {codigo}: {info['recebidas']} recebidas; {info['validas']} válidas; "
              f"{perdidas} válidas descartadas por ausência de outro ativo; "
              f"{len(alinhados)} alinhadas; {info['em_andamento']} ainda não encerradas.")
        print(f"    Duplicatas idênticas removidas: {repeticoes[codigo]}.")
        for motivo, datas in info["motivos"].groupby(info["motivos"]):
            print(f"    {motivo}: {len(datas)} semana(s) — {listar_datas(datas.index)}.")
        if info["validas"] < MINIMO_SEMANAS:
            insuficientes.append(codigo)
    descartadas = precos.index.difference(alinhados.index)
    print(f"Semanas no calendário solicitado: {len(precos)}.")
    print(f"Semanas descartadas no alinhamento (sem duplicar datas): {len(descartadas)}.")
    if len(descartadas):
        print("Datas descartadas: " + listar_datas(descartadas))
    if insuficientes:
        raise ValueError("Histórico insuficiente: " + ", ".join(insuficientes)
                         + f". São necessárias ao menos {MINIMO_SEMANAS} semanas válidas por ativo.")
    if len(alinhados) < MINIMO_SEMANAS:
        raise ValueError(f"Histórico comum insuficiente: {len(alinhados)} semanas; mínimo de {MINIMO_SEMANAS}.")
    intervalos = alinhados.index.to_series().diff().iloc[1:]
    if intervalos.ne(pd.Timedelta(days=7)).any():
        raise ValueError(
            "O alinhamento deixou lacunas internas. Calcular entre essas datas produziria "
            "retornos de várias semanas. Nenhum CSV será substituído; confira a fonte "
            "ou escolha um período contínuo ajustando DATA_INICIAL."
        )
    return alinhados


def calcular_retornos(precos):
    """Retornos em PORCENTAGEM, após o alinhamento, sem qualquer preenchimento."""
    return ((precos / precos.shift(1) - 1) * 100).iloc[1:].copy()


def validar_resultados(precos, retornos, ativos, data_inicial, data_limite):
    """Valida datas, preços e a fórmula de cada retorno antes de salvar."""
    nomes = [codigo.removesuffix(".SA") for codigo in ativos]
    if list(precos.columns) != nomes or list(retornos.columns) != nomes:
        raise ValueError("Os resultados não contêm todos os ativos na ordem configurada.")
    if len(precos) < MINIMO_SEMANAS or len(retornos) != len(precos) - 1:
        raise ValueError("Quantidade de preços ou retornos inconsistente.")
    for nome, tabela in [("preços", precos), ("retornos", retornos)]:
        if not isinstance(tabela.index, pd.DatetimeIndex) or tabela.index.hasnans:
            raise ValueError(f"Há datas inválidas na tabela de {nome}.")
        if tabela.index.has_duplicates or not tabela.index.is_monotonic_increasing:
            raise ValueError(f"Há datas repetidas ou fora de ordem na tabela de {nome}.")
        if tabela.isna().any().any() or not tabela.map(isfinite).all().all():
            raise ValueError(f"Há valores ausentes ou infinitos na tabela de {nome}.")
    if not (precos > 0).all().all():
        raise ValueError("Todos os fechamentos precisam ser positivos.")
    if ((precos.index < pd.Timestamp(data_inicial)).any()
            or (precos.index >= pd.Timestamp(data_limite)).any()
            or (precos.index.day_of_week != 4).any()):
        raise ValueError("Há semanas fora do período, incompletas ou não encerradas na sexta.")
    if precos.index.to_series().diff().iloc[1:].ne(pd.Timedelta(days=7)).any():
        raise ValueError("Há lacunas entre as semanas de preços.")
    if not retornos.index.equals(precos.index[1:]):
        raise ValueError("As datas dos retornos não correspondem às semanas de preços.")
    for coluna in nomes:
        for posicao in range(1, len(precos)):
            esperado = (precos[coluna].iloc[posicao] / precos[coluna].iloc[posicao - 1] - 1) * 100
            if not isclose(retornos[coluna].iloc[posicao - 1], esperado, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Retorno incorreto em {coluna}, {precos.index[posicao]:%Y-%m-%d}.")


def salvar_resultados(precos, retornos, ativos, data_inicial, data_limite, pasta=PASTA_SAIDA):
    """Confere também os CSVs temporários antes de substituir os dois resultados."""
    validar_resultados(precos, retornos, ativos, data_inicial, data_limite)
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    nomes = ["precos_semanais.csv", "retornos_semanais.csv"]
    with tempfile.TemporaryDirectory(prefix="coleta_", dir=pasta) as temporaria:
        caminhos = [Path(temporaria) / nome for nome in nomes]
        for tabela, caminho in zip([precos, retornos], caminhos):
            tabela.to_csv(caminho, index_label="Data", date_format="%Y-%m-%d", encoding="utf-8")
        # float() preserva a representação numérica gravada sem arredondamento extra.
        salvos = [pd.read_csv(caminho, index_col="Data", parse_dates=["Data"],
                             converters={ativo: float for ativo in precos.columns})
                  for caminho in caminhos]
        validar_resultados(*salvos, ativos, data_inicial, data_limite)
        if not all(tabela.equals(salva) for tabela, salva in zip([precos, retornos], salvos)):
            raise ValueError("Os CSVs não preservaram a precisão dos dados calculados.")
        for caminho, nome in zip(caminhos, nomes):
            caminho.replace(pasta / nome)
    return salvos


def principal():
    """Executa somente a preparação de dados da Etapa 01, sem alterar o sistema atual."""
    try:
        # Reutiliza o corte existente: sexta-feira só é aceita após 19h de São Paulo.
        data_limite = calcular_data_limite()
        print(f"Período solicitado: {DATA_INICIAL:%Y-%m-%d} até {data_limite:%Y-%m-%d} (limite exclusivo).")
        historicos, repeticoes = coletar_historicos(ATIVOS, DATA_INICIAL, data_limite)
        series, informacoes = [], {}
        for codigo, historico in historicos.items():
            serie, informacoes[codigo] = agrupar_fechamentos(historico, DATA_INICIAL, data_limite)
            series.append(serie)
        precos = alinhar_datas(consolidar_precos(series), informacoes, repeticoes)
        retornos = calcular_retornos(precos)
        precos, retornos = salvar_resultados(precos, retornos, ATIVOS, DATA_INICIAL, data_limite)

        print("\nAtivos coletados: " + ", ".join(ATIVOS))
        print(f"Período efetivo dos preços (semanas): {precos.index[0]:%Y-%m-%d} a {precos.index[-1]:%Y-%m-%d}.")
        print(f"Semanas finais alinhadas: {len(precos)}; linhas de retornos: {len(retornos)}.")
        print("\nPreços semanais ajustados — cinco primeiros registros:")
        print(precos.head().to_string())
        print("\nRetornos semanais em PORCENTAGEM (%) — cinco primeiros registros:")
        print(retornos.head().to_string())
        print("\nConferência de três semanas usando os preços salvos (todas as ações):")
        for posicao in range(1, 4):
            for ativo in precos.columns:
                anterior, atual = precos[ativo].iloc[posicao - 1:posicao + 1]
                retorno = retornos[ativo].iloc[posicao - 1]
                print(f"  {precos.index[posicao]:%Y-%m-%d} · {ativo}: "
                      f"(({atual:.12g} / {anterior:.12g}) - 1) × 100 = {retorno:.10f}%")
        print("\nValidações aprovadas: preços positivos, datas únicas e comuns, semanas encerradas e consecutivas,")
        print("ausência de NaN/infinitos, fórmula dos retornos e precisão preservada nos CSVs.")
        print("A primeira semana foi mantida nos preços e removida somente dos retornos.")
        print("Nenhum valor foi preenchido artificialmente. Arquivos salvos em: " + str(PASTA_SAIDA))
        return 0
    except Exception as erro:
        print(f"Falha na preparação da carteira: {erro}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(principal())
