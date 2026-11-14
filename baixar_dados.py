"""Coleta preços diários ajustados e os agrupa em semanas, meses ou anos.

Executado diretamente, salva a série semanal da Petrobras em dados/petrobras.csv.
As funções também são usadas pela interface app.py para outros ativos e períodos.
"""

from datetime import date, datetime, timedelta
from math import isfinite
from pathlib import Path
import shutil
import sys
import tempfile

import pandas as pd
import yfinance as yf

# Regras de agregação do pandas para cada frequência (rótulo = fim do período).
FREQUENCIAS = {"Semanal": "W-FRI", "Mensal": "ME", "Anual": "YE"}


def calcular_data_limite(data_final=None):
    """Primeira data NÃO incluída: o dia seguinte à data final, sem passar do pregão encerrado."""
    agora = pd.Timestamp.now(tz="America/Sao_Paulo")
    # Após 19h, o pregão do dia já terminou e pode ser incluído.
    limite = agora.date() + timedelta(days=int(agora.hour >= 19))
    if data_final is not None:
        limite = min(limite, data_final + timedelta(days=1))
    return limite


def baixar_precos_diarios(codigo, data_inicial, data_limite):
    """Baixa preços diários ajustados e marca os pregões com preços inválidos."""
    historico = yf.Ticker(codigo).history(
        start=data_inicial.isoformat(),
        end=data_limite.isoformat(),
        interval="1d",
        auto_adjust=True,
        actions=False,
        keepna=True,
        timeout=30,
    )
    if historico is None or historico.empty:
        raise ValueError(f"O Yahoo Finance não retornou dados para {codigo} no período escolhido.")
    if not {"Close", "High", "Low"}.issubset(historico.columns):
        raise ValueError("A resposta deve conter fechamento, máxima e mínima.")
    if not isinstance(historico.index, pd.DatetimeIndex):
        raise ValueError("A resposta não contém um índice de datas válido.")

    datas = historico.index
    if datas.tz is not None:
        datas = datas.tz_convert("America/Sao_Paulo").tz_localize(None)
    datas = datas.normalize()
    if datas.hasnans or datas.duplicated().any():
        raise ValueError("Há datas ausentes ou duplicadas na resposta.")
    if (datas < pd.Timestamp(data_inicial)).any() or (datas >= pd.Timestamp(data_limite)).any():
        raise ValueError("Há datas fora do período solicitado.")

    diarios = pd.DataFrame({
        "data": datas,
        "fechamento": pd.to_numeric(historico["Close"], errors="coerce").to_numpy(),
        "maxima": pd.to_numeric(historico["High"], errors="coerce").to_numpy(),
        "minima": pd.to_numeric(historico["Low"], errors="coerce").to_numpy(),
    }).sort_values("data").reset_index(drop=True)
    invalidos = pd.Series(False, index=diarios.index)
    for coluna in ["fechamento", "maxima", "minima"]:
        invalidos |= ~diarios[coluna].map(lambda valor: isfinite(valor) and valor > 0)
    invalidos |= (diarios["maxima"] < diarios["minima"])
    # O ajuste pode introduzir diferenças mínimas de ponto flutuante nos limites.
    tolerancia = diarios["fechamento"].abs() * 1e-10
    invalidos |= (diarios["fechamento"] < diarios["minima"] - tolerancia)
    invalidos |= (diarios["fechamento"] > diarios["maxima"] + tolerancia)
    diarios["invalido"] = invalidos
    return diarios


def agregar_periodos(diarios, regra, data_limite):
    """Agrega na frequência escolhida e calcula retorno e amplitude percentuais."""
    diarios = diarios.set_index("data")
    precos = diarios[["fechamento", "maxima", "minima"]].copy()
    precos.loc[diarios["invalido"], :] = float("nan")
    reamostragem = {"rule": regra, "closed": "right", "label": "right"}
    dados = precos.resample(**reamostragem).agg({"fechamento": "last", "maxima": "max", "minima": "min"})
    # Período com algum pregão inválido (máximo de verdadeiro/falso = "algum verdadeiro").
    periodos_invalidos = diarios["invalido"].astype(int).resample(**reamostragem).max().fillna(0) > 0

    # O período só entra depois de encerrado (ex.: a sexta, o fim do mês ou do ano).
    em_andamento = dados.index >= pd.Timestamp(data_limite)
    dados = dados.loc[~em_andamento].copy()
    # Um preço diário explicitamente inválido impede afirmar os extremos do período.
    dados.loc[periodos_invalidos.reindex(dados.index, fill_value=False), :] = float("nan")
    incompletos = dados.isna().any(axis=1)
    if int((~incompletos).sum()) < 2:
        raise ValueError("São necessários pelo menos dois períodos encerrados com preços válidos.")

    dados["variacao_percentual"] = dados["fechamento"].pct_change(fill_method=None) * 100
    fechamento_anterior = dados["fechamento"].shift(1)
    dados["amplitude_percentual"] = (dados["maxima"] - dados["minima"]) / fechamento_anterior * 100
    # Sem preenchimento: um período ausente não vira retorno de vários períodos.
    sem_referencia = dados["fechamento"].notna() & fechamento_anterior.isna()
    dados = dados.dropna().reset_index()
    if dados.empty:
        raise ValueError("Não há períodos com retorno e amplitude definidos.")
    for coluna in ["variacao_percentual", "amplitude_percentual"]:
        if not dados[coluna].map(isfinite).all():
            raise ValueError(f"O cálculo produziu um valor inválido em {coluna}.")
    informacoes = {
        "em_andamento": int(em_andamento.sum()),
        "incompletos": list(incompletos.index[incompletos].strftime("%Y-%m-%d")),
        "sem_referencia": list(sem_referencia.index[sem_referencia].strftime("%Y-%m-%d")),
    }
    return dados, informacoes


def principal():
    """Baixa, valida e salva os dados semanais da Petrobras; retorna 1 se falhar."""
    try:
        # Nunca consulta além de 2026, conforme o escopo original do projeto.
        data_limite = min(calcular_data_limite(), date(2027, 1, 1))
        diarios = baixar_precos_diarios("PETR4.SA", date(2020, 1, 1), data_limite)
        invalidos = diarios["invalido"]
        if invalidos.any():
            datas_invalidas = ", ".join(diarios.loc[invalidos, "data"].dt.strftime("%Y-%m-%d"))
            print(f"Descartados {int(invalidos.sum())} pregões com preços inválidos: {datas_invalidas}.")
        dados, informacoes = agregar_periodos(diarios, FREQUENCIAS["Semanal"], data_limite)
        print(f"Semanas ainda não encerradas excluídas: {informacoes['em_andamento']}")
        if informacoes["incompletos"]:
            print("Semanas sem preços completos descartadas: " + ", ".join(informacoes["incompletos"]))
        print("Semanas sem fechamento anterior excluídas: " + ", ".join(informacoes["sem_referencia"]))

        pasta_dados = Path(__file__).resolve().parent / "dados"
        pasta_dados.mkdir(exist_ok=True)
        destino = pasta_dados / "petrobras.csv"
        # Só substitui o CSV depois de gravar a nova coleta por completo.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", suffix=".tmp",
            prefix="petrobras_", dir=pasta_dados, delete=False,
        ) as temporario:
            dados.to_csv(temporario, index=False, date_format="%Y-%m-%d")
        if destino.exists():
            identificador = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            copia = pasta_dados / f"petrobras_anterior_{identificador}.csv"
            if copia.exists():
                raise FileExistsError("Já existe uma cópia com este identificador.")
            shutil.copy2(destino, copia)
        Path(temporario.name).replace(destino)

        print(f"Registros recebidos do Yahoo: {len(diarios)}")
        print(f"Pregões com preços inválidos descartados: {int(invalidos.sum())}")
        print(f"Pregões válidos coletados: {int((~invalidos).sum())}")
        print(f"Semanas salvas com retorno e amplitude: {len(dados)}")
        print(f"Data inicial do CSV: {dados['data'].iloc[0]:%Y-%m-%d}")
        print(f"Data final do CSV: {dados['data'].iloc[-1]:%Y-%m-%d}")
        print(f"Arquivo salvo: {destino}")
        return 0
    except Exception as erro:
        print(f"Falha na coleta da Petrobras: {erro}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(principal())
