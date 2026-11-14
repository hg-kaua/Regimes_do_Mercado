"""Painel interativo de regimes de mercado: K-Means e Cadeias de Markov.

Execute com: streamlit run app.py
Os cálculos vêm dos scripts do projeto (baixar_dados.py, analisar_dados.py e
calcular_markov.py); este arquivo apenas organiza a execução e os gráficos.
"""

from datetime import date
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analisar_dados import MINIMO_OBSERVACOES, NORMALIZADAS, VARIAVEIS, normalizar_e_agrupar
from baixar_dados import FREQUENCIAS, agregar_periodos, baixar_precos_diarios, calcular_data_limite
from calcular_markov import estimar_markov
from dados_carteiras import ESTRATEGIAS, ajustar_capital
from motor_carteiras import analisar_carteiras, validar_parametros, adicionar_personalizada, adicionar_maximo_sharpe
from fronteira_eficiente import calcular_fronteira
from benchmarks import BENCHMARKS, baixar_cdi, baixar_ibovespa, calcular_benchmarks
from backtest_carteira import CAPITAL_INICIAL
from walk_forward import (
    MINIMO_SEMANAS_AVALIACAO, MINIMO_SEMANAS_TREINO, SEMANAS_AVALIACAO_PADRAO, SEMANAS_TREINO_PADRAO,
    executar_walk_forward,
)
from otimizar_carteira import PESO_MAXIMO

ACOES = {
    "PETR4 — Petrobras": "PETR4.SA",
    "VALE3 — Vale": "VALE3.SA",
    "ITUB4 — Itaú Unibanco": "ITUB4.SA",
    "BBAS3 — Banco do Brasil": "BBAS3.SA",
    "BBDC4 — Bradesco": "BBDC4.SA",
    "ABEV3 — Ambev": "ABEV3.SA",
}
OUTRO_CODIGO = "Outro código da B3"
NOMES_GRUPOS = ["Grupo 1", "Grupo 2", "Grupo 3"]
CORES_GRUPOS = ["#2a78d6", "#eb6834", "#1baf7a"]  # Azul, laranja e verde-água.
# Expressões já flexionadas para cada frequência (semana é feminino; mês e ano, masculinos).
TEXTOS_PERIODO = {
    "Semanal": {"cada": "semana", "do": "da semana", "do_proximo": "da próxima semana",
                "do_atual": "da semana atual", "o_atual": "a semana atual", "o_proximo": "a próxima semana"},
    "Mensal": {"cada": "mês", "do": "do mês", "do_proximo": "do próximo mês",
               "do_atual": "do mês atual", "o_atual": "o mês atual", "o_proximo": "o próximo mês"},
    "Anual": {"cada": "ano", "do": "do ano", "do_proximo": "do próximo ano",
              "do_atual": "do ano atual", "o_atual": "o ano atual", "o_proximo": "o próximo ano"},
}
MINIMO_SAIDAS = 10  # Abaixo disso, a linha da matriz recebe um aviso de cautela.
NOMES_ESTRATEGIAS = {"igualitaria": "Igualitária", "minima_volatilidade": "Mínima volatilidade",
                     "personalizada": "Personalizada", "maximo_sharpe": "Máximo Sharpe"}
CORES_ESTRATEGIAS = {"igualitaria": CORES_GRUPOS[0], "minima_volatilidade": CORES_GRUPOS[1],
                     "personalizada": CORES_GRUPOS[2], "maximo_sharpe": "#aa82f0"}
# Dourado mais escuro que o amarelo do retorno-alvo: legível nos temas claro e escuro.
ESTILOS_BENCHMARKS = {"ibovespa": ("#d4a017", "dash"), "cdi": ("#9aa0a6", "dot")}
CAPITAL_PADRAO = 100_000.00
COLUNAS_REALIZADAS = {
    "retorno_acumulado_decimal": "Retorno acumulado",
    "retorno_anualizado_realizado_decimal": "Retorno anualizado",
    "volatilidade_anualizada_realizada_decimal": "Volatilidade anualizada",
    "drawdown_maximo_decimal": "Drawdown máximo",
}
LEGENDA_ABAIXO = dict(orientation="h", yanchor="top", y=-0.22, x=0)


def hoje_sp():
    """Data de hoje em São Paulo, referência de todos os limites de data."""
    return pd.Timestamp.now(tz="America/Sao_Paulo").date()


@st.cache_data(ttl=3600, show_spinner=False)
def baixar_com_cache(codigo, data_inicial, data_limite):
    """Guarda a consulta ao Yahoo Finance por uma hora."""
    return baixar_precos_diarios(codigo, data_inicial, data_limite)


@st.cache_data(ttl=3600, show_spinner=False)
def executar_analise(codigo, data_inicial, data_final, frequencia):
    """Executa o fluxo completo para uma combinação de ativo, período e frequência.

    O resultado fica em cache: trocar de aba ou de grupo não refaz os cálculos.
    """
    data_limite = calcular_data_limite(data_final)
    diarios = baixar_com_cache(codigo, data_inicial, data_limite)
    regra = FREQUENCIAS[frequencia]
    dados, informacoes = agregar_periodos(diarios, regra, data_limite)
    resultado = {
        "dados": dados, "informacoes": informacoes, "pregoes": len(diarios),
        "pregoes_invalidos": int(diarios["invalido"].sum()), "modelo": None, "motivo": None,
    }
    if len(dados) < MINIMO_OBSERVACOES:
        resultado["motivo"] = (
            f"Há apenas {len(dados)} períodos completos na frequência {frequencia.lower()}; "
            f"são necessários pelo menos {MINIMO_OBSERVACOES} para executar K-Means e Markov. "
            "Amplie o período ou escolha uma frequência menor."
        )
        return resultado
    try:
        analise, resumo, centroides, _ = normalizar_e_agrupar(dados)
    except ValueError as erro:
        resultado["motivo"] = f"Não foi possível executar o K-Means: {erro}"
        return resultado
    contagens, probabilidades, contagens_teste, probabilidades_teste = estimar_markov(analise, regra)
    resultado["modelo"] = {
        "analise": analise, "resumo": resumo, "centroides": centroides,
        "contagens": contagens, "probabilidades": probabilidades, "contagens_teste": contagens_teste,
        "probabilidades_teste": probabilidades_teste,
    }
    return resultado


def validar_filtros(escolha, codigo_digitado, data_inicial, data_final):
    """Retorna o código do Yahoo e uma lista de erros encontrados nos filtros."""
    erros = []
    hoje = hoje_sp()
    codigo = ACOES.get(escolha, "")
    if escolha == OUTRO_CODIGO:
        codigo = codigo_digitado.strip().upper()
        if codigo and not codigo.endswith(".SA"):
            codigo += ".SA"
        if not re.fullmatch(r"[A-Z0-9]{4,6}\.SA", codigo):
            erros.append("Digite um código válido da B3, por exemplo WEGE3 ou MGLU3.")
    if data_inicial >= data_final:
        erros.append("A data inicial deve ser anterior à data final.")
    if data_final > hoje:
        erros.append("A data final não pode estar no futuro.")
    return codigo, erros


def formatar_percentual(valor):
    return f"{valor:.2f}%".replace(".", ",")


def formatar_diferenca(valor):
    """Diferença em pontos percentuais com sinal, por exemplo +2,60 p.p."""
    if pd.isna(valor):
        return "Sem comparação"
    return f"{valor:+.2f}".replace(".", ",") + " p.p."


def formatar_moeda(valor):
    """Formato brasileiro: R$ 1.234,56."""
    texto = f"{abs(valor):,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{'-' if valor < 0 else ''}R$ {texto}"


def formatar_percentual_ou_vazio(valor):
    return "Sem dados" if pd.isna(valor) else formatar_percentual(valor)


def formatar_sharpe(retorno, volatilidade, taxa):
    """Sharpe estimado anual com duas casas; indefinido sem risco."""
    return f"{(retorno - taxa) / volatilidade:.2f}".replace(".", ",") if volatilidade > 1e-12 else "Indefinido"


def linhas_desempenho(patrimonio_final, indicadores):
    """Linhas da tabela de desempenho realizado (mesma ordem em todas as tabelas)."""
    return {"Patrimônio final": formatar_moeda(patrimonio_final),
            **{rotulo: formatar_percentual(indicadores[coluna] * 100) for coluna, rotulo in COLUNAS_REALIZADAS.items()}}


def grafico_patrimonio(series, capital, titulo_x="Data (fim de cada semana)"):
    """Curvas de patrimônio em R$. series: lista de (nome, Series, cor, tracejado, espessura)."""
    figura = go.Figure()
    for nome, serie, cor, tracejado, espessura in series:
        figura.add_trace(go.Scatter(
            x=serie.index, y=serie, mode="lines", name=nome,
            line=dict(color=cor, dash=tracejado, width=espessura),
            hovertemplate="%{fullData.name}: R$ %{y:,.2f}<extra></extra>",
        ))
    figura.add_hline(y=capital, line=dict(color="gray", dash="dash", width=1))
    figura.update_layout(
        height=480, separators=",.", hovermode="x unified", margin=dict(t=30, b=40),
        xaxis=dict(title=titulo_x, hoverformat="%d/%m/%Y", tickformat="%m/%Y"),
        yaxis=dict(title="Patrimônio", tickprefix="R$ ", tickformat=",.0f"),
        # Abaixo do gráfico: com várias séries, a legenda não cobre a barra de ferramentas.
        legend=LEGENDA_ABAIXO,
    )
    return figura


@st.cache_data(ttl=3600, show_spinner=False, max_entries=20)
def analisar_carteiras_com_cache(ativos, inicio, fim, maximo, limite_coleta):
    """Capital não entra na chave; o corte de pregão impede cache de semana aberta."""
    return analisar_carteiras(list(ativos), inicio, fim, maximo)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=20)
def analisar_sharpe_com_cache(carteiras, taxa_anual):
    """Inclui os dados efetivos e a taxa na chave, evitando misturar versões da coleta."""
    return adicionar_maximo_sharpe(carteiras, taxa_anual)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=20)
def calcular_fronteira_com_cache(treinamento, maximo):
    """Só treinamento e teto: capital e taxa não alteram a geometria da fronteira."""
    return calcular_fronteira(treinamento, maximo)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=20)
def ibovespa_com_cache(inicio, limite):
    """Falhas não entram no cache: uma nova submissão tenta a fonte novamente."""
    return baixar_ibovespa(inicio, limite)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=20)
def cdi_com_cache(inicio, fim):
    return baixar_cdi(inicio, fim)


def exibir_fronteira(fronteira, carteiras, taxa_anual):
    """Exibe soluções calculadas, sem interpolar pesos ou suavizar a curva."""
    st.subheader("Fronteira eficiente")
    st.caption("Estimativas anuais do treinamento · A taxa livre de risco afeta Máximo Sharpe, não a fronteira.")
    if fronteira["falhas"]:
        st.warning(f"{len(fronteira['falhas'])} alvo(s) não resolvido(s). As lacunas não são conectadas no gráfico.")
        with st.expander("Falhas da fronteira"):
            for falha in fronteira["falhas"]:
                st.write(falha)
    pontos = fronteira["pontos"]
    opcoes = list(fronteira["pesos"].index)
    if len(opcoes) == 1:
        st.info("As restrições e estimativas resultam em um único ponto disponível.")
    selecionado = st.selectbox(
        "Retorno-alvo anual esperado", opcoes,
        format_func=lambda indice: formatar_percentual(pontos.loc[indice, "retorno_alvo"] * 100),
        key="alvo_fronteira", help="Selecione entre os alvos efetivamente resolvidos pelo otimizador.")
    figura = go.Figure()

    def hover_sharpe(retornos, volatilidades):
        return [formatar_sharpe(r, v, taxa_anual) for r, v in zip(retornos, volatilidades)]

    hover = ("Volatilidade: %{x:.2f}% a.a.<br>Retorno esperado: %{y:.2f}% a.a."
             "<br>Sharpe estimado: %{customdata}<extra>%{fullData.name}</extra>")
    estimativas = carteiras["resumo_otimizacao"]
    pontos_grafico = pontos.copy()
    if "maximo_sharpe" in estimativas.index:
        # A solução tangente também pertence à fronteira. Incluí-la evita que
        # a ligação linear da malha grossa passe abaixo deste ponto real.
        sharpe = estimativas.loc["maximo_sharpe"]
        retorno_sharpe = sharpe["retorno_medio_anualizado_aritmetico_decimal"]
        pontos_grafico = pd.concat([pontos_grafico, pd.DataFrame([{
            "retorno_alvo": retorno_sharpe, "retorno_esperado": retorno_sharpe,
            "volatilidade": sharpe["volatilidade_anualizada_decimal"],
        }])], ignore_index=True).sort_values("retorno_alvo", kind="stable")
    figura.add_trace(go.Scatter(
        x=pontos_grafico["volatilidade"] * 100, y=pontos_grafico["retorno_esperado"] * 100,
        mode="lines+markers", name="Fronteira eficiente", connectgaps=False,
        line=dict(color="#1baf7a", width=2, shape="linear"), marker=dict(size=5),
        customdata=hover_sharpe(pontos_grafico["retorno_esperado"], pontos_grafico["volatilidade"]), hovertemplate=hover))
    for nome, coluna in [("minima_volatilidade", "peso_otimizado_decimal"),
                         ("maximo_sharpe", "peso_maximo_sharpe_decimal"),
                         ("igualitaria", "peso_igualitario_decimal")]:
        if nome not in estimativas.index or coluna not in carteiras["pesos"]:
            continue
        if (carteiras["pesos"][coluna] > carteiras["parametros"]["peso_maximo"] + 1e-7).any():
            continue
        retorno = estimativas.loc[nome, "retorno_medio_anualizado_aritmetico_decimal"]
        vol = estimativas.loc[nome, "volatilidade_anualizada_decimal"]
        figura.add_trace(go.Scatter(
            x=[vol * 100], y=[retorno * 100], mode="markers", name=NOMES_ESTRATEGIAS[nome],
            marker=dict(color=CORES_ESTRATEGIAS[nome], size=12, symbol="diamond"),
            customdata=hover_sharpe([retorno], [vol]), hovertemplate=hover))
    ponto = pontos.loc[selecionado]
    figura.add_trace(go.Scatter(
        x=[ponto.volatilidade * 100], y=[ponto.retorno_esperado * 100], name="Retorno-alvo",
        mode="markers", marker=dict(color="#f4c95d", size=15, symbol="circle-open", line=dict(width=2)),
        customdata=hover_sharpe([ponto.retorno_esperado], [ponto.volatilidade]), hovertemplate=hover))
    figura.update_layout(
        height=520, separators=",.", margin=dict(t=15, b=30, l=5, r=10),
        xaxis=dict(title="Volatilidade estimada (% a.a.)", ticksuffix="%", automargin=True),
        yaxis=dict(title="Retorno esperado (% a.a.)", ticksuffix="%", automargin=True),
        legend=dict(orientation="h", yanchor="top", y=-0.25, x=0),
    )
    st.plotly_chart(figura, width="stretch", key="grafico_fronteira")
    st.caption(f"Alvo selecionado: retorno esperado {formatar_percentual(ponto.retorno_esperado * 100)} a.a. · "
               f"Volatilidade estimada {formatar_percentual(ponto.volatilidade * 100)} a.a.")
    pesos_alvo = fronteira["pesos"].loc[selecionado]
    st.dataframe(pd.DataFrame({"Ativo": pesos_alvo.index,
                              "Peso no retorno-alvo": [formatar_percentual(p * 100) for p in pesos_alvo]}),
                 hide_index=True, width="stretch")


@st.fragment
def exibir_carteiras():
    """Formulário independente dos filtros de regimes; resultados da última submissão."""
    hoje_carteiras = hoje_sp()
    st.subheader("Configuração da simulação")
    with st.form("formulario_carteiras"):
        selecionados = st.multiselect(
            "Ativos da B3", list(ACOES),
            default=[nome for nome, codigo in ACOES.items() if codigo != "BBDC4.SA"],
            max_selections=12, key="ativos_carteiras")
        outros = st.text_input("Outros códigos da B3", placeholder="Ex.: WEGE3, RENT3",
                               help="Separe por vírgula, ponto e vírgula ou espaço. Total de 3 a 12 ativos.",
                               key="outros_carteiras")
        coluna_inicio, coluna_fim = st.columns(2)
        inicio = coluna_inicio.date_input("Data inicial", date(2020, 1, 1), min_value=date(2000, 1, 1),
                                           max_value=hoje_carteiras, format="DD/MM/YYYY", key="inicio_carteiras")
        fim = coluna_fim.date_input("Data final", hoje_carteiras, min_value=date(2000, 1, 1),
                                     max_value=hoje_carteiras, format="DD/MM/YYYY", key="fim_carteiras")
        coluna_limite, coluna_capital = st.columns(2)
        maximo_percentual = coluna_limite.number_input(
            "Limite máximo por ativo (%)", min_value=1.0, max_value=100.0, value=PESO_MAXIMO * 100,
            step=5.0, key="limite_carteiras")
        capital_informado = coluna_capital.number_input(
            "Capital inicial (R$)", min_value=1_000.0, max_value=1_000_000_000.0,
            value=CAPITAL_PADRAO, step=10_000.0, format="%.2f", key="capital_carteiras")
        taxa_percentual = st.number_input(
            "Taxa livre de risco anual (%) — hipótese da simulação", min_value=-99.0, max_value=1000.0,
            value=0.0, step=0.5, format="%.2f", key="taxa_livre_risco_carteiras",
            help="Informe sua hipótese anual. O valor inicial de 0% não representa a taxa vigente do mercado.")
        st.caption("Semanal · Treinamento/teste 80/20 · Buy and hold · 3 a 12 ativos")
        enviado_carteiras = st.form_submit_button("Analisar carteira", type="primary", width="stretch")
    st.caption("Mudar só o capital reaproveita a análise.")

    if enviado_carteiras:
        anterior = st.session_state.pop("analise_carteiras", None)
        st.session_state.pop("erro_carteiras", None)
        try:
            ativos = [ACOES[nome] for nome in selecionados]
            if outros.strip():
                ativos += re.split(r"[,;\s]+", outros.strip())
            codigos, inicio, fim, maximo = validar_parametros(ativos, inicio, fim, maximo_percentual / 100)
            chave = (tuple(codigos), inicio, fim, maximo, calcular_data_limite(fim))
            taxa_anual = taxa_percentual / 100.0
            # Mesma análise com outro capital: reaproveita tudo; o capital só reescala valores em R$.
            mesma_analise = (anterior is not None and anterior["chave"] == chave
                             and anterior.get("taxa_anual") == taxa_anual and anterior["capital"] != capital_informado)
            if mesma_analise:
                estado = {**anterior, "capital": capital_informado}
            else:
                with st.spinner("Coletando preços, otimizando e avaliando as carteiras..."):
                    carteiras = analisar_carteiras_com_cache(*chave)
                estado = {"chave": chave, "resultado": carteiras, "capital": capital_informado,
                          "taxa_anual": taxa_anual, "resultado_sharpe": None, "erro_sharpe": None,
                          "fronteira": None, "erro_fronteira": None}
                try:
                    with st.spinner("Calculando Máximo Sharpe..."):
                        estado["resultado_sharpe"] = analisar_sharpe_com_cache(carteiras, taxa_anual)
                except (ValueError, RuntimeError) as erro:
                    estado["erro_sharpe"] = f"Máximo Sharpe indisponível: {erro}"
                try:
                    with st.spinner("Calculando a fronteira eficiente..."):
                        estado["fronteira"] = calcular_fronteira_com_cache(carteiras["treinamento"], maximo)
                except (ValueError, RuntimeError) as erro:
                    estado["erro_fronteira"] = f"Fronteira indisponível: {erro}"
                # Benchmarks nas mesmas datas do teste; falhas ficam registradas sem interromper a análise.
                with st.spinner("Coletando Ibovespa e CDI..."):
                    estado["benchmarks"] = calcular_benchmarks(
                        carteiras["curva"].index, chave[4], ibovespa=ibovespa_com_cache, cdi=cdi_com_cache)
            # Preserva a seleção em mudanças de capital/taxa; reinicia em outro universo.
            if anterior is None or anterior["chave"] != chave:
                st.session_state.pop("alvo_fronteira", None)
            st.session_state.analise_carteiras = estado
        except Exception as erro:
            st.session_state.erro_carteiras = f"Não foi possível analisar a carteira: {erro}"

    if "erro_carteiras" in st.session_state:
        st.error(st.session_state.erro_carteiras)
        return
    if "analise_carteiras" not in st.session_state:
        st.info("Selecione os ativos e o período e clique em Analisar carteira.")
        return
    analise_atual = st.session_state.analise_carteiras
    carteiras = analise_atual.get("resultado_sharpe") or analise_atual["resultado"]
    capital, taxa_anual = analise_atual["capital"], analise_atual["taxa_anual"]
    if analise_atual.get("erro_sharpe"):
        st.warning(analise_atual["erro_sharpe"])
    ativos_manuais = tuple(carteiras["pesos"].index)
    st.subheader("Carteira personalizada")
    st.caption("Pesos de 0% a 100% · O limite do otimizador não se aplica à Personalizada.")
    igualar = st.button("Distribuir igualmente", key="igualar_personalizada")
    if (st.session_state.get("ativos_personalizada") != ativos_manuais or igualar
            or any(f"peso_manual_{ativo}" not in st.session_state for ativo in ativos_manuais)):
        for chave_peso in list(st.session_state):
            if chave_peso.startswith("peso_manual_"):
                del st.session_state[chave_peso]
        for ativo in ativos_manuais:
            st.session_state[f"peso_manual_{ativo}"] = 100.0 / len(ativos_manuais)
        st.session_state.ativos_personalizada = ativos_manuais
    pesos_manuais = {}
    colunas_pesos = st.columns(3)
    for i, ativo in enumerate(ativos_manuais):
        pesos_manuais[ativo] = colunas_pesos[i % 3].number_input(
            f"{ativo} (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.6f", key=f"peso_manual_{ativo}")
    st.caption(f"Soma: {formatar_percentual(sum(pesos_manuais.values()))}")
    estrategias = dict(ESTRATEGIAS)
    try:
        carteiras = adicionar_personalizada(carteiras, pesos_manuais)
        estrategias["personalizada"] = "peso_personalizado_decimal"
    except (ValueError, AssertionError) as erro:
        st.error(f"Personalizada indisponível: {erro}")
    if "peso_maximo_sharpe_decimal" in carteiras["pesos"]:
        estrategias["maximo_sharpe"] = "peso_maximo_sharpe_decimal"
    pesos, periodo, resumo = carteiras["pesos"], carteiras["periodo"], carteiras["resumo"]
    maximo = carteiras["parametros"]["peso_maximo"]

    def data(texto):
        return f"{pd.Timestamp(texto):%d/%m/%Y}"

    st.subheader("Resumo da simulação")
    parametros_simulacao = pd.DataFrame({"": [
        "Ativos", "Treinamento", "Teste", "Formação das carteiras", "Capital por estratégia", "Estratégias", "Execução",
    ], "Valor": [
        ", ".join(pesos.index),
        f"{data(periodo['inicio_treinamento'])} a {data(periodo['fim_treinamento'])} · {periodo['semanas_treinamento']} semanas",
        f"{data(periodo['inicio_teste'])} a {data(periodo['fim_teste'])} · {periodo['semanas_teste']} semanas",
        data(periodo["fim_treinamento"]),
        formatar_moeda(capital),
        " · ".join(NOMES_ESTRATEGIAS[nome] for nome in estrategias)
        + f" · Limite das otimizadas: {formatar_percentual(maximo * 100)} por ativo",
        "Buy and hold semanal, sem rebalanceamento",
    ]}).set_index("")
    # st.table quebra o texto em telas estreitas; st.dataframe cortaria os valores.
    st.table(parametros_simulacao)

    st.subheader("Composição")
    coluna_grafico, coluna_tabela = st.columns([1, 1])
    with coluna_grafico:
        figura = go.Figure()
        for nome, coluna in estrategias.items():
            figura.add_trace(go.Bar(
                y=pesos.index, x=pesos[coluna] * 100, orientation="h", name=NOMES_ESTRATEGIAS[nome],
                marker_color=CORES_ESTRATEGIAS[nome], customdata=pesos[coluna] * capital,
                text=[formatar_percentual(valor * 100) for valor in pesos[coluna]], textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}<br>Peso: %{x:.2f}%<br>Valor: R$ %{customdata:,.2f}<extra>%{fullData.name}</extra>",
            ))
        figura.update_layout(
            barmode="group", height=max(440, len(pesos) * 90), separators=",.", margin=dict(t=70, b=40),
            xaxis=dict(title="Peso (%)", range=[0, max(maximo, pesos.max().max()) * 100 + 8]),
            yaxis=dict(autorange="reversed"), legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        )
        st.plotly_chart(figura, width="stretch")
    with coluna_tabela:
        composicao = pd.DataFrame({"Ativo": list(pesos.index) + ["Total"]})
        for nome, coluna in estrategias.items():
            valores = list(pesos[coluna]) + [pesos[coluna].sum()]
            composicao[NOMES_ESTRATEGIAS[nome]] = [formatar_percentual(valor * 100) for valor in valores]
        st.dataframe(composicao, hide_index=True, width="stretch")

    st.subheader("Estimativas no treinamento")
    st.caption(f"Anualização: média e covariância semanais × 52 · "
               f"Taxa livre de risco: {formatar_percentual(taxa_anual * 100)} a.a.")
    estimativas = carteiras["resumo_otimizacao"]
    colunas_estimadas = {
        "retorno_medio_semanal_decimal": "Retorno médio semanal",
        "retorno_medio_anualizado_aritmetico_decimal": "Retorno médio anualizado (aritmético)",
        "volatilidade_semanal_decimal": "Volatilidade semanal",
        "volatilidade_anualizada_decimal": "Volatilidade anualizada",
    }
    tabela_estimadas = pd.DataFrame({NOMES_ESTRATEGIAS[nome]: {
        rotulo: formatar_percentual(estimativas.loc[nome, coluna] * 100)
        for coluna, rotulo in colunas_estimadas.items()
    } for nome in estrategias}).rename_axis("Indicador")
    tabela_estimadas.loc["Sharpe estimado (anual)"] = [
        formatar_sharpe(estimativas.loc[nome, "retorno_medio_anualizado_aritmetico_decimal"],
                        estimativas.loc[nome, "volatilidade_anualizada_decimal"], taxa_anual)
        for nome in estrategias]
    st.dataframe(tabela_estimadas, width="stretch")

    if analise_atual.get("erro_fronteira"):
        st.warning(analise_atual["erro_fronteira"])
    if analise_atual.get("fronteira") is not None:
        exibir_fronteira(analise_atual["fronteira"], carteiras, taxa_anual)

    st.subheader("Desempenho no teste")
    linha_base = resumo.loc["igualitaria"]
    st.caption(f"{data(linha_base['data_formacao'])} a {data(linha_base['data_final'])} · "
               f"{int(linha_base['semanas_teste'])} semanas · Mesmo capital por estratégia")
    patrimonio = ajustar_capital(carteiras, capital)
    referencias = analise_atual.get("benchmarks") or {"curva": pd.DataFrame(), "resumo": pd.DataFrame(),
                                                       "indisponiveis": {}, "capital": capital}
    for nome, motivo in referencias["indisponiveis"].items():
        st.warning(f"{BENCHMARKS[nome]} indisponível: {motivo}. As carteiras não foram afetadas.")
    disponiveis = [nome for nome in BENCHMARKS if nome in referencias["resumo"].index]
    benchmarks_visiveis = st.multiselect(
        "Benchmarks", disponiveis, default=disponiveis, format_func=BENCHMARKS.get,
        # A chave depende das opções: uma fonte que volta ou cai não deixa seleção inválida no estado.
        key="benchmarks_carteiras_" + "_".join(disponiveis), help="Ibovespa (índice de retorno total) e CDI no mesmo período e capital.")
    for nome in benchmarks_visiveis:
        patrimonio[nome] = referencias["curva"][nome] * (capital / referencias["capital"])
    tabela_realizadas = pd.DataFrame({NOMES_ESTRATEGIAS[nome]: linhas_desempenho(patrimonio[nome].iloc[-1], resumo.loc[nome])
                                      for nome in estrategias}).rename_axis("Indicador")
    for nome in benchmarks_visiveis:
        tabela_realizadas[BENCHMARKS[nome]] = list(
            linhas_desempenho(patrimonio[nome].iloc[-1], referencias["resumo"].loc[nome]).values())
    st.dataframe(tabela_realizadas, width="stretch")

    st.subheader("Evolução patrimonial")
    # Benchmarks tracejados e mais finos: referências, não estratégias.
    st.plotly_chart(grafico_patrimonio(
        [(NOMES_ESTRATEGIAS[nome], patrimonio[nome], CORES_ESTRATEGIAS[nome], "solid", 2) for nome in estrategias]
        + [(BENCHMARKS[nome], patrimonio[nome], *ESTILOS_BENCHMARKS[nome], 1.6) for nome in benchmarks_visiveis],
        capital), width="stretch")
    fontes = (" Ibovespa: ^BVSP, índice de retorno total da B3, sem custos de replicação. "
              "CDI: BCB/SGS 12, capitalizado por dia útil (não é a Selic).") if disponiveis else ""
    st.caption("Simulação histórica com preços ajustados; sem custos, impostos ou rebalanceamento." + fontes
               + " Resultados passados não garantem retornos futuros.")

    pesos_fixos = pesos["peso_personalizado_decimal"] if "personalizada" in estrategias else None
    exibir_walk_forward(carteiras, capital, taxa_anual, maximo, pesos_fixos)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=10)
def walk_forward_com_cache(retornos, treino, avaliacao, maximo, taxa, pesos_fixos):
    """Calculado com o capital de referência; a interface só reescala os valores em R$."""
    return executar_walk_forward(retornos, treino, avaliacao, maximo, taxa, CAPITAL_INICIAL,
                                 None if pesos_fixos is None else dict(pesos_fixos))


def exibir_walk_forward(carteiras, capital, taxa_anual, maximo, pesos_fixos):
    """Metodologia alternativa; o backtest 80/20 acima permanece como está."""
    st.subheader("Validação walk-forward")
    st.caption("Pesos reestimados a cada janela só com semanas anteriores; rebalanceamento no início da janela "
               "e buy and hold dentro dela · Mesmas datas para todas as estratégias · Sem custos de transação · "
               "Taxa livre de risco informada aplicada a todas as janelas.")
    with st.form("formulario_walk_forward"):
        coluna_treino, coluna_avaliacao = st.columns(2)
        treino = coluna_treino.number_input("Treino móvel (semanas)", min_value=MINIMO_SEMANAS_TREINO, max_value=520,
                                            value=SEMANAS_TREINO_PADRAO, step=13, key="wf_treino")
        avaliacao = coluna_avaliacao.number_input("Janela de avaliação (semanas)", min_value=MINIMO_SEMANAS_AVALIACAO,
                                                  max_value=104, value=SEMANAS_AVALIACAO_PADRAO, step=1, key="wf_avaliacao")
        executar = st.form_submit_button("Executar walk-forward", width="stretch")
    fixos = None if pesos_fixos is None else tuple(pesos_fixos.items())
    retornos = carteiras["retornos_percentuais"]
    chave = (tuple(retornos.columns), retornos.index[0], retornos.index[-1], maximo, taxa_anual, fixos,
             int(treino), int(avaliacao))
    if executar:
        try:
            with st.spinner("Reestimando os pesos em cada janela..."):
                resultado = walk_forward_com_cache(retornos, int(treino), int(avaliacao), maximo, taxa_anual, fixos)
            st.session_state.walk_forward = {"chave": chave, "resultado": resultado}
        except (ValueError, RuntimeError, AssertionError) as erro:
            st.session_state.walk_forward = {"chave": chave, "erro": str(erro)}
    estado = st.session_state.get("walk_forward")
    if estado is None:
        st.info("Defina as janelas e clique em Executar walk-forward.")
        return
    if estado["chave"] != chave:
        st.info("Os parâmetros da análise mudaram. Clique em Executar walk-forward para atualizar.")
        return
    if "erro" in estado:
        st.error(f"Walk-forward indisponível: {estado['erro']}")
        return
    resultado = estado["resultado"]
    for nome, motivo in resultado["indisponiveis"].items():
        st.warning(f"{NOMES_ESTRATEGIAS[nome]} indisponível no walk-forward: {motivo}")
    for nome, falhas in resultado["falhas"].items():
        if nome not in resultado["indisponiveis"]:
            st.warning(f"{NOMES_ESTRATEGIAS[nome]}: {len(falhas)} janela(s) sem nova estimativa; posições mantidas.")
    janelas, parametros = resultado["janelas"], resultado["parametros"]
    st.caption(f"{len(janelas)} janelas de {parametros['semanas_avaliacao']} semanas · "
               f"{janelas['inicio_avaliacao'].iloc[0]:%d/%m/%Y} a {janelas['fim_avaliacao'].iloc[-1]:%d/%m/%Y} · "
               f"treino móvel de {parametros['semanas_treino']} semanas · "
               f"{parametros['semanas_descartadas_inicio']} semana(s) iniciais sem uso")

    escala = capital / parametros["capital_inicial"]
    curva, resumo, consistencia = resultado["curva"] * escala, resultado["resumo"], resultado["consistencia"]
    nomes = list(curva)
    tabela = pd.DataFrame({NOMES_ESTRATEGIAS[nome]: {
        **linhas_desempenho(curva[nome].iloc[-1], resumo.loc[nome]),
        "Rebalanceamentos": str(int(resumo.loc[nome, "rebalanceamentos"])),
        "Giro médio por rebalanceamento": formatar_percentual(resumo.loc[nome, "giro_medio_rebalanceamento"] * 100),
        "Janelas com retorno positivo": f"{int(consistencia.loc[nome, 'janelas_positivas'])} de {len(janelas)}",
        "Janelas acima da Igualitária": ("—" if nome == "igualitaria" or "janelas_acima_igualitaria" not in consistencia
                                         else f"{int(consistencia.loc[nome, 'janelas_acima_igualitaria'])} de {len(janelas)}"),
    } for nome in nomes}).rename_axis("Indicador")
    st.dataframe(tabela, width="stretch")

    figura = grafico_patrimonio(
        [(NOMES_ESTRATEGIAS[nome], curva[nome], CORES_ESTRATEGIAS[nome], "solid", 2) for nome in nomes],
        capital, "Data (fim de cada semana) · linhas verticais: rebalanceamentos")
    for data_rebalanceamento in janelas["fim_treino"].iloc[1:]:
        figura.add_vline(x=data_rebalanceamento, line=dict(color="gray", width=1, dash="dot"), opacity=0.55)
    st.plotly_chart(figura, width="stretch", key="grafico_walk_forward")

    por_janela = resultado["por_janela"]
    tabela_janelas = pd.DataFrame({
        "Avaliação": [f"{inicio:%d/%m/%Y} – {fim:%d/%m/%Y}"
                      for inicio, fim in zip(por_janela["inicio_avaliacao"], por_janela["fim_avaliacao"])],
        **{NOMES_ESTRATEGIAS[nome]: [formatar_percentual(valor * 100) for valor in por_janela[f"retorno_{nome}"]]
           for nome in nomes},
    }, index=por_janela.index.rename("Janela"))
    st.markdown("**Retorno por janela**")
    st.dataframe(tabela_janelas, width="stretch", height=min(38 + 35 * len(tabela_janelas), 420))

    st.markdown("**Evolução dos pesos-alvo**")
    escolhida = st.selectbox("Estratégia", [n for n in nomes if n != "igualitaria"] or nomes,
                             format_func=NOMES_ESTRATEGIAS.get, key="wf_estrategia_pesos")
    pesos_alvo = resultado["pesos"][escolhida] * 100
    figura = go.Figure([go.Bar(x=pesos_alvo.index, y=pesos_alvo[ativo], name=ativo,
                               hovertemplate=f"{ativo}: %{{y:.2f}}%<extra></extra>") for ativo in pesos_alvo.columns])
    figura.update_layout(
        barmode="stack", height=380, separators=",.", hovermode="x unified", margin=dict(t=30, b=40),
        xaxis=dict(title="Formação de cada janela", hoverformat="%d/%m/%Y", tickformat="%m/%Y"),
        yaxis=dict(title="Peso-alvo (%)", range=[0, 100]), legend=LEGENDA_ABAIXO,
    )
    st.plotly_chart(figura, width="stretch", key="grafico_pesos_walk_forward")
    st.caption("Barra ausente: janela sem nova estimativa (posições mantidas).")


def mapa_de_calor(probabilidades, titulo, periodo):
    """Matriz de transição em % (linha = grupo atual), na escala fixa de 0 a 100."""
    textos = probabilidades.map(formatar_percentual_ou_vazio)
    figura = go.Figure(go.Heatmap(
        z=probabilidades.to_numpy(), x=NOMES_GRUPOS, y=NOMES_GRUPOS, text=textos.to_numpy(),
        texttemplate="%{text}", textfont=dict(size=15), colorscale="Blues", zmin=0, zmax=100,
        colorbar=dict(title="%"), xgap=2, ygap=2,
        hovertemplate="De %{y} para %{x}: %{text}<extra></extra>",
    ))
    figura.update_layout(
        title=titulo, height=420, margin=dict(t=60, b=40), separators=",.",
        xaxis_title=f"Grupo {periodo['do_proximo']}", yaxis_title=f"Grupo {periodo['do_atual']}",
        yaxis=dict(autorange="reversed"),
    )
    return figura


# ---------------------------------------------------------------------------
# Página e filtros da barra lateral
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Regimes do Mercado", layout="wide")
hoje = hoje_sp()

if "parametros" not in st.session_state:
    # Análise padrão exibida ao abrir o painel.
    st.session_state.parametros = {
        "nome": "PETR4 — Petrobras", "codigo": "PETR4.SA",
        "data_inicial": date(2020, 1, 1), "data_final": hoje, "frequencia": "Semanal",
    }

with st.sidebar:
    st.header("Regimes do ativo")
    with st.form("filtros"):
        escolha = st.selectbox("Ação", list(ACOES) + [OUTRO_CODIGO])
        codigo_digitado = st.text_input("Código (se escolher outro)", placeholder="Ex.: WEGE3")
        data_inicial = st.date_input("Data inicial", date(2020, 1, 1), min_value=date(2000, 1, 1),
                                     max_value=hoje, format="DD/MM/YYYY")
        data_final = st.date_input("Data final", hoje, min_value=date(2000, 1, 1),
                                   max_value=hoje, format="DD/MM/YYYY")
        frequencia = st.radio("Frequência", list(FREQUENCIAS), horizontal=True)
        enviado = st.form_submit_button("Analisar ativo", type="primary", width="stretch")
    if enviado:
        codigo, erros = validar_filtros(escolha, codigo_digitado, data_inicial, data_final)
        if erros:
            for erro in erros:
                st.error(erro)
            st.info("A análise anterior continua exibida.")
        else:
            nome = escolha if escolha != OUTRO_CODIGO else codigo.removesuffix(".SA")
            st.session_state.parametros = {
                "nome": nome, "codigo": codigo, "data_inicial": data_inicial,
                "data_final": data_final, "frequencia": frequencia,
            }
    st.caption("Vale para as abas de regimes · Recalcula ao clicar em Analisar ativo · Cotações em cache por 1 h.")

parametros = st.session_state.parametros
frequencia = parametros["frequencia"]
periodo = TEXTOS_PERIODO[frequencia]

st.title("Regimes do Mercado")
st.caption("Regimes de mercado (K-Means e Cadeias de Markov) e laboratório de carteiras.")

aba_geral, aba_kmeans, aba_markov, aba_dados, aba_carteiras = st.tabs(
    ["Visão geral", "K-Means", "Cadeias de Markov", "Dados e normalização", "Carteiras"]
)
# Carteiras permanece acessível mesmo se a consulta do ativo da barra lateral falhar.
with aba_carteiras:
    exibir_carteiras()

with aba_geral, st.spinner("Baixando dados e executando a análise..."):
    try:
        resultado = executar_analise(parametros["codigo"], parametros["data_inicial"],
                                     parametros["data_final"], frequencia)
    except Exception as erro:
        st.error(f"Não foi possível analisar {parametros['nome']} ({parametros['codigo']}): {erro}")
        st.stop()

dados = resultado["dados"]
modelo = resultado["modelo"]
if modelo is not None:
    analise = modelo["analise"]
    treinamento = analise[analise["conjunto"] == "treinamento"]
    inicio_teste = analise.loc[analise["conjunto"] == "teste", "data"].min()

# ---------------------------------------------------------------------------
# Aba 1 — Visão geral
# ---------------------------------------------------------------------------
with aba_geral:
    st.subheader(f"{parametros['nome']} ({parametros['codigo']})")
    st.caption(
        f"Período selecionado: {parametros['data_inicial']:%d/%m/%Y} a {parametros['data_final']:%d/%m/%Y}. "
        "Início e Fim indicam os períodos completos analisados dentro dessa seleção."
    )
    coluna1, coluna2, coluna3, coluna4 = st.columns(4)
    coluna1.metric("Início", f"{dados['data'].iloc[0]:%d/%m/%Y}", border=True)
    coluna2.metric("Fim", f"{dados['data'].iloc[-1]:%d/%m/%Y}", border=True)
    coluna3.metric("Frequência", frequencia, border=True)
    coluna4.metric("Observações", len(dados), border=True)
    coluna1, coluna2, coluna3, coluna4 = st.columns(4)
    coluna1.metric("Último fechamento ajustado", formatar_moeda(dados["fechamento"].iloc[-1]), border=True)
    retorno_acumulado = (dados["fechamento"].iloc[-1] / dados["fechamento"].iloc[0] - 1) * 100
    coluna2.metric("Retorno acumulado", formatar_percentual(retorno_acumulado), border=True)
    if modelo is not None:
        coluna3.metric("Treinamento / teste", f"{len(treinamento)} / {len(analise) - len(treinamento)}", border=True)
        coluna4.metric("Início do teste", f"{inicio_teste:%d/%m/%Y}", border=True)
    else:
        coluna3.metric("Treinamento / teste", "Não executado", border=True)

    figura = go.Figure(go.Scatter(
        x=dados["data"], y=dados["fechamento"], mode="lines", name="Fechamento ajustado",
        line=dict(color=CORES_GRUPOS[0], width=2),
        hovertemplate="%{x|%d/%m/%Y}<br>Fechamento: R$ %{y:.2f}<extra></extra>",
    ))
    if modelo is not None:
        figura.add_shape(type="line", x0=inicio_teste, x1=inicio_teste, y0=0, y1=1, yref="paper",
                         line=dict(color="gray", dash="dash", width=1))
        figura.add_annotation(x=inicio_teste, y=1, yref="paper", text="Início do teste",
                              showarrow=False, xanchor="left", yanchor="bottom")
    figura.update_layout(
        title=f"Fechamento ajustado — frequência {frequencia.lower()}", height=430,
        xaxis_title=f"Data (fim de cada {periodo['cada']})", yaxis_title="Preço ajustado (R$)",
        hovermode="x unified", margin=dict(t=60, b=40), separators=",.",
    )
    st.plotly_chart(figura, width="stretch")
    informacoes = resultado["informacoes"]
    st.caption(
        f"Retorno acumulado entre o primeiro e o último fechamento exibidos. "
        f"Pregões recebidos: {resultado['pregoes']} (inválidos: {resultado['pregoes_invalidos']}). "
        f"Períodos em andamento excluídos: {informacoes['em_andamento']}. "
        f"Períodos incompletos descartados: {len(informacoes['incompletos'])}. "
        f"O primeiro período serve apenas de referência para o primeiro retorno."
    )
    if modelo is None:
        st.warning(resultado["motivo"])

# ---------------------------------------------------------------------------
# Aba 2 — K-Means
# ---------------------------------------------------------------------------
with aba_kmeans:
    if modelo is None:
        st.warning(resultado["motivo"])
    else:
        resumo = modelo["resumo"]
        figura = go.Figure()
        for grupo, cor in zip([1, 2, 3], CORES_GRUPOS):
            selecao = analise[analise["grupo"] == grupo]
            figura.add_trace(go.Scatter(
                x=selecao["variacao_percentual"], y=selecao["amplitude_percentual"], mode="markers",
                name=f"Grupo {grupo}", marker=dict(color=cor, size=8, opacity=0.7, line=dict(width=1, color="white")),
                customdata=selecao[["data", "conjunto"]].assign(data=selecao["data"].dt.strftime("%d/%m/%Y")),
                hovertemplate=(
                    "Data: %{customdata[0]}<br>Retorno: %{x:.2f}%<br>Amplitude: %{y:.2f}%"
                    f"<br>Grupo: {grupo}<br>Conjunto: %{{customdata[1]}}<extra></extra>"
                ),
            ))
        for grupo, cor in zip([1, 2, 3], CORES_GRUPOS):
            centroide = modelo["centroides"][grupo - 1]
            figura.add_trace(go.Scatter(
                x=[centroide[0]], y=[centroide[1]], mode="markers", name=f"Centroide do grupo {grupo}",
                marker=dict(symbol="x", size=18, color=cor, line=dict(width=1.5, color="black")),
                hovertemplate=f"Centroide do grupo {grupo}<br>Retorno: %{{x:.2f}}%<br>Amplitude: %{{y:.2f}}%<extra></extra>",
            ))
        figura.update_layout(
            title=f"Agrupamentos K-Means — retorno × amplitude por {periodo['cada']}", height=520,
            xaxis_title=f"Retorno percentual {periodo['do']} (%)", yaxis_title=f"Amplitude percentual {periodo['do']} (%)",
            margin=dict(t=60, b=40), separators=",.", legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
        )
        st.plotly_chart(figura, width="stretch")

        tabela_grupos = pd.DataFrame({
            "Grupo": [f"Grupo {grupo}" for grupo in resumo.index],
            "Observações": resumo["semanas"].to_numpy(),
            "Retorno médio (%)": resumo["retorno_medio"].to_numpy(),
            "Amplitude média (%)": resumo["amplitude_media"].to_numpy(),
            "Centroide do retorno (%)": resumo["centroide_variacao"].to_numpy(),
            "Centroide da amplitude (%)": resumo["centroide_amplitude"].to_numpy(),
        })
        # Três linhas: texto em pt-BR (vírgula decimal), como nas tabelas de Carteiras.
        for coluna in tabela_grupos.columns[2:]:
            tabela_grupos[coluna] = tabela_grupos[coluna].map(lambda valor: f"{valor:.2f}".replace(".", ","))
        st.dataframe(tabela_grupos, hide_index=True, width="stretch")
        st.caption("Observações e médias incluem treinamento e teste · Centroides (X) ajustados só no treinamento, "
                   "com retorno e amplitude normalizados · Teste atribuído ao centroide mais próximo · "
                   "Grupos ordenados pelo retorno do centroide e usados como estados de Markov.")

# ---------------------------------------------------------------------------
# Aba 3 — Cadeias de Markov
# ---------------------------------------------------------------------------
with aba_markov:
    if modelo is None:
        st.warning(resultado["motivo"])
    elif modelo["contagens"].sum() == 0:
        st.warning("Não há transições entre períodos consecutivos no treinamento; a matriz não pode ser estimada.")
    else:
        contagens = modelo["contagens"]
        probabilidades = pd.DataFrame(modelo["probabilidades"] * 100, index=NOMES_GRUPOS, columns=NOMES_GRUPOS)
        saidas = contagens.sum(axis=1)

        coluna1, coluna2, coluna3 = st.columns(3)
        coluna1.metric("Transições no treinamento", int(contagens.sum()), border=True)
        coluna2.metric("Transições no teste (não usadas)", int(modelo["contagens_teste"].sum()), border=True)
        coluna3.metric("Períodos de treinamento", len(treinamento), border=True)

        coluna_mapa, coluna_contagens = st.columns([3, 2])
        with coluna_mapa:
            st.plotly_chart(mapa_de_calor(probabilidades, "Matriz de transição (treinamento)", periodo), width="stretch")
        with coluna_contagens:
            st.markdown("**Contagem de transições no treinamento**")
            tabela_contagens = pd.DataFrame(contagens, index=NOMES_GRUPOS, columns=NOMES_GRUPOS)
            tabela_contagens["Total de saídas"] = saidas
            st.dataframe(tabela_contagens, width="stretch")
            st.caption(
                f"Linhas: grupo atual. Colunas: grupo {periodo['do_proximo']}. Só são contados pares de "
                "períodos consecutivos dentro do treinamento; a passagem treino → teste não entra."
            )
            for grupo, total in enumerate(saidas, start=1):
                if total == 0:
                    st.warning(f"Grupo {grupo}: nenhuma transição observada; probabilidades indefinidas.")
                elif total < MINIMO_SAIDAS:
                    palavra = "transição" if total == 1 else "transições"
                    st.warning(f"Grupo {grupo}: apenas {total} {palavra}; estimativas pouco estáveis.")

        st.divider()
        st.subheader("Consulta de transições")
        grupo_escolhido = st.selectbox("Grupo atual", NOMES_GRUPOS)
        linha = probabilidades.loc[grupo_escolhido]
        if linha.isna().any():
            st.warning(f"Não houve transições a partir do {grupo_escolhido} no treinamento.")
        else:
            coluna_valores, coluna_barras = st.columns([1, 2])
            with coluna_valores:
                for destino in NOMES_GRUPOS:
                    rotulo = f"Permanecer no {destino}" if destino == grupo_escolhido else f"Ir para o {destino}"
                    st.metric(rotulo, formatar_percentual(linha[destino]), border=True)
            with coluna_barras:
                figura = go.Figure(go.Bar(
                    x=NOMES_GRUPOS, y=linha.to_numpy(), marker_color=CORES_GRUPOS,
                    text=[formatar_percentual(valor) for valor in linha], textposition="outside",
                    hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
                ))
                figura.update_layout(
                    title=f"Probabilidades a partir do {grupo_escolhido}", height=360,
                    xaxis_title=f"Grupo {periodo['do_proximo']}", yaxis_title="Probabilidade (%)",
                    yaxis=dict(range=[0, 105]), margin=dict(t=60, b=40), separators=",.",
                )
                st.plotly_chart(figura, width="stretch")
            total = int(saidas[NOMES_GRUPOS.index(grupo_escolhido)])
            st.caption(f"Base: {total} transições do treinamento · Soma da linha: {formatar_percentual(linha.sum())} · "
                       "Frequências históricas; não garantem comportamento futuro nem constituem recomendação.")

        # --- Validação do modelo: treinamento × teste ---------------------------------
        st.divider()
        st.subheader("Validação do modelo — Treinamento × Teste")
        st.caption("Matriz estimada no treinamento; o teste (períodos posteriores) serve só para comparação.")
        contagens_teste = modelo["contagens_teste"]
        saidas_teste = contagens_teste.sum(axis=1)
        frequencias_teste = pd.DataFrame(modelo["probabilidades_teste"] * 100, index=NOMES_GRUPOS, columns=NOMES_GRUPOS)

        coluna_treino, coluna_teste = st.columns(2)
        with coluna_treino:
            st.plotly_chart(mapa_de_calor(probabilidades, "Treinamento", periodo), width="stretch")
            st.caption(f"Probabilidades estimadas com {int(contagens.sum())} transições do treinamento. "
                       "É a matriz utilizada pelo modelo de Markov.")
        with coluna_teste:
            st.plotly_chart(mapa_de_calor(frequencias_teste, "Teste", periodo), width="stretch")
            st.caption(f"Frequências relativas de {int(contagens_teste.sum())} transições observadas no teste. "
                       "Usada somente para avaliação; não altera a matriz do treinamento.")

        # Tabela comparativa das três transições que partem do grupo escolhido acima.
        origem = NOMES_GRUPOS.index(grupo_escolhido)
        comparacao = pd.DataFrame({
            "Grupo atual": grupo_escolhido,
            "Próximo grupo": NOMES_GRUPOS,
            "Probabilidade no treinamento (%)": probabilidades.loc[grupo_escolhido].to_numpy(),
            "Frequência observada no teste (%)": frequencias_teste.loc[grupo_escolhido].to_numpy(),
        })
        comparacao["Diferença (p.p.)"] = (
            comparacao["Frequência observada no teste (%)"] - comparacao["Probabilidade no treinamento (%)"]
        )
        comparacao["Transições de saída no treinamento"] = int(saidas[origem])
        comparacao["Transições de saída no teste"] = int(saidas_teste[origem])
        exibicao = comparacao.copy()
        for coluna in ["Probabilidade no treinamento (%)", "Frequência observada no teste (%)"]:
            exibicao[coluna] = exibicao[coluna].map(formatar_percentual_ou_vazio)
        exibicao["Diferença (p.p.)"] = exibicao["Diferença (p.p.)"].map(formatar_diferenca)

        st.markdown(f"**Transições a partir do {grupo_escolhido}** (grupo escolhido em *Consulta de transições*)")
        st.dataframe(exibicao, hide_index=True, width="stretch",
                     # Cabeçalhos curtos para caber na tela; o nome completo aparece ao passar o mouse.
                     column_config={coluna: st.column_config.Column(curto, help=coluna) for coluna, curto in [
                         ("Probabilidade no treinamento (%)", "Prob. treino (%)"),
                         ("Frequência observada no teste (%)", "Freq. teste (%)"),
                         ("Transições de saída no treinamento", "Saídas treino"),
                         ("Transições de saída no teste", "Saídas teste"),
                     ]})
        st.caption("Diferença = frequência no teste − probabilidade no treinamento.")
        coluna_saidas, _ = st.columns([1, 2])
        with coluna_saidas:
            st.markdown("**Transições de saída por grupo**")
            st.dataframe(pd.DataFrame({"Grupo": NOMES_GRUPOS, "Treino": saidas, "Teste": saidas_teste}),
                         hide_index=True, width="stretch")

        # Interpretação objetiva da permanência no grupo escolhido.
        permanencia_treino = probabilidades.loc[grupo_escolhido, grupo_escolhido]
        permanencia_teste = frequencias_teste.loc[grupo_escolhido, grupo_escolhido]
        if pd.isna(permanencia_treino):
            frase = f"Não houve transições a partir do {grupo_escolhido} no treinamento; não há probabilidade estimada para comparar."
        elif pd.isna(permanencia_teste):
            frase = (f"No treinamento, a probabilidade estimada de permanência no {grupo_escolhido} foi de "
                     f"{formatar_percentual(permanencia_treino)}. No teste, não houve transições a partir desse grupo.")
        else:
            permanencias = int(contagens_teste[origem, origem])
            frase = (f"No treinamento, a probabilidade estimada de permanência no {grupo_escolhido} foi de "
                     f"{formatar_percentual(permanencia_treino)}. No teste, essa transição ocorreu em "
                     f"{formatar_percentual(permanencia_teste)} das observações ({permanencias} de "
                     f"{int(saidas_teste[origem])}; diferença de {formatar_diferenca(permanencia_teste - permanencia_treino)}).")
        st.info(frase)

        for grupo, total in enumerate(saidas_teste, start=1):
            if total == 0:
                st.warning(f"O Grupo {grupo} não possui transições no conjunto de teste; não há frequências para comparar.")
            elif total < MINIMO_SAIDAS:
                palavra = "transição" if total == 1 else "transições"
                st.warning(f"Grupo {grupo}: apenas {total} {palavra} no teste; frequências instáveis.")
        st.caption("Diferenças pequenas indicam frequências parecidas, não poder preditivo.")

# ---------------------------------------------------------------------------
# Aba 4 — Dados e normalização
# ---------------------------------------------------------------------------
with aba_dados:
    formato_data = st.column_config.DateColumn("Data", format="DD/MM/YYYY")
    formato_numero = st.column_config.NumberColumn(format="%.2f")
    formato_normalizado = st.column_config.NumberColumn(format="%.4f")

    st.subheader("Dados históricos")
    historico = dados.rename(columns={
        "data": "Data", "fechamento": "Fechamento (R$)", "maxima": "Máxima (R$)", "minima": "Mínima (R$)",
        "variacao_percentual": "Retorno (%)", "amplitude_percentual": "Amplitude (%)",
    })
    st.dataframe(historico, hide_index=True, width="stretch", height=300,
                 column_config={"Data": formato_data} | {coluna: formato_numero for coluna in historico.columns[1:]})

    if modelo is None:
        st.warning(resultado["motivo"])
    else:
        st.subheader("Normalização Min-Max")
        st.caption("(valor − mínimo) / (máximo − mínimo), com mínimo e máximo do treinamento; "
                   "valores de teste podem ficar fora de [0, 1].")
        parametros_minmax = treinamento[VARIAVEIS].agg(["min", "max"]).T
        parametros_minmax.index = ["Retorno (%)", "Amplitude (%)"]
        parametros_minmax.columns = ["Mínimo no treino", "Máximo no treino"]
        st.dataframe(parametros_minmax, width="content",
                     column_config={coluna: formato_numero for coluna in parametros_minmax.columns})
        normalizados = analise[["data", VARIAVEIS[0], NORMALIZADAS[0], VARIAVEIS[1], NORMALIZADAS[1]]]
        normalizados.columns = ["Data", "Retorno original (%)", "Retorno normalizado",
                                "Amplitude original (%)", "Amplitude normalizada"]
        st.dataframe(normalizados, hide_index=True, width="stretch", height=300, column_config={
            "Data": formato_data, "Retorno original (%)": formato_numero, "Amplitude original (%)": formato_numero,
            "Retorno normalizado": formato_normalizado, "Amplitude normalizada": formato_normalizado,
        })

        st.subheader("Classificação")
        classificacao = analise[["data", "grupo", "conjunto"]].copy()
        classificacao["grupo"] = "Grupo " + classificacao["grupo"].astype(str)
        classificacao.columns = ["Data", "Grupo", "Conjunto"]
        st.dataframe(classificacao, hide_index=True, width="stretch", height=300,
                     column_config={"Data": formato_data})

    tabela_final = analise if modelo is not None else dados
    nome_arquivo = f"{parametros['codigo'].removesuffix('.SA')}_{frequencia.lower()}_analise.csv"
    st.download_button("Baixar tabela final (CSV)", tabela_final.to_csv(index=False, date_format="%Y-%m-%d"),
                       file_name=nome_arquivo, mime="text/csv")
    st.caption("Os valores são arredondados apenas na exibição; o CSV mantém a precisão completa.")
