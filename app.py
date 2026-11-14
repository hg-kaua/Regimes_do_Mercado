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
        analise, resumo, centroides, corte = normalizar_e_agrupar(dados)
    except ValueError as erro:
        resultado["motivo"] = f"Não foi possível executar o K-Means: {erro}"
        return resultado
    contagens, probabilidades, contagens_teste, probabilidades_teste = estimar_markov(analise, regra)
    resultado["modelo"] = {
        "analise": analise, "resumo": resumo, "centroides": centroides, "corte": corte,
        "contagens": contagens, "probabilidades": probabilidades, "contagens_teste": contagens_teste,
        "probabilidades_teste": probabilidades_teste,
    }
    return resultado


def validar_filtros(escolha, codigo_digitado, data_inicial, data_final):
    """Retorna o código do Yahoo e uma lista de erros encontrados nos filtros."""
    erros = []
    hoje = pd.Timestamp.now(tz="America/Sao_Paulo").date()
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


def mapa_de_calor(probabilidades, titulo, periodo):
    """Matriz de transição em % (linha = grupo atual), na escala fixa de 0 a 100."""
    textos = probabilidades.map(lambda valor: "Sem dados" if pd.isna(valor) else formatar_percentual(valor))
    figura = go.Figure(go.Heatmap(
        z=probabilidades.to_numpy(), x=NOMES_GRUPOS, y=NOMES_GRUPOS, text=textos.to_numpy(),
        texttemplate="%{text}", textfont=dict(size=15), colorscale="Blues", zmin=0, zmax=100,
        colorbar=dict(title="%"), xgap=2, ygap=2,
        hovertemplate="De %{y} para %{x}: %{text}<extra></extra>",
    ))
    figura.update_layout(
        title=titulo, height=400, margin=dict(t=60, b=40),
        xaxis_title=f"Grupo {periodo['do_proximo']}", yaxis_title=f"Grupo {periodo['do_atual']}",
        yaxis=dict(autorange="reversed"),
    )
    return figura


# ---------------------------------------------------------------------------
# Página e filtros da barra lateral
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Regimes do Mercado", layout="wide")
hoje = pd.Timestamp.now(tz="America/Sao_Paulo").date()

if "parametros" not in st.session_state:
    # Análise padrão exibida ao abrir o painel.
    st.session_state.parametros = {
        "nome": "PETR4 — Petrobras", "codigo": "PETR4.SA",
        "data_inicial": date(2020, 1, 1), "data_final": hoje, "frequencia": "Semanal",
    }

with st.sidebar:
    st.header("Filtros da análise")
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
    st.caption(
        "Os resultados só são recalculados ao clicar em **Analisar ativo**. "
        "Os dados do Yahoo Finance ficam em cache por uma hora."
    )

parametros = st.session_state.parametros
frequencia = parametros["frequencia"]
periodo = TEXTOS_PERIODO[frequencia]

st.title("Regimes do Mercado")
st.caption(
    "Agrupamento de períodos por retorno e amplitude com K-Means e transições entre "
    "grupos com Cadeias de Markov. Projeto da disciplina Tópicos Especiais em Inteligência Artificial."
)

with st.spinner("Baixando dados e executando a análise..."):
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

aba_geral, aba_kmeans, aba_markov, aba_dados = st.tabs(
    ["Visão geral", "K-Means", "Cadeias de Markov", "Dados e normalização"]
)

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
    coluna1.metric("Último fechamento ajustado", f"R$ {dados['fechamento'].iloc[-1]:.2f}".replace(".", ","), border=True)
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
        hovermode="x unified", margin=dict(t=60, b=40),
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
            margin=dict(t=60, b=40),
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
        st.dataframe(tabela_grupos, hide_index=True, width="stretch",
                     column_config={coluna: st.column_config.NumberColumn(format="%.2f")
                                    for coluna in tabela_grupos.columns[2:]})
        st.caption("Observações e médias incluem treinamento e teste; os centroides foram ajustados apenas no treinamento.")
        st.markdown(
            "**Como interpretar.** O K-Means separa os períodos em três grupos com retorno e "
            "amplitude semelhantes, usando as duas variáveis normalizadas. Cada **centroide** (X) é o "
            "ponto médio de um grupo no treinamento, convertido de volta para porcentagem: ele "
            "representa o comportamento típico daquele grupo. Os períodos de teste são atribuídos "
            "ao centroide mais próximo. Os grupos são numerados pelo retorno do centroide, do menor "
            "para o maior, e servem como estados da Cadeia de Markov."
        )

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
            textos = probabilidades.map(lambda valor: "Sem dados" if pd.isna(valor) else formatar_percentual(valor))
            figura = go.Figure(go.Heatmap(
                z=probabilidades.to_numpy(), x=NOMES_GRUPOS, y=NOMES_GRUPOS, text=textos.to_numpy(),
                texttemplate="%{text}", textfont=dict(size=15), colorscale="Blues", zmin=0, zmax=100,
                colorbar=dict(title="%"), xgap=2, ygap=2,
                hovertemplate="De %{y} para %{x}: %{text}<extra></extra>",
            ))
            figura.update_layout(
                title="Matriz de transição (treinamento)", height=420, margin=dict(t=60, b=40),
                xaxis_title=f"Grupo {periodo['do_proximo']}", yaxis_title=f"Grupo {periodo['do_atual']}",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(figura, width="stretch")
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
                    yaxis=dict(range=[0, 105]), margin=dict(t=60, b=40),
                )
                st.plotly_chart(figura, width="stretch")
            total = int(saidas[NOMES_GRUPOS.index(grupo_escolhido)])
            st.markdown(
                f"Considerando que {periodo['o_atual']} pertence ao **{grupo_escolhido}**, o modelo estima, "
                f"com base em {total} transições do treinamento, as probabilidades de {periodo['o_proximo']} "
                f"pertencer aos Grupos 1, 2 ou 3. Soma da linha: {formatar_percentual(linha.sum())}."
            )
        st.info(
            "As probabilidades são frequências históricas observadas no treinamento. Elas não "
            "garantem o comportamento futuro do ativo e não constituem recomendação de compra ou venda."
        )

        # --- Validação do modelo: treinamento × teste ---------------------------------
        st.divider()
        st.subheader("Validação do modelo — Treinamento × Teste")
        st.markdown(
            "O treinamento é utilizado para estimar as probabilidades de transição. O teste contém "
            "períodos posteriores que não participaram dessa estimação. A comparação permite verificar "
            "se as frequências observadas no teste são semelhantes às probabilidades calculadas no treinamento."
        )
        contagens_teste = modelo["contagens_teste"]
        saidas_teste = contagens_teste.sum(axis=1)
        frequencias_teste = pd.DataFrame(modelo["probabilidades_teste"] * 100, index=NOMES_GRUPOS, columns=NOMES_GRUPOS)

        coluna_treino, coluna_teste = st.columns(2)
        with coluna_treino:
            st.plotly_chart(mapa_de_calor(probabilidades, "Matriz 1 — Treinamento", periodo), width="stretch")
            st.caption(f"Probabilidades estimadas com {int(contagens.sum())} transições do treinamento. "
                       "É a matriz utilizada pelo modelo de Markov.")
        with coluna_teste:
            st.plotly_chart(mapa_de_calor(frequencias_teste, "Matriz 2 — Teste", periodo), width="stretch")
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
            exibicao[coluna] = exibicao[coluna].map(lambda valor: "Sem dados" if pd.isna(valor) else formatar_percentual(valor))
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
        st.caption("Diferença =frequência no teste − probabilidade no treinamento.")
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
                st.warning(f"O Grupo {grupo} possui poucas transições no conjunto de teste. As frequências "
                           "observadas podem ser instáveis e não permitem conclusões confiáveis.")
        st.caption(
            "Diferenças pequenas entre as matrizes não comprovam bom desempenho preditivo: indicam apenas "
            "que as frequências históricas dos dois períodos foram parecidas."
        )

# ---------------------------------------------------------------------------
# Aba 4 — Dados e normalização
# ---------------------------------------------------------------------------
with aba_dados:
    formato_data = st.column_config.DateColumn("Data", format="DD/MM/YYYY")
    formato_numero = st.column_config.NumberColumn(format="%.2f")
    formato_normalizado = st.column_config.NumberColumn(format="%.4f")

    st.subheader("Tabela 1 — Dados históricos")
    historico = dados.rename(columns={
        "data": "Data", "fechamento": "Fechamento (R$)", "maxima": "Máxima (R$)", "minima": "Mínima (R$)",
        "variacao_percentual": "Retorno (%)", "amplitude_percentual": "Amplitude (%)",
    })
    st.dataframe(historico, hide_index=True, width="stretch", height=300,
                 column_config={"Data": formato_data} | {coluna: formato_numero for coluna in historico.columns[1:]})

    if modelo is None:
        st.warning(resultado["motivo"])
    else:
        st.subheader("Tabela 2 — Dados normalizados (Min-Max)")
        st.markdown(
            "Cada variável é levada para uma escala comum com "
            "`(valor − mínimo) / (máximo − mínimo)`. O mínimo e o máximo são calculados "
            "**somente no treinamento** e reaproveitados no teste; por isso, valores de teste "
            "podem ficar abaixo de 0 ou acima de 1."
        )
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

        st.subheader("Tabela 3 — Classificação")
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
