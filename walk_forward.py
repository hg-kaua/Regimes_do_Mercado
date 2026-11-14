"""Etapa 11: validação walk-forward das carteiras (o backtest 80/20 é preservado).

Metodologia (regras fixadas antes de qualquer resultado):
  - Janelas móveis: cada ciclo estima os pesos com as `semanas_treino` semanas de
    retornos imediatamente ANTERIORES à sua janela de avaliação de
    `semanas_avaliacao` semanas. As janelas de avaliação são consecutivas e não se
    sobrepõem; o treino avança junto (janela móvel, não expansiva).
  - Ancoragem: a última janela termina na semana mais recente. Semanas que sobram
    ficam no INÍCIO do histórico e não são usadas; todas as janelas têm o mesmo tamanho.
  - Execução igual para todas as estratégias: na formação de cada ciclo (fechamento
    da última semana de treino) a carteira volta aos pesos-alvo; dentro da janela é
    buy and hold, e os pesos efetivos variam com os preços. A primeira alocação é a
    formação; cada retorno aos pesos-alvo nos ciclos seguintes conta como um
    rebalanceamento (a Igualitária também rebalanceia: volta a 1/n).
  - Patrimônio contínuo: cada ciclo começa com o valor final do anterior.
  - Falha de estimativa num ciclo (ex.: Máximo Sharpe sem excesso de retorno
    positivo): mantém as posições vigentes, sem negociar, e registra o ciclo. Se a
    falha ocorrer no primeiro ciclo, a estratégia fica indisponível. Nunca há pesos
    substitutos.
  - Custos de transação: parâmetro `custo_transacao` (decimal sobre o valor
    negociado). Padrão ZERO. Giro = Σ|peso_alvo − peso_atual|; o patrimônio após
    negociar é patrimônio × (1 − custo × giro). A formação inicial tem giro 1.

Mesmas premissas do backtest: preços ajustados, posições fracionárias, sem
impostos, spread ou slippage. Estimativas: média semanal × 52 e covariância × 52.
"""

from numbers import Integral, Real

import numpy as np
import pandas as pd

from analisar_carteira import SEMANAS_ANO, calcular_estatisticas
from backtest_carteira import CAPITAL_INICIAL, calcular_drawdown, calcular_indicadores
from maximo_sharpe import otimizar_maximo_sharpe, validar_taxa_livre_risco
from otimizar_carteira import PESO_MAXIMO, otimizar_pesos, validar_limites


SEMANAS_TREINO_PADRAO = 104
SEMANAS_AVALIACAO_PADRAO = 13
MINIMO_SEMANAS_TREINO = 52   # Mesmo mínimo operacional do backtest 80/20.
MINIMO_SEMANAS_AVALIACAO = 4  # Permite medir variação semanal dentro da janela.
MINIMO_JANELAS = 2            # Consistência exige comparar ao menos duas janelas.
ESTRATEGIAS_WF = ["igualitaria", "minima_volatilidade", "maximo_sharpe", "personalizada"]


def gerar_janelas(datas, semanas_treino=SEMANAS_TREINO_PADRAO, semanas_avaliacao=SEMANAS_AVALIACAO_PADRAO):
    """Posições [início, fim) de treino e avaliação para cada ciclo, em ordem cronológica."""
    for nome, valor, minimo in [("treino", semanas_treino, MINIMO_SEMANAS_TREINO),
                                ("avaliação", semanas_avaliacao, MINIMO_SEMANAS_AVALIACAO)]:
        if isinstance(valor, bool) or not isinstance(valor, Integral) or valor < minimo:
            raise ValueError(f"A janela de {nome} deve ser um número inteiro de semanas, no mínimo {minimo}.")
    total = len(datas)
    ciclos = (total - semanas_treino) // semanas_avaliacao if total > semanas_treino else 0
    if ciclos < MINIMO_JANELAS:
        necessario = semanas_treino + MINIMO_JANELAS * semanas_avaliacao
        raise ValueError(f"Dados insuficientes: {total} semanas de retornos; são necessárias {necessario} "
                         f"({semanas_treino} de treino + {MINIMO_JANELAS} janelas de {semanas_avaliacao}).")
    deslocamento = total - semanas_treino - ciclos * semanas_avaliacao
    linhas = []
    for ciclo in range(ciclos):
        inicio_treino = deslocamento + ciclo * semanas_avaliacao
        inicio_avaliacao = inicio_treino + semanas_treino
        fim_avaliacao = inicio_avaliacao + semanas_avaliacao
        linhas.append({
            "ciclo": ciclo + 1, "i_treino": inicio_treino, "i_avaliacao": inicio_avaliacao, "i_fim": fim_avaliacao,
            "inicio_treino": datas[inicio_treino], "fim_treino": datas[inicio_avaliacao - 1],
            "inicio_avaliacao": datas[inicio_avaliacao], "fim_avaliacao": datas[fim_avaliacao - 1],
        })
    janelas = pd.DataFrame(linhas).set_index("ciclo")
    janelas.attrs["semanas_descartadas_inicio"] = deslocamento
    return janelas


def estimar_pesos(treino, peso_maximo, taxa_livre_risco, pesos_fixos=None):
    """Pesos-alvo de um ciclo usando SOMENTE o treino recebido. Retorna (pesos, falhas)."""
    ativos = treino.columns
    pesos = {"igualitaria": pd.Series(1.0 / len(ativos), index=ativos)}
    falhas = {}
    try:
        _, _, covariancia = calcular_estatisticas(treino)
        covariancia_anual = covariancia * SEMANAS_ANO
    except ValueError as erro:
        falhas["minima_volatilidade"] = falhas["maximo_sharpe"] = f"estatísticas inválidas: {erro}"
    else:
        for nome, estimar in [
            ("minima_volatilidade", lambda: otimizar_pesos(covariancia_anual, 0.0, peso_maximo)),
            ("maximo_sharpe", lambda: otimizar_maximo_sharpe(
                treino.mean() * SEMANAS_ANO, covariancia_anual, taxa_livre_risco, peso_maximo)),
        ]:
            try:
                pesos[nome] = estimar().astype(float)
            except (ValueError, RuntimeError) as erro:
                falhas[nome] = str(erro)
    if pesos_fixos is not None:
        pesos["personalizada"] = pesos_fixos
    return pesos, falhas


def simular_walk_forward(retornos, janelas, alvos, capital=CAPITAL_INICIAL, custo_transacao=0.0):
    """Simula uma estratégia ciclo a ciclo. alvos[i] = pesos do ciclo i ou None (manter posições).

    retornos: decimais semanais. Retorna (patrimônio, registros por ciclo). O ponto da
    curva na data de formação é o valor ANTES da negociação (no primeiro ciclo, o
    capital bruto); o custo (zero por padrão) reduz as posições e aparece na variação
    da primeira semana da janela. Assim, o retorno acumulado e o retorno de cada janela
    incluem os custos, e o produto dos retornos por janela reproduz o acumulado.
    """
    if not 0 <= custo_transacao < 1:
        raise ValueError("O custo de transação deve ser decimal entre 0 e 1 (0.001 = 0,1%).")
    if alvos[0] is None:
        raise ValueError("Não há pesos para formar a carteira no primeiro ciclo.")
    posicoes, pontos, registros = None, {}, []
    for (ciclo, janela), alvo in zip(janelas.iterrows(), alvos):
        valor = capital if posicoes is None else float(posicoes.sum())
        atuais = pd.Series(0.0, index=retornos.columns) if posicoes is None else posicoes / valor
        giro = 0.0 if alvo is None else float((alvo.reindex(retornos.columns) - atuais).abs().sum())
        valor_formacao = valor * (1 - custo_transacao * giro)
        if alvo is not None:
            posicoes = valor_formacao * alvo.reindex(retornos.columns).astype(float)
        pontos[janela["fim_treino"]] = valor
        bloco = retornos.iloc[janela["i_avaliacao"]:janela["i_fim"]]
        caminho = (1 + bloco).cumprod() * posicoes
        valores = caminho.sum(axis=1)
        pontos.update(valores.to_dict())
        posicoes = caminho.iloc[-1]
        serie = pd.concat([pd.Series([valor], index=[janela["fim_treino"]]), valores])
        registros.append({
            "ciclo": ciclo, "rebalanceou": alvo is not None and ciclo > janelas.index[0],
            "nova_estimativa": alvo is not None, "giro": giro, "custo": valor - valor_formacao,
            "patrimonio_formacao": valor_formacao, "patrimonio_final": float(valores.iloc[-1]),
            "retorno_janela_decimal": float(valores.iloc[-1] / valor - 1),
            "drawdown_janela_decimal": float(calcular_drawdown(serie).min()),
        })
    return pd.Series(pontos).sort_index(), pd.DataFrame(registros).set_index("ciclo")


def resumir_consistencia(por_janela, estrategias):
    """Frequência de janelas positivas e acima da Igualitária, sem inferência causal."""
    linhas = {}
    for nome in estrategias:
        retornos = por_janela[f"retorno_{nome}"]
        linha = {"janelas": len(retornos), "janelas_positivas": int((retornos > 0).sum()),
                 "mediana_janela_decimal": float(retornos.median()),
                 "pior_janela_decimal": float(retornos.min()), "melhor_janela_decimal": float(retornos.max())}
        if nome != "igualitaria" and "retorno_igualitaria" in por_janela:
            linha["janelas_acima_igualitaria"] = int((retornos > por_janela["retorno_igualitaria"]).sum())
        linhas[nome] = linha
    return pd.DataFrame(linhas).T


def executar_walk_forward(retornos_percentuais, semanas_treino=SEMANAS_TREINO_PADRAO,
                          semanas_avaliacao=SEMANAS_AVALIACAO_PADRAO, peso_maximo=PESO_MAXIMO,
                          taxa_livre_risco=0.0, capital=CAPITAL_INICIAL, pesos_fixos=None,
                          custo_transacao=0.0):
    """Walk-forward completo a partir dos retornos semanais em PORCENTAGEM.

    pesos_fixos: decimais da carteira manual (política: voltar aos mesmos pesos a
    cada ciclo); None omite a estratégia. Retorna dict com parametros, janelas,
    curva, resumo, por_janela, consistencia, pesos, registros e indisponiveis.
    """
    retornos_percentuais = pd.DataFrame(retornos_percentuais)
    valores = retornos_percentuais.to_numpy(dtype=float)
    if not np.isfinite(valores).all() or (valores <= -100).any():
        raise ValueError("Há retornos ausentes, infinitos ou que zerariam o preço.")
    if retornos_percentuais.index.has_duplicates or not retornos_percentuais.index.is_monotonic_increasing:
        raise ValueError("Os retornos precisam ter datas únicas em ordem cronológica.")
    if isinstance(capital, bool) or not isinstance(capital, Real) or not np.isfinite(capital) or capital <= 0:
        raise ValueError("O capital inicial deve ser positivo.")
    ativos = retornos_percentuais.columns
    validar_limites(len(ativos), 0.0, peso_maximo)
    taxa = validar_taxa_livre_risco(taxa_livre_risco)
    if pesos_fixos is not None:
        pesos_fixos = pd.Series(pesos_fixos, dtype=float).reindex(ativos)
        if (not np.isfinite(pesos_fixos).all() or (pesos_fixos < 0).any()
                or not np.isclose(pesos_fixos.sum(), 1, rtol=0, atol=1e-10)):
            raise ValueError("Os pesos fixos devem cobrir todos os ativos, ser não negativos e somar 100%.")
    decimais = retornos_percentuais / 100.0
    janelas = gerar_janelas(decimais.index, semanas_treino, semanas_avaliacao)

    alvos = {nome: [] for nome in ESTRATEGIAS_WF}
    falhas = {nome: [] for nome in ESTRATEGIAS_WF}
    for ciclo, janela in janelas.iterrows():
        treino = decimais.iloc[janela["i_treino"]:janela["i_avaliacao"]]
        # Garantia explícita contra vazamento: o treino termina antes da avaliação.
        if len(treino) != semanas_treino or not treino.index[-1] < decimais.index[janela["i_avaliacao"]]:
            raise AssertionError("Janela de treino inconsistente com a avaliação.")
        estimados, erros = estimar_pesos(treino, peso_maximo, taxa, pesos_fixos)
        for nome in ESTRATEGIAS_WF:
            if nome == "personalizada" and pesos_fixos is None:
                continue
            alvos[nome].append(estimados.get(nome))
            if nome in erros:
                falhas[nome].append(f"ciclo {ciclo}: {erros[nome]}")

    curvas, registros, indisponiveis = {}, {}, {}
    for nome, lista in alvos.items():
        if not lista:
            continue
        if lista[0] is None:
            indisponiveis[nome] = "sem solução no primeiro ciclo — " + falhas[nome][0]
            continue
        curvas[nome], registros[nome] = simular_walk_forward(decimais, janelas, lista, capital, custo_transacao)
    curva = pd.DataFrame(curvas)
    if curva.isna().any().any():
        raise AssertionError("As estratégias não compartilham as mesmas datas.")

    resumo = pd.DataFrame({nome: calcular_indicadores(serie) for nome, serie in curva.items()}).T
    for nome in curva:
        resumo.loc[nome, "rebalanceamentos"] = int(registros[nome]["rebalanceou"].sum())
        resumo.loc[nome, "ciclos_sem_nova_estimativa"] = int((~registros[nome]["nova_estimativa"]).sum())
        resumo.loc[nome, "giro_medio_rebalanceamento"] = float(
            registros[nome].loc[registros[nome]["rebalanceou"], "giro"].mean()) if registros[nome]["rebalanceou"].any() else 0.0
        resumo.loc[nome, "custos_totais"] = float(registros[nome]["custo"].sum())
    por_janela = janelas[["inicio_treino", "fim_treino", "inicio_avaliacao", "fim_avaliacao"]].copy()
    for nome in curva:
        por_janela[f"retorno_{nome}"] = registros[nome]["retorno_janela_decimal"]
        por_janela[f"drawdown_{nome}"] = registros[nome]["drawdown_janela_decimal"]
    # Linha vazia (NaN) = ciclo sem nova estimativa: as posições vigentes foram mantidas.
    pesos = {nome: pd.DataFrame([pd.Series(np.nan, index=ativos) if alvo is None else alvo for alvo in alvos[nome]],
                                index=janelas["fim_treino"]).rename_axis("formacao")
             for nome in curva}
    return {
        "parametros": {"semanas_treino": semanas_treino, "semanas_avaliacao": semanas_avaliacao,
                       "peso_maximo": peso_maximo, "taxa_livre_risco_anual_decimal": taxa,
                       "capital_inicial": float(capital), "custo_transacao": custo_transacao,
                       "semanas_descartadas_inicio": janelas.attrs["semanas_descartadas_inicio"]},
        "janelas": janelas, "curva": curva, "resumo": resumo, "por_janela": por_janela,
        "consistencia": resumir_consistencia(por_janela, list(curva)), "pesos": pesos,
        "registros": registros, "falhas": {n: f for n, f in falhas.items() if f}, "indisponiveis": indisponiveis,
    }
