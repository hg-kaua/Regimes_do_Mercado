"""Testes do walk-forward: python -m unittest test_walk_forward -v.

Somente dados sintéticos, sem rede. Os casos de resultado conhecido usam
retornos escolhidos para que o patrimônio possa ser conferido à mão.
"""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import walk_forward as wf
from backtest_carteira import calcular_indicadores


def datas(n):
    return pd.date_range("2020-01-10", periods=n, freq="W-FRI")


def retornos_aleatorios(n=200, ativos=("A", "B", "C", "D"), semente=11):
    """Retornos semanais em PORCENTAGEM, como o motor fornece."""
    gerador = np.random.default_rng(semente)
    medias = np.linspace(0.05, 0.4, len(ativos))
    valores = gerador.normal(medias, 2.0 + np.arange(len(ativos)), size=(n, len(ativos)))
    return pd.DataFrame(valores, index=datas(n), columns=list(ativos))


class TestarJanelas(unittest.TestCase):
    def test_ordem_sem_sobreposicao_e_ancoragem_no_fim(self):
        for total, descartadas in [(130, 0), (135, 5)]:
            janelas = wf.gerar_janelas(datas(total), 52, 13)
            self.assertEqual(len(janelas), 6)
            self.assertEqual(janelas.attrs["semanas_descartadas_inicio"], descartadas)
            self.assertEqual(janelas["i_fim"].iloc[-1], total)  # Última janela termina na semana mais recente.
            self.assertTrue((janelas["i_avaliacao"] - janelas["i_treino"] == 52).all())
            self.assertTrue((janelas["i_fim"] - janelas["i_avaliacao"] == 13).all())
            # Avaliações consecutivas, sem lacuna nem sobreposição.
            self.assertTrue((janelas["i_avaliacao"].iloc[1:].to_numpy() == janelas["i_fim"].iloc[:-1].to_numpy()).all())
            self.assertTrue((janelas["fim_treino"] < janelas["inicio_avaliacao"]).all())
            self.assertTrue(janelas["inicio_avaliacao"].is_monotonic_increasing)
            # A formação de cada ciclo é o fim da avaliação anterior (patrimônio contínuo).
            self.assertTrue((janelas["fim_treino"].iloc[1:].to_numpy() == janelas["fim_avaliacao"].iloc[:-1].to_numpy()).all())

    def test_dados_insuficientes_e_parametros_invalidos(self):
        with self.assertRaisesRegex(ValueError, "insuficientes"):
            wf.gerar_janelas(datas(52 + 13 + 12), 52, 13)  # Só uma janela completa.
        for treino, avaliacao in [(51, 13), (104, 3), (104.0, 13), (True, 13), (104, "13")]:
            with self.assertRaises(ValueError):
                wf.gerar_janelas(datas(400), treino, avaliacao)
        with self.assertRaisesRegex(ValueError, "insuficientes"):
            wf.executar_walk_forward(retornos_aleatorios(100))  # Padrão 104 + 2×13.


class TestarSimulacao(unittest.TestCase):
    """Dois ativos, 52 semanas de treino e duas janelas de 4 semanas."""

    def preparar(self):
        retornos = pd.DataFrame(0.0, index=datas(60), columns=["A", "B"])
        retornos.iloc[52:56] = [[0.10, -0.50], [0.10, 1.00], [0, 0], [0, 0]]
        retornos.iloc[56:60] = [[0.10, 0], [0, 0], [0, 0], [0, 0]]
        return retornos, wf.gerar_janelas(retornos.index, 52, 4)

    def test_rebalanceamento_com_resultado_conhecido(self):
        retornos, janelas = self.preparar()
        meio = pd.Series([0.5, 0.5], index=["A", "B"])
        curva, registros = wf.simular_walk_forward(retornos, janelas, [meio, meio], 100.0)
        # Ciclo 1: A 50→60,5 e B 50→50 → 110,5. Ciclo 2 volta a 55,25/55,25; A sobe 10% → 116,025.
        self.assertAlmostEqual(curva.iloc[0], 100.0)
        self.assertAlmostEqual(curva[janelas["fim_avaliacao"].iloc[0]], 110.5)
        self.assertAlmostEqual(curva.iloc[-1], 116.025)
        self.assertEqual(list(registros["rebalanceou"]), [False, True])
        self.assertAlmostEqual(registros["giro"].iloc[1], 2 * abs(0.5 - 60.5 / 110.5))
        self.assertAlmostEqual(registros["retorno_janela_decimal"].iloc[1], 116.025 / 110.5 - 1)
        # Semana 1 do ciclo 1: A 55 + B 25 = 80 → queda de 20% desde a formação.
        self.assertAlmostEqual(registros["drawdown_janela_decimal"].iloc[0], -0.20)
        # Sem nova estimativa no ciclo 2, as posições derivadas são mantidas: 60,5×1,1 + 50.
        curva_mantida, registros_mantidos = wf.simular_walk_forward(retornos, janelas, [meio, None], 100.0)
        self.assertAlmostEqual(curva_mantida.iloc[-1], 116.55)
        self.assertEqual(list(registros_mantidos["rebalanceou"]), [False, False])
        self.assertEqual(registros_mantidos["giro"].iloc[1], 0.0)

    def test_custos_reduzem_patrimonio_pelo_giro(self):
        retornos, janelas = self.preparar()
        meio = pd.Series([0.5, 0.5], index=["A", "B"])
        curva, registros = wf.simular_walk_forward(retornos, janelas, [meio, meio], 100.0, custo_transacao=0.01)
        inicial = 100 * (1 - 0.01 * 1.0)  # Formação: giro 1 (compra de 100%).
        fim_1 = inicial * 1.105
        giro = 2 * abs(0.5 - 60.5 / 110.5)
        apos = fim_1 * (1 - 0.01 * giro)
        # A curva começa no capital bruto: o custo da formação entra no retorno medido.
        self.assertAlmostEqual(curva.iloc[0], 100.0)
        self.assertAlmostEqual(registros["patrimonio_formacao"].iloc[0], inicial)
        self.assertAlmostEqual(curva.iloc[-1], apos * (0.5 * 1.1 + 0.5))
        self.assertAlmostEqual(registros["custo"].sum(), 1.0 + fim_1 * 0.01 * giro)
        acumulado = calcular_indicadores(curva)["retorno_acumulado_decimal"]
        self.assertAlmostEqual(acumulado, curva.iloc[-1] / 100.0 - 1)
        self.assertAlmostEqual(float(np.prod(1 + registros["retorno_janela_decimal"]) - 1), acumulado)
        self.assertAlmostEqual(registros["retorno_janela_decimal"].iloc[0], fim_1 / 100.0 - 1)
        with self.assertRaises(ValueError):
            wf.simular_walk_forward(retornos, janelas, [meio, meio], 100.0, custo_transacao=-0.01)
        with self.assertRaises(ValueError):
            wf.simular_walk_forward(retornos, janelas, [None, meio], 100.0)


class TestarExecucaoCompleta(unittest.TestCase):
    def setUp(self):
        self.retornos = retornos_aleatorios()
        self.resultado = wf.executar_walk_forward(self.retornos, 104, 13, peso_maximo=0.5)

    def test_mesmas_datas_continuidade_e_pesos_aplicados(self):
        r, janelas = self.resultado, self.resultado["janelas"]
        curva = r["curva"]
        self.assertEqual(list(curva), ["igualitaria", "minima_volatilidade", "maximo_sharpe"])
        self.assertFalse(curva.isna().any().any())
        esperadas = pd.DatetimeIndex([janelas["fim_treino"].iloc[0]]).append(
            self.retornos.index[janelas["i_avaliacao"].iloc[0]:])
        pd.testing.assert_index_equal(curva.index, esperadas, check_names=False)
        decimais = self.retornos / 100
        for nome in curva:
            registros = r["registros"][nome]
            # Cada ciclo começa com o patrimônio final do anterior (custos zero).
            np.testing.assert_allclose(registros["patrimonio_formacao"].iloc[1:], registros["patrimonio_final"].iloc[:-1])
            # Dentro de cada janela, patrimônio = valor de formação × Σ w_i Π(1 + r_i).
            for ciclo, janela in janelas.iterrows():
                pesos = r["pesos"][nome].loc[janela["fim_treino"]]
                bloco = decimais.iloc[janela["i_avaliacao"]:janela["i_fim"]]
                esperado = registros.loc[ciclo, "patrimonio_formacao"] * ((1 + bloco).cumprod() @ pesos)
                np.testing.assert_allclose(curva[nome].loc[bloco.index], esperado, rtol=1e-12)
            self.assertEqual(r["resumo"].loc[nome, "rebalanceamentos"], len(janelas) - 1)
            indicadores = calcular_indicadores(curva[nome])
            self.assertEqual(r["resumo"].loc[nome, "retorno_acumulado_decimal"], indicadores["retorno_acumulado_decimal"])
            self.assertTrue((r["pesos"][nome].sum(axis=1).sub(1).abs() < 1e-7).all())
            self.assertTrue((r["pesos"][nome] <= 0.5 + 1e-7).all().all())
        np.testing.assert_allclose(r["pesos"]["igualitaria"], 0.25)

    def test_sem_vazamento_temporal(self):
        janelas = self.resultado["janelas"]
        ciclo = 4
        inicio = janelas.loc[ciclo, "i_avaliacao"]
        alterados = self.retornos.copy()
        alterados.iloc[inicio:] = 25.0 * np.sign(np.sin(np.arange(len(alterados) - inicio)))[:, None] + 1
        outro = wf.executar_walk_forward(alterados, 104, 13, peso_maximo=0.5)
        for nome in self.resultado["curva"]:
            # Pesos até o ciclo alterado (inclusive) dependem só de semanas anteriores à sua avaliação.
            pd.testing.assert_frame_equal(self.resultado["pesos"][nome].iloc[:ciclo], outro["pesos"][nome].iloc[:ciclo],
                                          check_exact=True)
            # A curva até a formação do ciclo alterado não muda.
            limite = janelas.loc[ciclo, "fim_treino"]
            pd.testing.assert_series_equal(self.resultado["curva"][nome].loc[:limite], outro["curva"][nome].loc[:limite],
                                           check_exact=True)
            self.assertFalse(self.resultado["pesos"]["minima_volatilidade"].iloc[ciclo:].equals(
                outro["pesos"]["minima_volatilidade"].iloc[ciclo:]))

    def test_estimativas_usam_somente_o_treino_do_ciclo(self):
        chamadas = []
        original = wf.calcular_estatisticas

        def registrar(treino):
            chamadas.append((treino.index[0], treino.index[-1], len(treino)))
            return original(treino)

        with patch.object(wf, "calcular_estatisticas", side_effect=registrar):
            r = wf.executar_walk_forward(self.retornos, 104, 13, peso_maximo=0.5)
        janelas = r["janelas"]
        self.assertEqual(len(chamadas), len(janelas))
        for (inicio, fim, tamanho), (_, janela) in zip(chamadas, janelas.iterrows()):
            self.assertEqual((inicio, fim, tamanho), (janela["inicio_treino"], janela["fim_treino"], 104))
            self.assertLess(fim, janela["inicio_avaliacao"])

    def test_capital_nao_altera_percentuais(self):
        outro = wf.executar_walk_forward(self.retornos, 104, 13, peso_maximo=0.5, capital=1_000.0)
        for coluna in ["retorno_acumulado_decimal", "volatilidade_anualizada_realizada_decimal", "drawdown_maximo_decimal"]:
            np.testing.assert_allclose(outro["resumo"][coluna].astype(float), self.resultado["resumo"][coluna].astype(float))
        np.testing.assert_allclose(outro["curva"] * 100, self.resultado["curva"], rtol=1e-12)

    def test_falhas_do_sharpe_e_carteira_manual(self):
        # Taxa inalcançável: falha no primeiro ciclo → estratégia indisponível; as demais seguem.
        r = wf.executar_walk_forward(self.retornos, 104, 13, peso_maximo=0.5, taxa_livre_risco=5.0)
        self.assertIn("maximo_sharpe", r["indisponiveis"])
        self.assertEqual(list(r["curva"]), ["igualitaria", "minima_volatilidade"])
        # Falha apenas no segundo ciclo: mantém posições, sem rebalanceamento naquele ciclo.
        original, chamadas = wf.otimizar_maximo_sharpe, []

        def falhar_no_segundo(*args, **kwargs):
            chamadas.append(1)
            if len(chamadas) == 2:
                raise ValueError("falha simulada")
            return original(*args, **kwargs)

        with patch.object(wf, "otimizar_maximo_sharpe", side_effect=falhar_no_segundo):
            r = wf.executar_walk_forward(self.retornos, 104, 13, peso_maximo=0.5)
        registros = r["registros"]["maximo_sharpe"]
        self.assertFalse(registros["nova_estimativa"].iloc[1])
        self.assertEqual(r["resumo"].loc["maximo_sharpe", "rebalanceamentos"], len(r["janelas"]) - 2)
        self.assertEqual(r["resumo"].loc["maximo_sharpe", "ciclos_sem_nova_estimativa"], 1)
        self.assertTrue(r["pesos"]["maximo_sharpe"].iloc[1].isna().all())
        self.assertIn("ciclo 2: falha simulada", r["falhas"]["maximo_sharpe"][0])
        # Carteira manual: política fixa, volta aos mesmos pesos em todo ciclo.
        fixos = {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1}
        r = wf.executar_walk_forward(self.retornos, 104, 13, peso_maximo=0.5, pesos_fixos=fixos)
        self.assertIn("personalizada", r["curva"])
        np.testing.assert_allclose(r["pesos"]["personalizada"], np.tile([0.7, 0.1, 0.1, 0.1], (len(r["janelas"]), 1)))
        with self.assertRaises(ValueError):
            wf.executar_walk_forward(self.retornos, 104, 13, pesos_fixos={"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.2})

    def test_consistencia_por_janela(self):
        r = self.resultado
        consistencia, por_janela = r["consistencia"], r["por_janela"]
        for nome in r["curva"]:
            retornos = por_janela[f"retorno_{nome}"]
            self.assertEqual(consistencia.loc[nome, "janelas_positivas"], (retornos > 0).sum())
            self.assertEqual(consistencia.loc[nome, "pior_janela_decimal"], retornos.min())
            # O produto dos retornos por janela reproduz o retorno acumulado (custos zero).
            self.assertAlmostEqual(float(np.prod(1 + retornos) - 1), r["resumo"].loc[nome, "retorno_acumulado_decimal"])
        self.assertEqual(consistencia.loc["minima_volatilidade", "janelas_acima_igualitaria"],
                         (por_janela["retorno_minima_volatilidade"] > por_janela["retorno_igualitaria"]).sum())


class TestarInterfaceWalkForward(unittest.TestCase):
    """AppTest offline: preços, Ibovespa e CDI sintéticos (mesmos geradores de test_benchmarks)."""

    def setUp(self):
        import streamlit as st
        st.cache_data.clear()

    def test_execucao_tabelas_capital_e_parametros(self):
        from datetime import date
        from pathlib import Path
        from streamlit.testing.v1 import AppTest
        from test_benchmarks import BOTAO, cdi_sintetico, ibov_sintetico, precos_sinteticos

        def tabela_wf(at):
            for elemento in at.tabs[4].get("dataframe"):
                if "Rebalanceamentos" in elemento.value.index:
                    return elemento.value
            return None

        with patch("baixar_dados.baixar_precos_diarios", side_effect=ValueError("Regimes fora do teste")),                 patch("motor_carteiras.coletar_historicos", side_effect=precos_sinteticos),                 patch("benchmarks.baixar_ibovespa", side_effect=ibov_sintetico),                 patch("benchmarks.baixar_cdi", side_effect=cdi_sintetico),                 patch("walk_forward.estimar_pesos", wraps=wf.estimar_pesos) as estimar:
            at = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=180).run()
            at.date_input(key="fim_carteiras").set_value(date(2024, 12, 31))
            at.button(key=BOTAO).click().run()
            self.assertFalse(at.exception)
            self.assertIsNone(tabela_wf(at))  # Só executa quando solicitado; o backtest 80/20 continua.
            self.assertTrue(any("Executar walk-forward" in i.value for i in at.tabs[4].info))
            backtest = at.tabs[4].get("dataframe")[-1].value
            self.assertIn("Igualitária", backtest.columns)

            at.button(key="FormSubmitter:formulario_walk_forward-Executar walk-forward").click().run()
            self.assertFalse(at.exception)
            estado = at.session_state["walk_forward"]["resultado"]
            carteiras = at.session_state["analise_carteiras"]["resultado"]
            fixos = at.session_state["walk_forward"]["chave"][5]  # Pesos da Personalizada usados na execução.
            esperado = wf.executar_walk_forward(carteiras["retornos_percentuais"], 104, 13, carteiras["parametros"]["peso_maximo"],
                                                0.0, 100_000.0, dict(fixos) if fixos else None)
            pd.testing.assert_frame_equal(estado["curva"], esperado["curva"])
            tabela = tabela_wf(at)
            self.assertEqual(list(tabela.columns), ["Igualitária", "Mínima volatilidade", "Máximo Sharpe", "Personalizada"])
            for nome, rotulo in [("igualitaria", "Igualitária"), ("maximo_sharpe", "Máximo Sharpe")]:
                self.assertEqual(tabela.loc["Retorno acumulado", rotulo],
                                 f"{esperado['resumo'].loc[nome, 'retorno_acumulado_decimal'] * 100:.2f}%".replace(".", ","))
                self.assertEqual(tabela.loc["Rebalanceamentos", rotulo], str(len(esperado["janelas"]) - 1))
            janelas = [e.value for e in at.tabs[4].get("dataframe") if e.value.index.name == "Janela"][0]
            self.assertEqual(len(janelas), len(esperado["janelas"]))
            graficos = {c.proto.id for c in at.tabs[4].get("plotly_chart")}
            self.assertTrue(any("grafico_walk_forward" in g for g in graficos))
            self.assertTrue(any("grafico_pesos_walk_forward" in g for g in graficos))
            chamadas = estimar.call_count

            # Capital: reescala o patrimônio sem reestimar pesos; percentuais iguais.
            percentuais = tabela.drop(index="Patrimônio final")
            at.number_input(key="capital_carteiras").set_value(200_000.)
            at.button(key=BOTAO).click().run()
            at.button(key="FormSubmitter:formulario_walk_forward-Executar walk-forward").click().run()
            nova = tabela_wf(at)
            pd.testing.assert_frame_equal(nova.drop(index="Patrimônio final"), percentuais)
            final = esperado["curva"]["igualitaria"].iloc[-1] * 2
            self.assertEqual(nova.loc["Patrimônio final", "Igualitária"],
                             "R$ " + f"{final:,.2f}".replace(",", "_").replace(".", ",").replace("_", "."))
            self.assertEqual(estimar.call_count, chamadas)  # Resultado veio do cache.

            # Janela inviável para os dados: erro claro, backtest original preservado.
            at.number_input(key="wf_treino").set_value(520)
            at.button(key="FormSubmitter:formulario_walk_forward-Executar walk-forward").click().run()
            self.assertFalse(at.exception)
            self.assertTrue(any("insuficientes" in e.value for e in at.tabs[4].error))
            self.assertIsNone(tabela_wf(at))
            self.assertIn("Igualitária", [e.value for e in at.tabs[4].get("dataframe")][-1].columns)

            # Mudar a carteira manual torna o resultado anterior desatualizado.
            at.number_input(key="wf_treino").set_value(104)
            at.button(key="FormSubmitter:formulario_walk_forward-Executar walk-forward").click().run()
            self.assertIsNotNone(tabela_wf(at))
            ativo = carteiras["pesos"].index[0]
            at.number_input(key=f"peso_manual_{ativo}").set_value(0.0).run()
            self.assertIsNone(tabela_wf(at))


if __name__ == "__main__":
    unittest.main()
