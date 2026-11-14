"""Testes dos benchmarks: python -m unittest test_benchmarks -v.

Os testes matemáticos e de interface usam séries artificiais e não acessam a rede.
Defina TESTAR_DADOS_REAIS=1 para consultar Yahoo Finance e Banco Central e
conferir o CDI acumulado da série 12 com a série anualizada 4389 do BCB.
"""
from datetime import date
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

import benchmarks as b
from backtest_carteira import calcular_indicadores


ARQUIVO = Path(__file__).with_name("app.py")
BOTAO = "FormSubmitter:formulario_carteiras-Analisar carteira"
DATAS = pd.date_range("2024-01-05", periods=6, freq="W-FRI")


def taxas_uteis(inicio, fim, valor=0.05):
    dias = pd.bdate_range(inicio, fim)
    return pd.Series(valor, index=dias, name="cdi")


class TestarCalculos(unittest.TestCase):
    def test_cdi_capitaliza_dias_uteis_antes_de_cada_semana(self):
        taxas = taxas_uteis(DATAS[0], DATAS[-1])
        curva = b.curva_cdi(taxas, DATAS, 1000.0)
        self.assertEqual(curva.iloc[0], 1000.0)
        for data in DATAS:
            # Dias úteis f ≤ d < t: a taxa da própria sexta t ainda não remunerou.
            dias = ((taxas.index >= DATAS[0]) & (taxas.index < data)).sum()
            self.assertAlmostEqual(curva[data], 1000.0 * 1.0005 ** dias, places=9)
        self.assertEqual(((taxas.index >= DATAS[0]) & (taxas.index < DATAS[1])).sum(), 5)
        # Taxas posteriores à última data não alteram nada.
        futuras = pd.concat([taxas, taxas_uteis(DATAS[-1] + pd.Timedelta(days=3), DATAS[-1] + pd.Timedelta(days=30), 9.0 / 100)])
        pd.testing.assert_series_equal(b.curva_cdi(futuras.sort_index(), DATAS, 1000.0), curva)

    def test_cdi_recusa_lacunas_e_unidade_errada(self):
        taxas = taxas_uteis(DATAS[0], DATAS[-1])
        sem_semana = taxas[(taxas.index < DATAS[2]) | (taxas.index >= DATAS[3])]
        with self.assertRaisesRegex(ValueError, "lacunas"):
            b.curva_cdi(sem_semana, DATAS)
        with self.assertRaisesRegex(ValueError, "unidade"):
            b.curva_cdi(taxas * 100, DATAS)  # 5% ao dia: valor em formato errado.
        with self.assertRaisesRegex(ValueError, "lacunas"):
            b.curva_cdi(taxas[taxas.index < DATAS[0]], DATAS)
        # Feriado isolado (lacuna de 4 dias) é aceito, sem preencher o dia ausente.
        feriado = taxas.drop(pd.Timestamp("2024-01-19"))
        self.assertTrue(np.isfinite(b.curva_cdi(feriado, DATAS)).all())

    def test_ibovespa_razao_de_pontos_e_semana_ausente(self):
        pontos = pd.Series([100., 110., 99., 120., 118., 121.], index=DATAS)
        curva = b.curva_ibovespa(pontos, DATAS, 50.0)
        np.testing.assert_allclose(curva, 50.0 * pontos / 100.)
        with self.assertRaisesRegex(ValueError, "19/01/2024"):
            b.curva_ibovespa(pontos.drop(DATAS[2]), DATAS)
        with self.assertRaisesRegex(ValueError, "19/01/2024"):
            b.curva_ibovespa(pontos.where(pontos.index != DATAS[2]), DATAS)

    def test_benchmark_indisponivel_nao_interrompe_e_indicadores_conferem(self):
        pedidos = []

        def ibov(inicio, limite):
            pedidos.append(("ibov", inicio, limite))
            raise ConnectionError("Yahoo fora do ar")

        def cdi(inicio, fim):
            pedidos.append(("cdi", inicio, fim))
            return taxas_uteis(DATAS[0] - pd.Timedelta(days=10), DATAS[-1])

        r = b.calcular_benchmarks(DATAS, DATAS[-1] + pd.Timedelta(days=30), ibovespa=ibov, cdi=cdi)
        self.assertEqual(list(r["indisponiveis"]), ["ibovespa"])
        self.assertIn("Yahoo fora do ar", r["indisponiveis"]["ibovespa"])
        self.assertEqual(list(r["resumo"].index), ["cdi"])
        esperado = calcular_indicadores(r["curva"]["cdi"])
        for coluna in ["retorno_acumulado_decimal", "retorno_anualizado_realizado_decimal",
                       "volatilidade_anualizada_realizada_decimal", "drawdown_maximo_decimal"]:
            self.assertEqual(r["resumo"].loc["cdi", coluna], esperado[coluna])
        self.assertEqual(r["resumo"].loc["cdi", "drawdown_maximo_decimal"], 0.0)
        # Nenhuma coleta além da última semana avaliada, mesmo com limite maior.
        self.assertLessEqual(pedidos[0][2], (DATAS[-1] + pd.Timedelta(days=1)).date())
        self.assertEqual(pedidos[1][2], DATAS[-1].date())

    def test_percentuais_independem_do_capital(self):
        def cdi(inicio, fim):
            return taxas_uteis(inicio, fim)

        def ibov(inicio, limite):
            return pd.Series([100., 104., 101., 97., 103., 108.], index=DATAS)

        um = b.calcular_benchmarks(DATAS, DATAS[-1], 1.0, ibovespa=ibov, cdi=cdi)
        mil = b.calcular_benchmarks(DATAS, DATAS[-1], 1e6, ibovespa=ibov, cdi=cdi)
        pd.testing.assert_frame_equal(um["resumo"].drop(columns=["patrimonio_inicial", "patrimonio_final"]),
                                      mil["resumo"].drop(columns=["patrimonio_inicial", "patrimonio_final"]))

    def test_cdi_consulta_em_janelas_de_ate_10_anos(self):
        urls = []

        def resposta(requisicao, timeout):
            urls.append(requisicao.full_url)
            return io.BytesIO(json.dumps([{"data": "03/01/2000", "valor": "0.07"}]).encode())

        with patch.object(urllib.request, "urlopen", side_effect=resposta):
            taxas = b.baixar_cdi(date(2000, 1, 1), date(2021, 6, 1))
        self.assertEqual(len(urls), 3)
        self.assertTrue(all("bcdata.sgs.12" in url for url in urls))
        for url in urls:
            parametros = dict(parte.split("=") for parte in url.split("?")[1].split("&"))
            inicio, fim = (pd.to_datetime(parametros[c], format="%d/%m/%Y") for c in ["dataInicial", "dataFinal"])
            self.assertLessEqual((fim - inicio).days + 1, b.JANELA_MAXIMA_DIAS)
        self.assertIn("dataInicial=01/01/2000", urls[0])
        self.assertIn("dataFinal=01/06/2021", urls[-1])
        self.assertEqual(taxas.iloc[0], 0.07)


def precos_sinteticos(codigos, inicio, limite):
    """Histórico diário artificial com o mesmo formato de coletar_historicos."""
    dias = pd.bdate_range(inicio, pd.Timestamp(limite) - pd.Timedelta(days=1))
    gerador = np.random.default_rng(7)
    historicos = {}
    for posicao, codigo in enumerate(codigos):
        retornos = gerador.normal(0.0004 * (posicao + 1), 0.012 + 0.003 * posicao, len(dias))
        historicos[codigo] = pd.Series(30 * np.exp(np.cumsum(retornos)), index=dias,
                                       name=codigo.removesuffix(".SA"))
    return historicos, {codigo: 0 for codigo in codigos}


def ibov_sintetico(inicio, limite):
    semanas = pd.date_range(inicio, pd.Timestamp(limite) - pd.Timedelta(days=1), freq="W-FRI")
    return pd.Series(100_000 * 1.002 ** np.arange(len(semanas)) * (1 + 0.01 * np.sin(np.arange(len(semanas)))),
                     index=semanas)


def cdi_sintetico(inicio, fim):
    return taxas_uteis(inicio, fim, 0.04)


class TestarInterfaceBenchmarks(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()

    def abrir_e_analisar(self):
        at = AppTest.from_file(str(ARQUIVO), default_timeout=180).run()
        at.date_input(key="fim_carteiras").set_value(date(2024, 12, 31))
        at.button(key=BOTAO).click().run()
        self.assertFalse(at.exception)
        return at

    def tabela(self, at):
        return at.tabs[4].get("dataframe")[-1].value

    def grafico(self, at):
        figuras = [json.loads(g.proto.spec) for g in at.tabs[4].get("plotly_chart")]
        return [f for f in figuras if any(t.get("name") == "Igualitária" and t.get("mode") == "lines" for t in f["data"])][-1]

    def test_benchmarks_na_tabela_no_grafico_selecao_e_capital(self):
        with patch("baixar_dados.baixar_precos_diarios", side_effect=ValueError("Regimes fora do teste")), \
                patch("motor_carteiras.coletar_historicos", side_effect=precos_sinteticos), \
                patch("benchmarks.baixar_ibovespa", side_effect=ibov_sintetico) as ibov, \
                patch("benchmarks.baixar_cdi", side_effect=cdi_sintetico) as cdi:
            at = self.abrir_e_analisar()
            estado = at.session_state["analise_carteiras"]
            curva = estado["resultado"]["curva"]
            referencias = estado["benchmarks"]
            self.assertEqual(referencias["indisponiveis"], {})
            pd.testing.assert_index_equal(referencias["curva"].index, curva.index)
            self.assertEqual(referencias["curva"].iloc[0]["ibovespa"], referencias["curva"].iloc[0]["cdi"])
            # O CDI só é consultado até a última semana do teste.
            self.assertEqual(cdi.call_args.args[1], curva.index[-1].date())

            tabela = self.tabela(at)
            for nome, rotulo in [("ibovespa", "Ibovespa"), ("cdi", "CDI")]:
                linha = referencias["resumo"].loc[nome]
                self.assertEqual(tabela.loc["Retorno acumulado", rotulo],
                                 f"{linha['retorno_acumulado_decimal'] * 100:.2f}%".replace(".", ","))
                self.assertEqual(tabela.loc["Drawdown máximo", rotulo],
                                 f"{linha['drawdown_maximo_decimal'] * 100:.2f}%".replace(".", ","))
            estrategias = tabela[["Igualitária", "Mínima volatilidade"]].copy()
            figura = self.grafico(at)
            tracos = {t["name"]: t for t in figura["data"]}
            self.assertEqual(tracos["Ibovespa"]["line"]["dash"], "dash")
            self.assertEqual(tracos["CDI"]["line"]["dash"], "dot")
            self.assertIn("Igualitária", tracos)

            # Selecionar só o CDI remove o Ibovespa da tabela e do gráfico, sem afetar as carteiras.
            seletor = [m for m in at.tabs[4].multiselect if m.label == "Benchmarks"][0]
            seletor.set_value(["cdi"]).run()
            self.assertFalse(at.exception)
            tabela = self.tabela(at)
            self.assertNotIn("Ibovespa", tabela.columns)
            self.assertIn("CDI", tabela.columns)
            pd.testing.assert_frame_equal(tabela[["Igualitária", "Mínima volatilidade"]], estrategias)
            self.assertNotIn("Ibovespa", {t["name"] for t in self.grafico(at)["data"]})

            # Capital: patrimônio do CDI escala; percentuais permanecem.
            percentual = tabela.loc["Retorno acumulado", "CDI"]
            at.number_input(key="capital_carteiras").set_value(250_000.)
            at.button(key=BOTAO).click().run()
            tabela = self.tabela(at)
            final = referencias["curva"]["cdi"].iloc[-1] * 2.5
            self.assertEqual(tabela.loc["Patrimônio final", "CDI"],
                             "R$ " + f"{final:,.2f}".replace(",", "_").replace(".", ",").replace("_", "."))
            self.assertEqual(tabela.loc["Retorno acumulado", "CDI"], percentual)
            self.assertEqual(ibov.call_count, 1)  # Fonte em cache na segunda submissão.

    def test_falha_do_ibovespa_mantem_carteiras_e_cdi(self):
        with patch("baixar_dados.baixar_precos_diarios", side_effect=ValueError("Regimes fora do teste")), \
                patch("motor_carteiras.coletar_historicos", side_effect=precos_sinteticos), \
                patch("benchmarks.baixar_ibovespa", side_effect=ConnectionError("Yahoo fora do ar")), \
                patch("benchmarks.baixar_cdi", side_effect=cdi_sintetico):
            at = self.abrir_e_analisar()
            self.assertTrue(any("Ibovespa indisponível" in w.value and "Yahoo fora do ar" in w.value
                                for w in at.tabs[4].warning))
            tabela = self.tabela(at)
            self.assertIn("Igualitária", tabela.columns)
            self.assertIn("CDI", tabela.columns)
            self.assertNotIn("Ibovespa", tabela.columns)
            self.assertFalse(at.tabs[4].error)


@unittest.skipUnless(os.environ.get("TESTAR_DADOS_REAIS") == "1", "Ative TESTAR_DADOS_REAIS=1 para consultar as fontes")
class TestarFontesReais(unittest.TestCase):
    def test_fontes_reais_e_conferencia_do_cdi(self):
        datas = pd.date_range("2025-06-27", "2025-12-26", freq="W-FRI")
        r = b.calcular_benchmarks(datas, datas[-1] + pd.Timedelta(days=1))
        self.assertEqual(r["indisponiveis"], {})
        pd.testing.assert_index_equal(r["curva"].index, datas)
        url = b.URL_SGS.format(serie=4389, inicio="27/06/2025", fim="25/12/2025")
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "regimes-do-mercado"}), timeout=30) as resposta:
            anual = pd.Series([float(x["valor"]) for x in json.load(resposta)])
        fator_4389 = float(np.prod((1 + anual / 100) ** (1 / 252)))
        self.assertAlmostEqual(r["curva"]["cdi"].iloc[-1] / r["curva"]["cdi"].iloc[0], fator_4389, places=5)


if __name__ == "__main__":
    unittest.main()
