"""Testes locais: python -m unittest test_motor_carteiras -v.

Dados artificiais são usados somente nos testes matemáticos de covariância,
restrições e vazamento. As entradas inválidas devem falhar antes da coleta.
Para executar também a coleta real e a comparação com os CSVs tradicionais,
defina TESTAR_DADOS_REAIS=1 no ambiente. Esses testes exigem rede e os CSVs
originais em dados/carteiras; nunca os sobrescrevem.
"""
from contextlib import redirect_stdout
from datetime import date
from io import StringIO
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import motor_carteiras as motor
from analisar_carteira import analisar_retornos, ler_tabela
from otimizar_carteira import otimizar_resultados, validar_covariancia, validar_pesos
from backtest_carteira import conferir_com_precos
from dados_carteiras import ajustar_capital


class TestarMotorCarteiras(unittest.TestCase):
    def test_normalizacao_e_datas(self):
        codigos, inicio, fim, limite = motor.validar_parametros(
            [" petr4 ", "vale3.sa", "ITUB4"], "2020-01-01", date(2024, 12, 31), .5)
        self.assertEqual(codigos, ["PETR4.SA", "VALE3.SA", "ITUB4.SA"])
        self.assertEqual((inicio, fim, limite), (date(2020, 1, 1), date(2024, 12, 31), .5))
        codigos, *_ = motor.validar_parametros(['B3SA3', 'PETR4', 'VALE3'], inicio, fim, .5)
        self.assertEqual(codigos[0], 'B3SA3.SA')

    def test_entradas_invalidas_antes_da_rede(self):
        padrao = (["PETR4", "VALE3", "ITUB4"], "2020-01-01", "2024-12-31", .4)
        casos = [
            (["PETR4", "petr4.sa", "VALE3"], *padrao[1:]),
            (["PETR4", "NAO_EXISTE", "VALE3"], *padrao[1:]),
            (["PETR4", None, "VALE3"], *padrao[1:]),
            (["PETR4", "VALE3"], *padrao[1:]),
            (["AAAA" + str(i) for i in range(13)], *padrao[1:]),
            ("PETR4 VALE3 ITUB4", *padrao[1:]),
            (padrao[0], "2024-02-30", padrao[2], .4),
            (padrao[0], "01/01/2020", padrao[2], .4),
            (padrao[0], padrao[2], padrao[1], .4),
            (padrao[0], padrao[1], "2099-12-31", .4),
        ]
        casos += [(*padrao[:3], limite) for limite in [.3, 0, -1, 1.1, float('nan'), float('inf'), True, '40%']]
        with patch.object(motor, "coletar_historicos") as coleta:
            for parametros in casos:
                with self.subTest(parametros=parametros), self.assertRaises(ValueError):
                    motor.analisar_carteiras(*parametros)
            coleta.assert_not_called()

    def test_falha_da_fonte_nao_reutiliza_resultados(self):
        with patch.object(motor, "coletar_historicos", side_effect=ValueError("Ativo sem histórico")), \
                patch.object(motor, "otimizar_resultados") as otimizar:
            with self.assertRaisesRegex(ValueError, "sem histórico"):
                motor.analisar_carteiras(["PETR4", "VALE3", "ITUB4"], "2020-01-01", "2024-12-31")
            otimizar.assert_not_called()

    def test_matematica_limites_matrizes_e_vazamento(self):
        # Séries artificiais determinísticas exclusivamente para validar a matemática.
        gerador = np.random.default_rng(42)
        for quantidade, maximo in [(3, 1/3), (3, .8), (5, .4), (6, .25), (12, .15)]:
            with self.subTest(quantidade=quantidade, maximo=maximo):
                retornos = pd.DataFrame(gerador.normal(0, 2, (100, quantidade)),
                                        columns=[f"Ativo{i}" for i in range(quantidade)],
                                        index=pd.date_range("2020-01-10", periods=100, freq="W-FRI"))
                estatisticas = analisar_retornos(retornos)
                resultado = otimizar_resultados(estatisticas, maximo=maximo)
                cov = estatisticas["matriz_covariancia.csv"]
                validar_covariancia(cov)
                np.testing.assert_allclose(cov, (retornos.iloc[:80] / 100).cov(), atol=1e-15)
                for coluna in resultado["pesos_otimizados.csv"]:
                    validar_pesos(resultado["pesos_otimizados.csv"][coluna], cov.index, maximo=maximo)
                alterados = retornos.copy()
                alterados.iloc[80:] *= -10
                outro = otimizar_resultados(analisar_retornos(alterados), maximo=maximo)
                for nome in resultado:
                    pd.testing.assert_frame_equal(resultado[nome], outro[nome], check_exact=True)

    def test_conferencia_backtest_em_memoria(self):
        datas = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
        precos = pd.DataFrame({"A": [10., 11., 12.], "B": [20., 18., 22.]}, index=datas)
        pesos = pd.DataFrame({"peso_igualitario_decimal": [.5, .5],
                              "peso_otimizado_decimal": [.3, .7]}, index=precos.columns)
        posicoes = {nome: precos / precos.iloc[0] * (100 * pesos[coluna]) for nome, coluna in
                    [("igualitaria", "peso_igualitario_decimal"), ("minima_volatilidade", "peso_otimizado_decimal")]}
        with patch('backtest_carteira.ler_tabela', side_effect=AssertionError("Não deve ler CSV")):
            conferir_com_precos(posicoes, pesos, 100, precos=precos)
            posicoes["igualitaria"].iloc[-1, 0] += 1
            with self.assertRaises(AssertionError):
                conferir_com_precos(posicoes, pesos, 100, precos=precos)


@unittest.skipUnless(os.environ.get("TESTAR_DADOS_REAIS") == "1", "Ative TESTAR_DADOS_REAIS=1 para a integração real")
class TestarMotorComDadosReais(unittest.TestCase):
    originais = ["PETR4", "VALE3", "ITUB4", "BBAS3", "ABEV3"]

    def test_comparacao_historico_arquivado(self):
        # Dados reais arquivados eliminam o efeito das revisões posteriores da fonte.
        pasta = Path(__file__).parent / "dados" / "carteiras"
        precos = ler_tabela(pasta / "precos_semanais.csv")
        periodo = pd.read_csv(pasta / "resumo_carteira_igualitaria.csv", index_col=0).iloc[0]

        def coleta(codigos, inicio, limite):
            return ({c: precos[c.removesuffix('.SA')].copy() for c in codigos}, {c: 0 for c in codigos})

        with patch.object(motor, "coletar_historicos", side_effect=coleta), \
                patch.object(pd, "read_csv", side_effect=AssertionError("Não deve ler CSV")), \
                patch.object(pd.DataFrame, "to_csv", side_effect=AssertionError("Não deve salvar CSV")), \
                redirect_stdout(StringIO()):
            resultado = motor.analisar_carteiras(list(precos.columns), precos.index[0].date(), str(periodo['fim_teste']))
        arquivos = {"pesos": "pesos_otimizados.csv", "resumo_otimizacao": "resumo_otimizacao.csv",
                    "resumo_risco": "resumo_risco.csv", "correlacao": "matriz_correlacao.csv",
                    "covariancia_semanal": "matriz_covariancia.csv", "resumo": "resumo_backtest.csv",
                    "curva": "curva_patrimonial_backtest.csv"}
        for chave, nome in arquivos.items():
            esperado = resultado[chave]
            numericas = [c for c in esperado if pd.to_numeric(esperado[c], errors='coerce').notna().all()]
            salvo = pd.read_csv(pasta / nome, index_col=0, keep_default_na=False,
                                converters={c: float for c in numericas})
            if chave == 'curva':
                salvo.index = pd.to_datetime(salvo.index)
            self.assertTrue(esperado.index.equals(salvo.index))
            self.assertEqual(list(esperado.columns), list(salvo.columns))
            for coluna in esperado:
                if coluna in numericas:
                    np.testing.assert_allclose(esperado[coluna].astype(float), salvo[coluna].astype(float), rtol=1e-9, atol=1e-8)
                else:
                    self.assertEqual(esperado[coluna].astype(str).tolist(), salvo[coluna].astype(str).tolist())

    def test_coleta_tres_cinco_seis_ativos(self):
        anteriores = []
        for ativos, inicio, fim, teto in [
            (self.originais, '2020-01-01', '2026-11-13', .4),
            (self.originais[:3], '2021-03-10', '2024-12-31', .6),
            (self.originais + ['BBDC4'], '2022-01-01', '2025-12-18', .25),
        ]:
            with self.subTest(ativos=ativos), redirect_stdout(StringIO()), \
                    patch.object(pd, 'read_csv', side_effect=AssertionError('Não deve ler CSV')), \
                    patch.object(pd.DataFrame, 'to_csv', side_effect=AssertionError('Não deve salvar CSV')):
                resultado = motor.analisar_carteiras(ativos, inicio, fim, teto)
            self.assertEqual(list(resultado['precos'].columns), ativos)
            self.assertEqual(len(resultado['treinamento']), int(.8 * len(resultado['retornos_percentuais'])))
            self.assertLess(resultado['treinamento'].index[-1], resultado['teste'].index[0])
            for chave in ['precos', 'retornos_percentuais', 'pesos', 'correlacao', 'covariancia_semanal', 'curva']:
                self.assertTrue(np.isfinite(resultado[chave].to_numpy()).all())
            np.testing.assert_allclose(resultado['covariancia_semanal'], resultado['treinamento'].cov(), atol=1e-15)
            validar_covariancia(resultado['covariancia_semanal'])
            for coluna in resultado['pesos']:
                validar_pesos(resultado['pesos'][coluna], resultado['precos'].columns, maximo=teto)
            np.testing.assert_allclose(ajustar_capital(resultado, 200000), resultado['curva'].iloc[:, :2] * 2)
            if len(ativos) == 6:
                self.assertEqual(resultado['precos'].index[-1], pd.Timestamp('2025-12-12'))
            # As próximas chamadas não podem alterar tabelas retornadas anteriormente.
            for anterior, copia in anteriores:
                pd.testing.assert_frame_equal(anterior['pesos'], copia, check_exact=True)
            anteriores.append((resultado, resultado['pesos'].copy()))

    def test_erros_na_fonte_e_historico_insuficiente(self):
        with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, 'insuficiente'):
            motor.analisar_carteiras(self.originais[:3], '2026-09-01', '2026-11-13')
        with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, 'ZZZZ99'):
            motor.analisar_carteiras(['PETR4', 'VALE3', 'ZZZZ99'], '2024-01-01', '2025-12-31')


if __name__ == "__main__":
    unittest.main()
