"""python -m unittest test_personalizada -v.

Testes matemáticos sintéticos; TESTAR_INTERFACE_REAL=1 habilita AppTest com
dados reais. Nenhum dado artificial é gravado nos resultados financeiros.
"""
from copy import deepcopy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from analisar_carteira import analisar_retornos, separar_conjuntos
from otimizar_carteira import otimizar_resultados
from backtest_carteira import executar_backtest
from dados_carteiras import ajustar_capital
from motor_carteiras import adicionar_personalizada
import motor_carteiras


def base_matematica():
    retornos = pd.DataFrame(np.random.default_rng(42).normal(.1, 2, (100, 3)),
                            index=pd.date_range('2020-01-10', periods=100, freq='W-FRI'), columns=['A', 'B', 'C'])
    precos = pd.concat([pd.DataFrame([[100.]*3], columns=retornos.columns, index=[pd.Timestamp('2020-01-03')]),
                        100 * (1 + retornos/100).cumprod()])
    estatisticas = analisar_retornos(retornos)
    otimizacao = otimizar_resultados(estatisticas)
    periodo = estatisticas['resumo_carteira_igualitaria.csv'].iloc[0]
    backtest, posicoes = executar_backtest(retornos, otimizacao['pesos_otimizados.csv'],
                                          otimizacao['resumo_otimizacao.csv'], periodo)
    treino, teste = separar_conjuntos(retornos)
    return {'parametros': {'capital_inicial': 100000.}, 'precos': precos,
            'retornos_percentuais': retornos, 'treinamento': treino, 'teste': teste,
            'resumo_risco': estatisticas['resumo_risco.csv'], 'covariancia_semanal': estatisticas['matriz_covariancia.csv'],
            'pesos': otimizacao['pesos_otimizados.csv'], 'resumo_otimizacao': otimizacao['resumo_otimizacao.csv'],
            'periodo': periodo, 'resumo': backtest['resumo_backtest.csv'],
            'curva': backtest['curva_patrimonial_backtest.csv'], 'posicoes': posicoes}


class TestarPersonalizada(unittest.TestCase):
    def setUp(self):
        self.base = base_matematica()

    def test_pesos_invalidos(self):
        for pesos in [{'A': 0, 'B': 0, 'C': 0}, {'A': 30, 'B': 30, 'C': 30}, {'A': 50, 'B': 50.01, 'C': 0},
                      {'A': -1, 'B': 51, 'C': 50}, {'A': 101, 'B': 0, 'C': 0},
                      {'A': float('nan'), 'B': 50, 'C': 50}, {'A': float('inf'), 'B': 0, 'C': 0},
                      {'A': '50', 'B': 50, 'C': 0}, {'A': True, 'B': 49, 'C': 50},
                      {'A': 50, 'B': 50}, {'A': 50, 'B': 50, 'D': 0},
                      pd.Series([50, 50, 0], index=['A', 'A', 'C'])]:
            with self.subTest(pesos=pesos), self.assertRaises(ValueError):
                adicionar_personalizada(self.base, pesos)

    def test_zero_concentracao_e_ordem(self):
        resultado = adicionar_personalizada(self.base, {'C': 0, 'B': 0, 'A': 100})
        np.testing.assert_array_equal(resultado['pesos']['peso_personalizado_decimal'], [1, 0, 0])
        curva = resultado['curva']['patrimonio_personalizada']
        esperado = self.base['precos'].loc[curva.index, 'A'] / self.base['precos'].loc[curva.index[0], 'A'] * 100000
        np.testing.assert_allclose(curva, esperado, rtol=1e-12)
        self.assertTrue((resultado['posicoes']['personalizada'][['B', 'C']] == 0).all().all())
        self.assertEqual(resultado['resumo'].loc['personalizada', 'semanas_teste'], self.base['periodo']['semanas_teste'])

    def test_igualitaria_e_capital(self):
        resultado = adicionar_personalizada(self.base, dict.fromkeys(['A','B','C'], 100/3))
        np.testing.assert_allclose(resultado['curva']['patrimonio_personalizada'], resultado['curva']['patrimonio_igualitaria'], rtol=1e-12)
        np.testing.assert_allclose(resultado['resumo_otimizacao'].loc['personalizada'], resultado['resumo_otimizacao'].loc['igualitaria'], rtol=1e-12)
        patrimonio = ajustar_capital(resultado, 200000)
        self.assertEqual(list(patrimonio), ['igualitaria','minima_volatilidade','personalizada'])
        np.testing.assert_allclose(patrimonio['personalizada'], resultado['curva']['patrimonio_personalizada'] * 2)

    def test_estimativas_independencia_e_vazamento(self):
        anterior = deepcopy(self.base)
        with patch('motor_carteiras.otimizar_resultados', side_effect=AssertionError('Não deve otimizar novamente')):
            resultado = adicionar_personalizada(self.base, {'A': 70, 'B': 30, 'C': 0})
        for chave in ['pesos','curva','resumo','resumo_otimizacao']:
            pd.testing.assert_frame_equal(self.base[chave], anterior[chave], check_exact=True)
            pd.testing.assert_frame_equal(resultado[chave].loc[anterior[chave].index, anterior[chave].columns], anterior[chave], check_exact=True)
        serie = self.base['treinamento'] @ np.array([.7, .3, 0])
        np.testing.assert_allclose(resultado['resumo_otimizacao'].loc['personalizada'],
                                   [serie.mean(), serie.mean()*52, serie.std(ddof=1), serie.std(ddof=1)*np.sqrt(52)], rtol=1e-12)
        alterada = deepcopy(self.base)
        alterada['teste'].iloc[-1] = [.3, -.2, .1]
        # Preços de teste coerentes com os novos retornos, mantendo o treinamento intacto.
        alterada['precos'].loc[alterada['teste'].index] = ((1 + alterada['teste']).cumprod() *
                                                        alterada['precos'].loc[pd.Timestamp(alterada['periodo']['fim_treinamento'])]).to_numpy()
        outro = adicionar_personalizada(alterada, {'A': 70, 'B': 30, 'C': 0})
        pd.testing.assert_frame_equal(resultado['resumo_otimizacao'], outro['resumo_otimizacao'], check_exact=True)
        pd.testing.assert_series_equal(resultado['curva']['patrimonio_personalizada'].iloc[:-1],
                                       outro['curva']['patrimonio_personalizada'].iloc[:-1], check_exact=True)
        self.assertNotEqual(resultado['curva']['patrimonio_personalizada'].iloc[-1], outro['curva']['patrimonio_personalizada'].iloc[-1])
        resultado['pesos'].iloc[0,0] = 0
        pd.testing.assert_frame_equal(self.base['pesos'], anterior['pesos'], check_exact=True)


@unittest.skipUnless(os.environ.get('TESTAR_INTERFACE_REAL') == '1', 'Ative TESTAR_INTERFACE_REAL=1 para AppTest real')
class TestarPersonalizadaInterface(unittest.TestCase):
    def test_edicao_igualar_capital_e_troca_de_ativos(self):
        st.cache_data.clear()
        with patch('motor_carteiras.analisar_carteiras', wraps=motor_carteiras.analisar_carteiras) as motor:
            at = AppTest.from_file(str(Path(__file__).with_name('app.py')), default_timeout=120).run()
            enviar = 'FormSubmitter:formulario_carteiras-Analisar carteira'
            at.button(key=enviar).click().run()
            self.assertFalse(at.exception)
            base = at.session_state['analise_carteiras']['resultado']
            pesos_base = base['pesos'].copy()
            self.assertEqual(motor.call_count, 1)
            self.assertIn('Personalizada', at.tabs[4].get('dataframe')[-1].value.columns)
            for ativo in base['pesos'].index:
                at.number_input(key=f'peso_manual_{ativo}').set_value(100. if ativo == 'PETR4' else 0.)
            at.run()
            self.assertFalse(at.exception)
            self.assertEqual(motor.call_count, 1)
            esperado = adicionar_personalizada(base, {a: 100. if a == 'PETR4' else 0. for a in pesos_base.index})
            valor = esperado['resumo'].loc['personalizada','patrimonio_final']
            formatado = 'R$ ' + f'{valor:,.2f}'.replace(',', '_').replace('.', ',').replace('_', '.')
            tabela = at.tabs[4].get('dataframe')[-1].value
            self.assertEqual(tabela.loc['Patrimônio final','Personalizada'], formatado)
            at.number_input(key='peso_manual_PETR4').set_value(90.).run()
            self.assertTrue(any('Soma informada' in e.value for e in at.error))
            self.assertNotIn('Personalizada', at.tabs[4].get('dataframe')[-1].value.columns)
            pd.testing.assert_frame_equal(at.session_state['analise_carteiras']['resultado']['pesos'], pesos_base, check_exact=True)
            at.button(key='igualar_personalizada').click().run()
            tabela = at.tabs[4].get('dataframe')[-1].value
            pd.testing.assert_series_equal(tabela['Igualitária'], tabela['Personalizada'], check_names=False)
            at.number_input(key='capital_carteiras').set_value(200000.)
            at.button(key=enviar).click().run()
            self.assertEqual(motor.call_count, 1)
            self.assertFalse(at.exception)
            at.multiselect(key='ativos_carteiras').set_value(at.multiselect(key='ativos_carteiras').value[:3])
            at.button(key=enviar).click().run()
            self.assertEqual(motor.call_count, 2)
            self.assertFalse(at.exception)
            self.assertEqual(len([x for x in at.number_input if x.key.startswith('peso_manual_')]), 3)
            tabela = at.tabs[4].get('dataframe')[-1].value
            pd.testing.assert_series_equal(tabela['Igualitária'], tabela['Personalizada'], check_names=False)


if __name__ == '__main__':
    unittest.main()
