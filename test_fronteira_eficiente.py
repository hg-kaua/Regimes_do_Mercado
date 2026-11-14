"""unittest test_fronteira_eficiente; TESTAR_INTERFACE_REAL=1 ativa Yahoo/AppTest."""
from copy import deepcopy
from datetime import date
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier
import cvxpy as cp
import streamlit as st
from streamlit.testing.v1 import AppTest

from fronteira_eficiente import calcular_fronteira
from test_personalizada import base_matematica


class TestarFronteira(unittest.TestCase):
    def setUp(self):
        self.base = base_matematica()
        self.treino = self.base['treinamento']

    def test_pontos_restricoes_unidades_e_correspondencia(self):
        original = self.treino.copy()
        with patch.object(EfficientFrontier, 'clean_weights', side_effect=AssertionError('Não arredondar')):
            resultado = calcular_fronteira(self.treino, .4)
        self.assertFalse(resultado['falhas'])
        self.assertEqual(len(resultado['pesos']), 31)
        mu, cov = self.treino.mean()*52, self.treino.cov()*52
        for indice, pesos in resultado['pesos'].iterrows():
            ponto = resultado['pontos'].loc[indice]
            self.assertAlmostEqual(pesos.sum(), 1., places=7)
            self.assertTrue(((pesos >= -1e-7) & (pesos <= .4+1e-7)).all())
            self.assertAlmostEqual(ponto.retorno_esperado, float(pesos @ mu), places=12)
            self.assertAlmostEqual(ponto.volatilidade, np.sqrt(float(pesos @ cov @ pesos)), places=12)
            self.assertAlmostEqual(ponto.retorno_alvo, ponto.retorno_esperado, places=7)
        self.assertTrue((resultado['pontos'][['retorno_esperado','volatilidade']].diff().iloc[1:] >= -1e-8).all().all())
        np.testing.assert_allclose(resultado['pesos'].iloc[0], self.base['pesos']['peso_otimizado_decimal'], atol=1e-7)
        # API especializada independente confirma uma solução interior.
        alvo = float(resultado['pontos'].iloc[15].retorno_alvo)
        ef = EfficientFrontier(mu, cov, weight_bounds=(0,.4), solver='CLARABEL',
                               solver_options={'tol_gap_abs':1e-11,'tol_gap_rel':1e-11,'tol_feas':1e-11})
        ef.efficient_return(alvo)
        np.testing.assert_allclose(resultado['pesos'].iloc[15], ef.weights, atol=1e-6)
        # O máximo viável com 3 ativos e teto .4 tem .4/.4/.2 nos maiores retornos.
        self.assertAlmostEqual(resultado['retorno_maximo'], float(mu.sort_values(ascending=False) @ [.4,.4,.2]), places=7)
        pd.testing.assert_frame_equal(self.treino, original, check_exact=True)

    def test_continuidade_e_sem_dados_futuros(self):
        a = calcular_fronteira(self.treino, .4, quantidade=11)
        b = calcular_fronteira(self.treino, .4, quantidade=21)
        self.assertFalse(a['falhas'] or b['falhas'])
        np.testing.assert_allclose(a['pontos'], b['pontos'].iloc[::2], atol=1e-8)
        np.testing.assert_allclose(a['pesos'], b['pesos'].iloc[::2], atol=1e-6)
        outra = deepcopy(self.base)
        outra['teste'][:] = .5
        outra['precos'].iloc[-10:] *= 10
        c = calcular_fronteira(outra['treinamento'], .4, quantidade=11)
        pd.testing.assert_frame_equal(a['pontos'], c['pontos'], check_exact=True)
        pd.testing.assert_frame_equal(a['pesos'], c['pesos'], check_exact=True)

    def test_estimativas_negativas_colapso_e_invalidas(self):
        negativos = self.treino - .02
        resultado = calcular_fronteira(negativos, .4)
        self.assertFalse(resultado['falhas'])
        self.assertTrue((resultado['pontos'].retorno_esperado < 0).all())
        unico = calcular_fronteira(self.treino, 1/3)
        self.assertEqual(len(unico['pesos']), 1)
        np.testing.assert_allclose(unico['pesos'], 1/3, atol=1e-7)
        mesma_media = self.treino - self.treino.mean() + .001
        self.assertEqual(len(calcular_fronteira(mesma_media, .4)['pesos']), 1)
        for treino, teto in [(self.treino,.3), (self.treino*np.nan,.4),
                             (self.treino*0,.4), (self.treino.iloc[::-1],.4),
                             (self.treino-2,.4)]:
            with self.subTest(teto=teto), self.assertRaises(ValueError):
                calcular_fronteira(treino,teto)

    def test_falha_interior_nao_inventa_nem_conecta(self):
        original = EfficientFrontier.min_volatility
        chamadas = 0
        def executar(instancia):
            nonlocal chamadas
            chamadas += 1
            if chamadas == 4:
                raise cp.error.SolverError('Falha intencional do teste')
            return original(instancia)
        with patch.object(EfficientFrontier,'min_volatility',executar):
            resultado = calcular_fronteira(self.treino,.4,quantidade=7)
        self.assertEqual(len(resultado['falhas']),1)
        self.assertNotIn(3,resultado['pesos'].index)
        self.assertTrue(resultado['pontos'].loc[3,['retorno_esperado','volatilidade']].isna().all())
        self.assertEqual(len(resultado['pesos']),6)
        with patch.object(EfficientFrontier,'min_volatility',side_effect=cp.error.SolverError('Falha')):
            with self.assertRaises(RuntimeError):
                calcular_fronteira(self.treino,.4)


@unittest.skipUnless(os.environ.get('TESTAR_INTERFACE_REAL')=='1','Ative TESTAR_INTERFACE_REAL=1 para dados reais')
class TestarFronteiraInterface(unittest.TestCase):
    def test_alvo_cache_taxa_capital_e_nova_selecao(self):
        st.cache_data.clear()
        with patch('fronteira_eficiente.calcular_fronteira',wraps=calcular_fronteira) as calculo:
            at = AppTest.from_file(str(Path(__file__).with_name('app.py')),default_timeout=150).run()
            botao = 'FormSubmitter:formulario_carteiras-Analisar carteira'
            at.date_input(key='fim_carteiras').set_value(date(2026,11,13))
            at.number_input(key='taxa_livre_risco_carteiras').set_value(5.)
            at.button(key=botao).click().run()
            self.assertFalse(at.exception)
            estado = at.session_state['analise_carteiras']
            frente = estado['fronteira']
            self.assertIsNotNone(frente,estado['erro_fronteira'])
            self.assertFalse(frente['falhas'])
            pd.testing.assert_frame_equal(calculo.call_args.args[0],estado['resultado']['treinamento'])
            self.assertEqual(calculo.call_count,1)
            at.selectbox(key='alvo_fronteira').select(15).run()
            self.assertFalse(at.exception)
            self.assertEqual(calculo.call_count,1)
            tabelas = [x.value for x in at.tabs[4].get('dataframe')]
            pesos_ui = next(x for x in tabelas if 'Peso no retorno-alvo' in x)
            self.assertEqual(list(pesos_ui['Peso no retorno-alvo']),
                             [f'{p*100:.2f}%'.replace('.',',') for p in frente['pesos'].loc[15]])
            figuras = [json.loads(g.proto.spec) for g in at.tabs[4].get('plotly_chart')]
            figura = next(g for g in figuras if g['data'][0]['name']=='Fronteira eficiente')
            self.assertFalse(figura['data'][0]['connectgaps'])
            self.assertIn('% a.a.',figura['layout']['xaxis']['title']['text'])
            self.assertIn('% a.a.',figura['layout']['yaxis']['title']['text'])
            nomes = [t['name'] for t in figura['data']]
            self.assertIn('Máximo Sharpe',nomes)
            self.assertIn('Mínima volatilidade',nomes)
            self.assertIn('Igualitária',nomes)
            # Séries de pontos são percentuais; os dados da API continuam decimais.
            import base64
            valores_x = figura['data'][0]['x']
            x = (np.frombuffer(base64.b64decode(valores_x['bdata']),dtype=valores_x['dtype'])
                 if isinstance(valores_x,dict) else valores_x)
            estimativa_sharpe = estado['resultado_sharpe']['resumo_otimizacao'].loc['maximo_sharpe']
            esperados = pd.concat([frente['pontos'], pd.DataFrame([{
                'retorno_alvo':estimativa_sharpe['retorno_medio_anualizado_aritmetico_decimal'],
                'volatilidade':estimativa_sharpe['volatilidade_anualizada_decimal'],
            }])]).sort_values('retorno_alvo',kind='stable')
            np.testing.assert_allclose(x,esperados['volatilidade']*100)
            at.number_input(key='capital_carteiras').set_value(200000.)
            at.button(key=botao).click().run()
            self.assertEqual(calculo.call_count,1)
            self.assertEqual(at.selectbox(key='alvo_fronteira').value,15)
            at.number_input(key='taxa_livre_risco_carteiras').set_value(100.)
            at.button(key=botao).click().run()
            self.assertFalse(at.exception)
            self.assertEqual(calculo.call_count,1)
            pd.testing.assert_frame_equal(frente['pontos'],at.session_state['analise_carteiras']['fronteira']['pontos'])
            figura = next(json.loads(g.proto.spec) for g in at.tabs[4].get('plotly_chart')
                          if json.loads(g.proto.spec)['data'][0]['name']=='Fronteira eficiente')
            self.assertNotIn('Máximo Sharpe',[t['name'] for t in figura['data']])
            at.multiselect(key='ativos_carteiras').set_value(at.multiselect(key='ativos_carteiras').value[:3])
            at.date_input(key='fim_carteiras').set_value(date(2024,12,31))
            at.button(key=botao).click().run()
            self.assertFalse(at.exception)
            self.assertEqual(calculo.call_count,2)
            self.assertEqual(len(at.session_state['analise_carteiras']['fronteira']['pesos'].columns),3)
            at.number_input(key='limite_carteiras').set_value(50.)
            at.button(key=botao).click().run()
            self.assertEqual(calculo.call_count,3)
            at.number_input(key='limite_carteiras').set_value(20.)
            at.button(key=botao).click().run()
            self.assertNotIn('analise_carteiras',at.session_state)
            self.assertFalse(at.tabs[4].get('plotly_chart'))


if __name__ == '__main__':
    unittest.main()
