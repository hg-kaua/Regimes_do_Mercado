"""python -m unittest test_maximo_sharpe -v.
TESTAR_INTERFACE_REAL=1 habilita integração com AppTest e dados reais.
"""
from copy import deepcopy
from datetime import date
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest
from pypfopt import EfficientFrontier
import cvxpy as cp

from maximo_sharpe import otimizar_maximo_sharpe, validar_taxa_livre_risco
from motor_carteiras import adicionar_maximo_sharpe, adicionar_personalizada
from test_personalizada import base_matematica
import motor_carteiras


class TestarMaximoSharpe(unittest.TestCase):
    def setUp(self):
        self.ativos = pd.Index(['A', 'B', 'C'])
        self.mu = pd.Series([.06,.09,.12], index=self.ativos)
        self.cov = pd.DataFrame(np.diag([.04,.09,.16]), index=self.ativos, columns=self.ativos)

    def test_solucao_analitica_unidades_e_taxa(self):
        taxa = .03
        esperado = np.linalg.solve(self.cov, self.mu - taxa)
        esperado /= esperado.sum()  # Solução analítica somente para a comparação do teste.
        with patch.object(EfficientFrontier, 'clean_weights', side_effect=AssertionError('Não arredondar')):
            pesos = otimizar_maximo_sharpe(self.mu, self.cov, taxa)
        np.testing.assert_allclose(pesos, esperado, atol=1e-6)
        # Mesma razão em outra unidade temporal se TODOS os termos forem convertidos.
        outros = otimizar_maximo_sharpe(self.mu/52, self.cov/52, taxa/52)
        np.testing.assert_allclose(outros, pesos, atol=1e-6)
        sem_taxa = otimizar_maximo_sharpe(self.mu, self.cov, 0., maximo=1.)
        self.assertFalse(np.allclose(pesos, sem_taxa))
        for taxa_invalida in [float('nan'),float('inf'),True,'10%',-1.]:
            with self.assertRaises(ValueError):
                validar_taxa_livre_risco(taxa_invalida)

    def test_restricoes_desfavoraveis_e_solver(self):
        for mu, taxa, maximo in [(self.mu, .2, .4), (-self.mu, 0., .4),
                                (pd.Series([.12,-.3,-.3],index=self.ativos), 0., .4),
                                (self.mu, .03, .3)]:
            with self.subTest(mu=list(mu),taxa=taxa), patch('pypfopt.EfficientFrontier') as fronteira:
                with self.assertRaises(ValueError):
                    otimizar_maximo_sharpe(mu,self.cov,taxa,maximo)
                fronteira.assert_not_called()
        pesos = otimizar_maximo_sharpe(self.mu,self.cov,0.,maximo=.4)
        self.assertAlmostEqual(pesos.sum(),1.,places=8)
        self.assertTrue((pesos >= -1e-7).all() and (pesos <= .4+1e-7).all())
        with self.assertRaises(ValueError):
            otimizar_maximo_sharpe(self.mu.iloc[::-1],self.cov,0.)
        with self.assertRaises(RuntimeError):
            otimizar_maximo_sharpe(self.mu,self.cov,0.,solver='INEXISTENTE')
        with patch.object(EfficientFrontier,'max_sharpe',side_effect=cp.error.SolverError('Teste')):
            with self.assertRaises(RuntimeError):
                otimizar_maximo_sharpe(self.mu,self.cov,0.)
        with patch.dict(sys.modules,{'pypfopt':None}):
            with self.assertRaises(RuntimeError):
                otimizar_maximo_sharpe(self.mu,self.cov,0.)
        with self.assertRaises(ValueError):
            otimizar_maximo_sharpe(self.mu,self.cov*0,0.)

    def test_treinamento_anualizacao_backtest_e_independencia(self):
        base = base_matematica()
        base['parametros']['peso_maximo'] = .4
        anterior = deepcopy(base)
        # Usa uma taxa negativa hipotética para garantir excesso positivo neste fixture matemático.
        taxa = -.5
        with patch('pypfopt.EfficientFrontier',wraps=EfficientFrontier) as chamada:
            resultado = adicionar_maximo_sharpe(base,taxa)
        np.testing.assert_allclose(chamada.call_args.kwargs['expected_returns'],base['treinamento'].mean()*52)
        np.testing.assert_allclose(chamada.call_args.kwargs['cov_matrix'],base['treinamento'].cov()*52)
        for chave in ['pesos','curva','resumo','resumo_otimizacao']:
            pd.testing.assert_frame_equal(base[chave],anterior[chave],check_exact=True)
            pd.testing.assert_frame_equal(resultado[chave].loc[base[chave].index,base[chave].columns],base[chave],check_exact=True)
        pesos = resultado['pesos']['peso_maximo_sharpe_decimal']
        serie = base['treinamento'] @ pesos
        np.testing.assert_allclose(resultado['resumo_otimizacao'].loc['maximo_sharpe'],
                                  [serie.mean(),serie.mean()*52,serie.std(ddof=1),serie.std(ddof=1)*np.sqrt(52)],rtol=1e-12)
        curva = resultado['curva']['patrimonio_maximo_sharpe']
        esperado = (base['precos'].loc[curva.index] / base['precos'].loc[curva.index[0]]) @ pesos * 100000
        np.testing.assert_allclose(curva,esperado,rtol=1e-12)
        alterada = deepcopy(base)
        alterada['teste'].iloc[-1] = [.2,-.1,.3]
        alterada['precos'].loc[alterada['teste'].index] = ((1+alterada['teste']).cumprod() *
            alterada['precos'].loc[pd.Timestamp(alterada['periodo']['fim_treinamento'])]).to_numpy()
        outro = adicionar_maximo_sharpe(alterada,taxa)
        pd.testing.assert_series_equal(pesos,outro['pesos']['peso_maximo_sharpe_decimal'],check_exact=True)
        pd.testing.assert_frame_equal(resultado['resumo_otimizacao'],outro['resumo_otimizacao'],check_exact=True)
        pd.testing.assert_series_equal(curva.iloc[:-1],outro['curva']['patrimonio_maximo_sharpe'].iloc[:-1],check_exact=True)
        manual = adicionar_personalizada(resultado,{'A':100,'B':0,'C':0})
        pd.testing.assert_series_equal(pesos,manual['pesos']['peso_maximo_sharpe_decimal'],check_exact=True)
        indice = lambda w: (float(w @ (base['treinamento'].mean()*52))-taxa) / np.sqrt(float(w @ (base['treinamento'].cov()*52) @ w))
        self.assertGreaterEqual(indice(pesos)+1e-8,indice(base['pesos']['peso_igualitario_decimal']))
        self.assertGreaterEqual(indice(pesos)+1e-8,indice(base['pesos']['peso_otimizado_decimal']))


@unittest.skipUnless(os.environ.get('TESTAR_INTERFACE_REAL')=='1','Ative TESTAR_INTERFACE_REAL=1 para AppTest real')
class TestarSharpeInterface(unittest.TestCase):
    def test_taxa_cache_resultados_e_inviabilidade(self):
        st.cache_data.clear()
        with patch('motor_carteiras.analisar_carteiras',wraps=motor_carteiras.analisar_carteiras) as motor:
            at=AppTest.from_file(str(Path(__file__).with_name('app.py')),default_timeout=120).run()
            botao='FormSubmitter:formulario_carteiras-Analisar carteira'
            at.date_input(key='fim_carteiras').set_value(date(2026,11,13))
            at.number_input(key='taxa_livre_risco_carteiras').set_value(5.)
            at.button(key=botao).click().run()
            self.assertFalse(at.exception)
            estado=at.session_state['analise_carteiras']
            self.assertEqual(estado['taxa_anual'],.05)
            self.assertEqual(estado['resultado_sharpe']['parametros']['taxa_livre_risco_anual_decimal'],.05)
            self.assertIn('Máximo Sharpe',at.tabs[4].get('dataframe')[-1].value)
            original=estado['resultado']['pesos'].copy()
            esperado=estado['resultado_sharpe']['resumo'].loc['maximo_sharpe','retorno_acumulado_decimal']
            self.assertEqual(at.tabs[4].get('dataframe')[-1].value.loc['Retorno acumulado','Máximo Sharpe'],f'{esperado*100:.2f}%'.replace('.',','))
            at.number_input(key='taxa_livre_risco_carteiras').set_value(100.)
            at.button(key=botao).click().run()
            self.assertFalse(at.exception)
            self.assertEqual(motor.call_count,1)
            self.assertTrue(any('Nenhuma carteira' in w.value for w in at.warning))
            tabela=at.tabs[4].get('dataframe')[-1].value
            self.assertNotIn('Máximo Sharpe',tabela)
            # Etapa 10: Ibovespa e CDI são colunas de referência, não estratégias.
            self.assertEqual([c for c in tabela if c not in ('Ibovespa','CDI')],['Igualitária','Mínima volatilidade','Personalizada'])
            pd.testing.assert_frame_equal(original,at.session_state['analise_carteiras']['resultado']['pesos'])
            at.number_input(key='taxa_livre_risco_carteiras').set_value(5.)
            at.button(key=botao).click().run()
            self.assertIn('Máximo Sharpe',at.tabs[4].get('dataframe')[-1].value)
            pesos=at.session_state['analise_carteiras']['resultado_sharpe']['pesos'].copy()
            at.number_input(key='capital_carteiras').set_value(200000.)
            at.button(key=botao).click().run()
            self.assertEqual(motor.call_count,1)
            pd.testing.assert_frame_equal(pesos,at.session_state['analise_carteiras']['resultado_sharpe']['pesos'])


if __name__=='__main__':
    unittest.main()
