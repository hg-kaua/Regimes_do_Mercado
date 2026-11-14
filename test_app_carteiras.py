"""AppTest: python -m unittest test_app_carteiras -v.

Defina TESTAR_INTERFACE_REAL=1 para testar também consultas reais e indicadores.
Não grava CSVs nem altera os algoritmos financeiros. Falhas são simuladas apenas
para verificar as mensagens e a remoção dos resultados anteriores.
"""
from datetime import date
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

import baixar_dados
import motor_carteiras


ARQUIVO = Path(__file__).with_name('app.py')
BOTAO = 'FormSubmitter:formulario_carteiras-Analisar carteira'
PADRAO = ['PETR4 — Petrobras', 'VALE3 — Vale', 'ITUB4 — Itaú Unibanco',
          'BBAS3 — Banco do Brasil', 'ABEV3 — Ambev']


def moeda(valor):
    return 'R$ ' + f'{valor:,.2f}'.replace(',', '_').replace('.', ',').replace('_', '.')


class TestarFormularioCarteiras(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()

    def test_disponivel_sem_csv_e_sem_analise_de_regimes(self):
        with patch('baixar_dados.baixar_precos_diarios', side_effect=ValueError('Fonte indisponível')), \
                patch('dados_carteiras.carregar_carteiras', side_effect=AssertionError('CSV indevido')), \
                patch('motor_carteiras.analisar_carteiras') as motor:
            at = AppTest.from_file(str(ARQUIVO), default_timeout=30).run()
            self.assertFalse(at.exception)
            self.assertEqual([t.label for t in at.tabs],
                             ['Visão geral', 'K-Means', 'Cadeias de Markov', 'Dados e normalização', 'Carteiras'])
            self.assertEqual(at.multiselect(key='ativos_carteiras').value, PADRAO)
            self.assertFalse(at.tabs[4].get('metric'))
            for selecionados, extra, limite, inicio, fim, mensagem in [
                (PADRAO[:2], '', 40., date(2020,1,1), date(2025,1,1), '3 a 12'),
                (PADRAO, 'petr4.sa', 40., date(2020,1,1), date(2025,1,1), 'duplicados'),
                (PADRAO, 'INVÁLIDO', 40., date(2020,1,1), date(2025,1,1), 'Código inválido'),
                (PADRAO, ' '.join('AAAA'+str(i) for i in range(8)), 40., date(2020,1,1), date(2025,1,1), '3 a 12'),
                (PADRAO, '', 10., date(2020,1,1), date(2025,1,1), 'inviáveis'),
                (PADRAO, '', 40., date(2025,1,1), date(2020,1,1), 'anterior'),
            ]:
                at.multiselect(key='ativos_carteiras').set_value(selecionados)
                at.text_input(key='outros_carteiras').input(extra)
                at.number_input(key='limite_carteiras').set_value(limite)
                at.date_input(key='inicio_carteiras').set_value(inicio)
                at.date_input(key='fim_carteiras').set_value(fim)
                at.button(key=BOTAO).click().run()
                self.assertFalse(at.exception)
                self.assertTrue(any(mensagem in e.value for e in at.error))
                self.assertNotIn('analise_carteiras', at.session_state)
            motor.assert_not_called()

    @unittest.skipUnless(os.environ.get('TESTAR_INTERFACE_REAL') == '1', 'Ative TESTAR_INTERFACE_REAL=1 para consultas reais')
    def test_motor_cache_capital_erros_e_navegacao(self):
        with patch('motor_carteiras.analisar_carteiras', wraps=motor_carteiras.analisar_carteiras) as motor, \
                patch('baixar_dados.baixar_precos_diarios', wraps=baixar_dados.baixar_precos_diarios) as regimes, \
                patch('dados_carteiras.carregar_carteiras', side_effect=AssertionError('CSV indevido')):
            at = AppTest.from_file(str(ARQUIVO), default_timeout=120).run()
            self.assertFalse(at.exception)
            self.assertEqual(regimes.call_count, 1)
            self.assertEqual(motor.call_count, 0)
            at.date_input(key='fim_carteiras').set_value(date(2026,11,13))
            at.button(key=BOTAO).click().run()
            self.assertFalse(at.exception)
            self.assertEqual(motor.call_count, 1)
            resultado = at.session_state['analise_carteiras']['resultado']
            self.assertEqual(list(resultado['pesos'].index), ['PETR4','VALE3','ITUB4','BBAS3','ABEV3'])
            tabela = at.tabs[4].get('dataframe')[-1].value
            for estrategia, nome in [('igualitaria', 'Igualitária'), ('minima_volatilidade', 'Mínima volatilidade')]:
                self.assertEqual(tabela.loc['Patrimônio final',nome], moeda(resultado['resumo'].loc[estrategia,'patrimonio_final']))
                for coluna, rotulo in [('retorno_acumulado_decimal', 'Retorno acumulado'),
                                       ('retorno_anualizado_realizado_decimal', 'Retorno anualizado'),
                                       ('volatilidade_anualizada_realizada_decimal', 'Volatilidade anualizada'),
                                       ('drawdown_maximo_decimal', 'Drawdown máximo')]:
                    esperado = f"{resultado['resumo'].loc[estrategia, coluna] * 100:.2f}%".replace('.', ',')
                    self.assertEqual(tabela.loc[rotulo, nome], esperado)
            percentuais = tabela.iloc[1:].copy()

            # O formulário só aplica o novo capital na submissão; motor/cache não dependem dele.
            at.number_input(key='capital_carteiras').set_value(200000.)
            at.button(key=BOTAO).click().run()
            self.assertEqual(motor.call_count, 1)
            self.assertEqual(regimes.call_count, 1)
            tabela = at.tabs[4].get('dataframe')[-1].value
            self.assertEqual(tabela.loc['Patrimônio final','Igualitária'], moeda(resultado['resumo'].loc['igualitaria','patrimonio_final'] * 2))
            pd.testing.assert_frame_equal(tabela.iloc[1:], percentuais)
            pd.testing.assert_frame_equal(resultado['pesos'], at.session_state['analise_carteiras']['resultado']['pesos'])
            at.button(key=BOTAO).click().run()
            self.assertEqual(motor.call_count, 1)

            # Três ativos, incluindo um código livre; período e limite diferentes.
            at.multiselect(key='ativos_carteiras').set_value(PADRAO[:2])
            at.text_input(key='outros_carteiras').input('b3sa3.sa')
            at.date_input(key='inicio_carteiras').set_value(date(2021,3,10))
            at.date_input(key='fim_carteiras').set_value(date(2024,12,31))
            at.number_input(key='limite_carteiras').set_value(60.)
            at.button(key=BOTAO).click().run()
            self.assertFalse(at.exception)
            self.assertEqual(motor.call_count, 2)
            resultado_tres = at.session_state['analise_carteiras']['resultado']
            self.assertEqual(list(resultado_tres['pesos'].index), ['PETR4','VALE3','B3SA3'])
            self.assertEqual(resultado_tres['periodo']['fim_teste'], '2024-12-27')

            # Seis ativos e novo período; a chave precisa mudar.
            at.multiselect(key='ativos_carteiras').set_value(PADRAO + ['BBDC4 — Bradesco'])
            at.text_input(key='outros_carteiras').input('')
            at.date_input(key='inicio_carteiras').set_value(date(2022,1,1))
            at.date_input(key='fim_carteiras').set_value(date(2025,12,18))
            at.number_input(key='limite_carteiras').set_value(25.)
            at.button(key=BOTAO).click().run()
            self.assertFalse(at.exception)
            self.assertEqual(motor.call_count, 3)
            self.assertEqual(len(at.session_state['analise_carteiras']['resultado']['pesos']), 6)

            # Nova tentativa inválida remove os gráficos e métricas da combinação anterior.
            at.text_input(key='outros_carteiras').input('PETR4')
            at.button(key=BOTAO).click().run()
            self.assertTrue(any('duplicados' in e.value for e in at.error))
            self.assertNotIn('analise_carteiras', at.session_state)
            self.assertFalse(at.tabs[4].get('metric'))
            self.assertFalse(at.tabs[4].get('plotly_chart'))

            # Voltar à combinação original recupera só o cache correspondente.
            at.multiselect(key='ativos_carteiras').set_value(PADRAO)
            at.text_input(key='outros_carteiras').input('')
            at.date_input(key='inicio_carteiras').set_value(date(2020,1,1))
            at.date_input(key='fim_carteiras').set_value(date(2026,11,13))
            at.number_input(key='limite_carteiras').set_value(40.)
            at.button(key=BOTAO).click().run()
            self.assertEqual(motor.call_count, 3)
            pd.testing.assert_frame_equal(resultado['pesos'], at.session_state['analise_carteiras']['resultado']['pesos'])

            # Erros da coleta ou do solver nunca deixam resultados antigos visíveis.
            for limite, mensagem in [(45., 'Falha de coleta'), (50., 'Falha no solver')]:
                motor.side_effect = RuntimeError(mensagem)
                at.number_input(key='limite_carteiras').set_value(limite)
                at.button(key=BOTAO).click().run()
                self.assertTrue(any(mensagem in e.value for e in at.error))
                self.assertNotIn('analise_carteiras', at.session_state)
                self.assertFalse(at.tabs[4].get('metric'))
            motor.side_effect = None

            # Regimes continuam calculando e exibindo K-Means/Markov após usar Carteiras.
            at.sidebar.selectbox[0].select('VALE3 — Vale')
            at.button(key='FormSubmitter:filtros-Analisar ativo').click().run()
            self.assertFalse(at.exception)
            self.assertEqual(regimes.call_count, 2)
            self.assertTrue(at.tabs[1].get('plotly_chart'))
            self.assertTrue(at.tabs[2].get('plotly_chart'))


if __name__ == '__main__':
    unittest.main()
