# Regimes do Mercado

Laboratório quantitativo para ações da B3, em Streamlit. A aplicação tem duas partes:

- **Regimes de mercado:** agrupa períodos de um ativo por retorno e amplitude (K-Means)
  e estima as transições entre esses regimes (Cadeias de Markov).
- **Carteiras:** monta e compara carteiras de 3 a 12 ações (igualitária, mínima
  volatilidade, máximo Sharpe e personalizada). Inclui fronteira eficiente, backtest
  fora da amostra, benchmarks Ibovespa e CDI e validação walk-forward.

Projeto da disciplina Tópicos Especiais em Inteligência Artificial. Os resultados são
simulações históricas: não garantem retornos futuros e não são recomendação de investimento.

## Instalação

Python 3.12 ou superior (testado em 3.14). No PowerShell, na pasta do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

As versões em `requirements.txt` são as testadas e estão fixadas para reproduzir
resultados e testes. O otimizador exige o solver **CLARABEL** (instalado com `clarabel`);
nenhum solver alternativo é usado.

## Execução

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

O painel baixa os dados sozinho; a internet é necessária para Yahoo Finance e Banco Central.
Os scripts de linha de comando reproduzem os estudos de referência e gravam CSVs em
`dados/`, uma pasta ignorada pelo Git:

| Fluxo | Comandos, em ordem |
| --- | --- |
| Regimes (PETR4 semanal) | `baixar_dados.py` → `analisar_dados.py` → `calcular_markov.py` |
| Carteiras (5 ações, 80/20) | `preparar_carteira.py` → `analisar_carteira.py` → `otimizar_carteira.py` → `backtest_carteira.py` |

Testes automatizados (`unittest`):

```powershell
.\.venv\Scripts\python.exe -m unittest test_motor_carteiras test_maximo_sharpe test_fronteira_eficiente test_personalizada test_benchmarks test_walk_forward test_app_carteiras
.\.venv\Scripts\python.exe analisar_carteira.py --testar   # idem otimizar_carteira.py e backtest_carteira.py
```

Os testes padrão usam dados sintéticos e não acessam a rede. Para incluir consultas reais,
defina `TESTAR_DADOS_REAIS=1` e `TESTAR_INTERFACE_REAL=1`.

## Funcionalidades

**Barra lateral — regimes do ativo** (abas Visão geral, K-Means, Cadeias de Markov e
Dados e normalização):

- Ação da lista ou código digitado, período e frequência (semanal, mensal ou anual).
- Fechamento ajustado, grupos K-Means com centroides e matriz de transição do treinamento.
- Comparação treino × teste das frequências de transição.
- Dados normalizados, classificação de cada período e exportação em CSV.

**Aba Carteiras** (formulário próprio, independente da barra lateral):

1. **Configuração:** ativos (3 a 12), datas, limite máximo por ativo, capital e taxa livre
   de risco (uma hipótese informada pelo usuário).
2. **Carteira personalizada:** pesos manuais de 0% a 100%, que precisam somar 100%; nada é normalizado.
3. **Composição** das estratégias e **estimativas do treinamento** (retorno, volatilidade, Sharpe).
4. **Fronteira eficiente** com escolha de retorno-alvo e os pesos correspondentes.
5. **Desempenho no teste:** patrimônio final, retorno acumulado e anualizado,
   volatilidade e drawdown máximo, com seleção de benchmarks (Ibovespa e CDI).
6. **Evolução patrimonial** das estratégias e dos benchmarks.
7. **Validação walk-forward:** treino móvel e janelas de avaliação configuráveis, com
   tabela por janela, rebalanceamentos, consistência e evolução dos pesos-alvo.

Mudar apenas o capital reaproveita a análise: os valores em R$ são reescalados e os
percentuais não mudam.

## Fontes de dados

| Dado | Fonte | Unidade e tratamento |
| --- | --- | --- |
| Ações | Yahoo Finance via `yfinance`, `auto_adjust=True` | Preço ajustado por proventos e desdobramentos. Agregação semanal `W-FRI` pelo último pregão da semana. |
| Ibovespa | Yahoo Finance, `^BVSP` | Pontos do índice. Pela metodologia da B3, é um índice de **retorno total**. |
| CDI | Banco Central do Brasil, SGS **série 12** | **% ao dia útil**. Consultado em janelas de até 10 anos, o limite da API. Não é a Selic (série 11). |

Uma semana só entra depois de encerrada; o corte é sexta-feira às 19h de São Paulo.
Semanas sem dados não são preenchidas: o alinhamento usa apenas datas comuns a todos os ativos.

## Métodos financeiros

- **Retornos:** semanais simples, `P_t / P_{t-1} − 1`. Nos CSVs aparecem em porcentagem;
  nos cálculos, são convertidos uma única vez para decimal.
- **Divisão treino/teste:** cronológica, 80/20. Toda estimativa (médias, covariância,
  pesos, Min-Max, K-Means, matriz de Markov) usa somente o treinamento.
- **Estimativas anuais:** média aritmética semanal × 52, covariância semanal × 52 e
  volatilidade × √52, com estatísticas amostrais (ddof = 1).
- **Estratégias:**
  - Igualitária: 1/n por ativo.
  - Mínima volatilidade e máximo Sharpe: Markowitz com PyPortfolioOpt (CLARABEL), pesos
    entre 0 e o limite informado e soma igual a 1. Sem `clean_weights` e sem renormalização.
  - Personalizada: pesos informados pelo usuário.
- **Fronteira eficiente:** mínima variância para cada retorno-alvo resolvido. Alvos sem
  solução aparecem como lacunas, sem interpolação.
- **Backtest 80/20:** buy and hold a partir do fechamento da última semana de treino. Os
  pesos ficam congelados na formação e derivam com os preços.
- **Indicadores realizados:**
  - Retorno anualizado composto: `(final/inicial)^(52/semanas) − 1`.
  - Volatilidade: desvio amostral dos retornos semanais da curva × √52.
  - Drawdown: queda em relação ao máximo anterior.
- **Benchmarks:** curvas com o mesmo capital e as mesmas datas das carteiras.
  - Ibovespa: `P_t / P_formação`.
  - CDI: `Π(1 + taxa_d/100)` sobre os dias úteis d com formação ≤ d < t.
- **Walk-forward:**
  - Janelas móveis, por padrão 104 semanas de treino e 13 de avaliação, sem sobreposição
    e ancoradas na semana mais recente.
  - A cada janela, os pesos são reestimados só com dados anteriores e a carteira volta
    aos pesos-alvo; dentro da janela, é buy and hold.
  - Se uma estimativa falha, as posições são mantidas; se falha no primeiro ciclo, a estratégia fica indisponível.
- **Regimes:** Min-Max e K-Means de 3 grupos (`random_state=42`, `n_init=10`) ajustados no
  treino. Matriz de Markov feita só com pares consecutivos do treino.

## Premissas dos backtests

- Sem corretagem, impostos, spread ou slippage. No walk-forward, o parâmetro
  `custo_transacao` existe, mas vale 0 na interface.
- Posições fracionárias e nenhum aporte ou retirada.
- Preços ajustados: os proventos são tratados como reinvestidos no próprio ativo.
- Rebalanceamento: nenhum no backtest 80/20; no walk-forward, apenas no início de cada janela.
- A taxa livre de risco é uma hipótese fixa do usuário, aplicada a todo o período e a
  todas as janelas do walk-forward.

## Limitações

- **Viés de sobrevivência:** os ativos são escolhidos hoje, com conhecimento de que ainda existem.
- **Revisão retroativa de preços:** o Yahoo recalcula os preços ajustados a cada novo
  provento, então a mesma consulta feita em outro dia pode gerar pesos ligeiramente
  diferentes. Pela mesma razão, uma data final igual a hoje muda depois das 19h.
- **Ibovespa:** é um índice teórico, rebalanceado pela B3 e ponderado por free float, e
  não é investível sem custos. O tratamento de proventos difere do ajuste do Yahoo nas ações.
- **Taxa livre de risco fixa:** em janelas antigas pode não refletir o juro da época e
  afetar a viabilidade do Máximo Sharpe.
- **Amostras curtas:** poucas janelas no walk-forward (por exemplo, 19) e poucas
  transições por grupo no Markov tornam as frequências instáveis.
- **Anualização:** o fator √52 desconsidera autocorrelação. A média anualizada é
  aritmética, não um retorno realizado.
- **Sem prova de causalidade ou previsão:** os resultados descrevem o histórico observado.

## Estrutura

| Arquivo | Papel |
| --- | --- |
| `app.py` | Painel Streamlit (apresentação; reutiliza as funções abaixo) |
| `baixar_dados.py`, `analisar_dados.py`, `calcular_markov.py` | Regimes: coleta, K-Means e Markov |
| `preparar_carteira.py`, `analisar_carteira.py` | Preços semanais alinhados, retornos e estatísticas de treino |
| `otimizar_carteira.py`, `maximo_sharpe.py`, `fronteira_eficiente.py` | Otimização de Markowitz |
| `backtest_carteira.py`, `dados_carteiras.py` | Backtest buy and hold, indicadores e leitura dos CSVs |
| `motor_carteiras.py` | Pipeline em memória usado pela aba Carteiras |
| `benchmarks.py` | Ibovespa e CDI |
| `walk_forward.py` | Validação walk-forward |
| `test_*.py` | Testes automatizados |
