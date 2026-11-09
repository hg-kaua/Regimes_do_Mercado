# Regimes do Mercado

Identificação de regimes de comportamento em séries financeiras com aprendizado não supervisionado (K-Means, com e sem PCA) e modelagem das transições entre regimes por cadeia de Markov de primeira ordem.

Projeto acadêmico da disciplina **Tópicos Especiais em Inteligência Artificial**. Não é sistema de negociação, recomendação de investimento nem previsão de preços.

## Objetivo

Para cada pregão `t`, calcular um vetor de indicadores usando apenas informações disponíveis até o fechamento de `t`, agrupar esses vetores em regimes e estimar a distribuição de probabilidade do regime em `t + h` (h = 1, 5, 10 e 20 pregões), avaliando-a fora da amostra contra referências simples.

## Dados

- Ativo: **BOVA11.SA** (ETF do Ibovespa negociado na B3), frequência diária, de 2018-01-01 até o último pregão completo na data da coleta.
- Fonte principal: Yahoo Finance via `yfinance` (API não oficial).
- Fonte de conferência: Séries Históricas (COTAHIST) da B3.
- Os dados não são versionados: são baixados e reconstruídos localmente pelos scripts.

## Tecnologias

Python, pandas, numpy, scikit-learn, matplotlib, yfinance, PyYAML e pytest.

## Estrutura

```
config/config.yaml   parâmetros do experimento
src/regimes/         código-fonte do pacote
tests/               testes automatizados (sem acesso à rede)
```

## Instalação

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Execução

Instruções adicionadas conforme os módulos forem implementados.
