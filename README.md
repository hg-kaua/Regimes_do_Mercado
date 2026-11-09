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
python -m pip install -e .
```

## Execução

Parâmetros em `config/config.yaml` (a data final do período está congelada para reprodutibilidade).

### 1. Coleta e validação dos dados

```bash
python -m regimes.data collect   # baixa yfinance e arquivos COTAHIST da B3; grava snapshots brutos imutáveis + metadados (SHA-256)
python -m regimes.data build     # valida, confere yfinance x B3 e gera data/processed/bova11_daily.csv + quality_report.json
python -m regimes.data recheck   # (opcional) nova consulta ao yfinance comparada ao snapshot
```

`collect` só grava novos snapshots se ainda não existirem (nunca sobrescreve). Os arquivos anuais da B3 somam cerca de 0,5 GB.

A conferência entre as fontes identificou falhas pontuais no Yahoo Finance (candles sem negociação, pregões ausentes). Por isso os indicadores usam o OHLCV oficial da B3 (`features.source: b3`); o yfinance permanece como coleta preservada e conferida. Nenhuma linha é removida: inconsistências viram colunas `flag_*`.

## Testes

```bash
python -m pytest
```

Os testes não acessam a rede: usam dados sintéticos e uma pequena amostra real do layout COTAHIST.
