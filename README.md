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

### 2. Indicadores e análise estatística

```bash
python -m regimes.eda
```

Calcula, por pregão `t` e usando apenas dados até `t`:

| Indicador | Definição |
|---|---|
| `log_return` | `ln(C_t / C_{t-1})` |
| `volatility_20` | desvio-padrão amostral dos últimos 20 log-retornos |
| `dist_sma_20` | `C_t / MM20_t - 1` (média móvel simples de 20 pregões, inclui `t`) |
| `rel_volume_20` | `ln(V_t / média de V nos 20 pregões anteriores)` |

Gera `data/processed/features.csv`, tabelas em `reports/etapa02/` (estatísticas descritivas, covariância, correlações, extremos) e figuras em `reports/figures/etapa02/`.

### 3. Divisão temporal e padronização

```bash
python -m regimes.preprocessing
```

- Remove explicitamente as linhas iniciais sem indicadores definidos (aquecimento das janelas de 20 pregões); valores ausentes após esse trecho interrompem a execução.
- Divide em blocos cronológicos de 70 % (treino), 15 % (validação) e 15 % (teste), sem embaralhar.
- Ajusta o `StandardScaler` **somente no treino** e aplica a mesma transformação à validação e ao teste.
- Grava `data/processed/dataset_etapa03.csv` (datas, conjunto, indicadores originais e padronizados), o scaler em `artifacts/etapa03/scaler.joblib` e verificações em `reports/etapa03/`.

### Ordem completa

```bash
python -m regimes.data collect && python -m regimes.data build && python -m regimes.eda && python -m regimes.preprocessing
```

## Testes

```bash
python -m pytest
```

Os testes não acessam a rede: usam dados sintéticos e uma pequena amostra real do layout COTAHIST.
