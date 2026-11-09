# Regimes do Mercado Financeiro — K-Means, PCA e Cadeias de Markov

Projeto acadêmico individual da disciplina **Tópicos Especiais em Inteligência Artificial**.

> **Status:** Etapa 00 (escopo e arquitetura) concluída e aguardando aprovação; ainda sem código e sem dados. AWS fora do escopo até nova decisão.

## Pergunta de pesquisa

É possível identificar regimes de comportamento em séries financeiras com aprendizado não supervisionado e estimar probabilidades de transição entre eles por uma cadeia de Markov de primeira ordem?

## Objetivo

Estudar características estatísticas de séries financeiras, comparar K-Means com e sem PCA, modelar transições dos grupos e avaliar as previsões probabilísticas em dados posteriores ao treinamento. Não é um sistema de negociação nem de previsão de preços.

## Recorte inicial (proposta, sujeito à validação de dados)

- Ativo: **BOVA11.SA** (ETF negociado na B3).
- Frequência: pregões diários; início tentativo: **2018-01-01** até último pregão efetivamente disponível na coleta.
- Indicadores candidatos: retorno diário, volatilidade móvel (20 pregões), distância do fechamento em relação à média móvel de 20 pregões e atividade relativa de volume.
- Regimes: aprendidos pelo K-Means. Não assumir a priori 'alta', 'lateralização' e 'queda'.
- Horizontes de Markov: **1, 5, 10 e 20 pregões**. Interpretar como distribuição probabilística de estados, não como previsão certeira do preço.
- Análise: treino/validação/teste separados cronologicamente.

## Tecnologias

Python, pandas, numpy, scikit-learn, matplotlib, seaborn (opcional), Streamlit e pytest. Dados com uma fonte pública a ser documentada. Git/GitHub para acompanhamento. AWS está explicitamente adiada.

## Regras científicas

1. Nenhuma variável pode usar dados posteriores ao momento da previsão.
2. `StandardScaler`, PCA e K-Means são ajustados **somente com o treino**; validação/teste usam `transform` e `predict` dos objetos já ajustados.
3. Os parâmetros da matriz de Markov são estimados somente com as transições do conjunto de treino; não usar dados da avaliação para formar a matriz.
4. Teste é intocado até as decisões metodológicas estarem definidas por treino e validação.
5. Relatar a fonte, data de coleta, valores ausentes, intervalos temporais, tamanho das amostras e as limitações de séries financeiras não estacionárias.
6. Não presumir que clusters representam estados econômicos objetivos; seus rótulos são interpretados **após** analisar os centroides.
7. Comparar previsões de Markov a pelo menos duas referências: persistência do regime atual e distribuição marginal de estados estimada no treino.
8. Não anunciar resultados, gráficos, métricas ou conclusões antes de executar o experimento.

## Organização e andamento

- [Plano de execução e entregas](docs/PLANO_EXECUCAO.md)
- [Instruções para configurar um Projeto no Claude](docs/CONFIGURAR_CLAUDE.md)
- [Instruções persistentes para Claude Code](CLAUDE.md)
- [Prompts de cada etapa](prompts/ETAPAS_CLAUDE.md)
- [Decisões metodológicas](docs/DECISOES.md)
- [Registro dos experimentos](docs/EXPERIMENTOS.md)
- [Modelo de registro da etapa](docs/MODELO_RELATORIO_ETAPA.md)

## Etapas

- [x] 00 — Escopo e repositório ([registro](docs/etapas/00_escopo.md))
- [ ] 01 — Dados e proveniência
- [ ] 02 — Análise exploratória e engenharia dos indicadores
- [ ] 03 — Divisão temporal e pré-processamento
- [ ] 04 — PCA e análise da representação
- [ ] 05 — K-Means e interpretação dos regimes
- [ ] 06 — Cadeias de Markov
- [ ] 07 — Validação e análise de sensibilidade
- [ ] 08 — Dashboard
- [ ] 09 — Relatório final, apresentação e reprodutibilidade

## Como executar

Os comandos de instalação e execução serão adicionados apenas após implementação e validação dos módulos correspondentes. Não há resultados experimentais nesta versão do repositório.
