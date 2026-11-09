# Registro de decisões metodológicas

| ID | Data | Decisão | Justificativa | Estado |
|---|---|---|---|---|
| D-001 | 2026-11-09 | Foco em K-Means, PCA e cadeia de Markov | Relação com disciplina e estudo futuro sobre LSTM | Definida |
| D-002 | 2026-11-09 | AWS/SageMaker excluído temporariamente | Precisa validar exigência com a professora e custo | Pendente de revisão |
| D-003 | 2026-11-09 | Sem LSTM e sem automação de trading | Evitar ampliar escopo do trabalho individual | Definida |
| D-004 | 2026-11-09 | BOVA11.SA como candidato inicial | Proxy amplo do mercado brasileiro; sujeito à disponibilidade e qualidade dos dados | Proposta |
| D-005 | 2026-11-09 | Dados diários, horizonte de 1/5/10/20 pregões | Conectado a transições de Markov | Proposta |
| D-006 | 2026-11-09 | Divisão cronológica e `fit` só em treino | Evitar vazamento temporal | Definida |
| D-007 | 2026-11-09 | Comparar K-Means com/sem PCA | Identificar efeito da representação nos regimes | Definida |
| D-008 | 2026-11-09 | Problema formal: estado `s_t` = K-Means(treino) sobre indicadores observáveis em t; prever distribuição de `s_{t+h}` por `e_{s_t}·P̂^h` | Torna explícitos alvo, horizonte e o que não é previsto (preço/retorno) | Definida |
| D-009 | 2026-11-09 | Métrica principal: Brier multiclasse por horizonte; log loss secundário; comparação entre configurações por Brier Skill Score vs marginal e persistência | Brier bruto não é comparável entre K/partições diferentes; persistência tem log loss infinito | Proposta |
| D-010 | 2026-11-09 | Comparar `P̂^h` com matriz direta de h passos `Q̂_h` estimada no treino | Testa empiricamente a hipótese de 1ª ordem | Proposta |
| D-011 | 2026-11-09 | Suavização aditiva (α) nas linhas de `P̂`, α escolhido na validação; contagens brutas reportadas | Estados raros geram linhas instáveis | Proposta |
| D-012 | 2026-11-09 | Comparação com/sem PCA só com descarte de componentes; incluir variante com 2 componentes; avaliar ampliar indicadores na Etapa 02 | PCA sem whitening é rotação: com todas as componentes, K-Means gera a mesma partição | Definida (ampliação: proposta) |
| D-013 | 2026-11-09 | Origens de avaliação e transições válidas só com t e t+h no mesmo conjunto | Evitar vazamento de alvo entre conjuntos | Definida |
| D-014 | 2026-11-09 | Arquitetura `src/regimes/` (data, features, split, preprocessing, clustering, markov, evaluation, plots, pipeline), `config.yaml` único, dashboard somente leitura | Modularidade, testabilidade, reprodutibilidade | Proposta |
| D-015 | 2026-11-09 | Seleção de K por silhouette/inércia no treino + tamanho mínimo + duração média + estabilidade (ARI) + interpretabilidade; teste não participa | Silhouette sozinho ignora dinâmica temporal e estabilidade | Proposta |
| D-016 | 2026-11-09 | Pré-registrar H1–H3 (ver `docs/etapas/00_escopo.md`, seção 9.2) | Evitar ajuste de hipóteses após ver resultados | Definida |

## Adicionar nova decisão

- ID:
- Data:
- Problema:
- Alternativas consideradas:
- Escolha:
- Justificativa:
- Consequências e riscos:
- Reversível?
