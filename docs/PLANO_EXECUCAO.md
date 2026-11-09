# Plano de execução e critérios de aceite

**Fluxo:** realizar uma etapa → revisar evidências e testes → escrever relatório da etapa → atualizar decisões e README → commit → avançar apenas com autorização.

| Etapa | Foco | Produto verificável | Documentação | Commit sugerido |
|---|---|---|---|---|
| 00 | Repositório, pergunta, escopo | Estrutura inicial e regras metodológicas | `docs/etapas/00_escopo.md` | `docs: define escopo e plano do projeto` |
| 01 | Coleta de dados reais | Função/script reprodutível, descrição e integridade | `01_dados.md` | `feat: implementa coleta e verificacao de dados` |
| 02 | EDA e indicadores | Tabela de atributos e gráficos exploratórios | `02_eda_indicadores.md` | `feat: cria indicadores financeiros e analise exploratoria` |
| 03 | Split e pré-processamento | Treino/validação/teste cronológicos; scaler treinado só no treino | `03_preprocessamento.md` | `feat: prepara pipeline temporal sem vazamento` |
| 04 | PCA | Componentes ajustados no treino, variância explicada | `04_pca.md` | `feat: adiciona experimentos de representacao com pca` |
| 05 | K-Means | Baseline e variante PCA, comparação de K e perfis dos clusters | `05_kmeans.md` | `feat: identifica regimes com kmeans` |
| 06 | Markov | Matrizes de transição e previsões para múltiplos horizontes | `06_markov.md` | `feat: modela transicoes entre regimes` |
| 07 | Avaliação | Métricas, baselines e análise de estabilidade | `07_avaliacao.md` | `test: valida modelos e compara baselines` |
| 08 | Dashboard | Interface interativa alimentada por resultados reais | `08_dashboard.md` | `feat: cria dashboard de regimes de mercado` |
| 09 | Consolidação | Reproduzir pipeline, relatório e roteiro de apresentação | `09_conclusao.md` | `docs: consolida metodologia e resultados do projeto` |

## Etapa 00 — Escopo e arquitetura

**Atividades:** especificar objetivo e hipótese; deixar AWS/LSTM de fora; definir stack, organização de módulos e condições de aprovação da professora.

**Aceite:** README e instruções coerentes; nenhuma dependência de nuvem e nenhum algoritmo já declarado como 'validado'.

## Etapa 01 — Dados e proveniência

**Atividades:** pesquisar/validar acesso a BOVA11.SA; coletar OHLCV diário com opção de preços ajustados, registrar fornecedor, licença/condições de uso, data/hora de coleta, intervalo efetivo, contagem, faltantes, duplicatas e alterações corporativas relevantes.

**Aceite:** reexecução da coleta funciona ou falha de forma clara; datas únicas e ordenadas; tabela de esquema; dados brutos preservados fora de `src`; não publicar segredos ou grandes arquivos inadvertidamente.

## Etapa 02 — Estatística e indicadores

**Atividades:** resumo descritivo; retorno diário e características calculadas sem dados futuros; volatilidade de 20 pregões; distância relativa da média móvel de 20; medida normalizada de volume; matriz de correlações, distribuições e tratamento de NaN/inícios de série.

**Aceite:** nomes, fórmulas, unidades e justificativas documentados; testes com série pequena; todos os indicadores de t dependem apenas de datas <=t.

## Etapa 03 — Split temporal e normalização

**Atividades:** definir recortes cronológicos (proposta 70%/15%/15%, após geração das características), aplicar scaler apenas no treino e transformar validação/teste.

**Aceite:** datas e quantidades exatas registradas; não existe `fit` em validação/teste; mesma ordem de colunas em todos os conjuntos; testes anti-vazamento.

## Etapa 04 — PCA

**Atividades:** ajustar PCA ao treino normalizado, examinar variância explicada e loadings, propor 90% ou 95% como alvo de variância (decidir por validação/justificativa), comparar dimensão reduzida com original.

**Aceite:** `fit` só no treino; gráfico de variância acumulada; interpretação das componentes com ressalva de que PCA maximiza variância, não previsibilidade.

## Etapa 05 — K-Means

**Atividades:** testar k=2..6, fixar semente e n_init, calcular inércia e silhouette em treino; inspecionar estabilidade e distribuição temporal; comparar original padronizado vs PCA; interpretar cada centroide em indicadores originais, **sem presumir alta/baixa/lateral**.

**Aceite:** protocolo de seleção não olha teste; não há cluster vazio ignorado; rótulos da validação/teste vêm de `predict`; modelos e versões são salvos/reproduzíveis.

## Etapa 06 — Cadeia de Markov

**Atividades:** converter a sequência cronológica de clusters em estados discretos; contar transições t→t+1 só no treino; normalizar linhas; estudar alternativas quando estados têm poucas transições; calcular P^h para 1, 5, 10, 20 pregões e prever distribuições condicionadas ao estado observado em t.

**Aceite:** cada linha de P soma 1 (com tratamento documentado de estados sem transições); teste unitário de matriz conhecida; interpretação em dias de pregão; sem utilizar estado do futuro na previsão.

## Etapa 07 — Avaliação fora da amostra

**Atividades:** comparar distribuições preditas com estados de validação/teste (fixando o modelo antes de abrir teste), aplicar Brier Score multiclasses ou log loss; comparar à persistência e às probabilidades marginais do treino; avaliar variação por período e possíveis violações de estabilidade temporal.

**Aceite:** definição explícita de alvo e horizonte, denominador e avaliação por horizonte; baselines calculados no mesmo protocolo; sem promessas financeiras; resultados negativos também apresentados.

## Etapa 08 — Dashboard

**Atividades:** implementar app Streamlit com seleção de intervalos, gráfico preço/regime, perfil dos clusters, matriz de transição, evolução de probabilidades e quadro de avaliação.

**Aceite:** nenhum resultado fictício apresentado como real; gráficos possuem datas/legendas; interação não recalibra modelagem com dados futuros; erros e dependências documentados.

## Etapa 09 — Documento acadêmico e entrega

**Atividades:** consolidar introdução, objetivo, referências aos slides, metodologia, ambiente, experimentos, resultados, limitações, conclusões e trabalhos futuros (eventual LSTM no TCC; AWS a validar). Preparar demonstração e roteiro oral.

**Aceite:** terceiro consegue reproduzir os resultados com instruções; figuras com fontes; links válidos; relatório distingue hipótese, método e evidência; versão/tag final criada com autorização.

## Validação científica fundamental

- Trata-se de **modelagem de regimes**, não previsão direta de preços ou de rentabilidade.
- O estado do dia t é calculado com o dado disponível até o fechamento de t; a previsão concerne ao estado do dia t+h.
- PCA e K-Means têm papéis diferentes: PCA modifica a representação e K-Means constrói agrupamentos.
- Clusters não equivalem automaticamente a 'mercado de alta', 'mercado de baixa' ou 'lateralização'.
- A hipótese de Markov de primeira ordem, e especialmente a estabilidade da matriz de transição, é aproximação que deve ser discutida criticamente.
