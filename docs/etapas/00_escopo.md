# Etapa 00 — Escopo e revisão da arquitetura

- **Data:** 2026-11-09
- **Ambiente:** revisão documental; nenhum código executado e nenhum dado coletado.
- **Status:** concluída, aguardando aprovação do responsável antes da Etapa 01.

## 1. Objetivo

Fixar o escopo do projeto, formalizar o problema de previsão, revisar se a combinação PCA → K-Means → cadeia de Markov é adequada para identificar regimes e prever probabilisticamente o regime futuro, propor a organização dos módulos Python, as dependências, os riscos metodológicos e os critérios de validação de cada etapa. Nenhum algoritmo é implementado nesta etapa.

## 2. Fontes consultadas

- `README.md`, `docs/PLANO_EXECUCAO.md`, `docs/DECISOES.md`.
- Instruções permanentes do Projeto no Claude. O arquivo `CLAUDE.md` **não estava na pasta do repositório**; as instruções do Projeto foram usadas como equivalentes (ver pendências).
- Slides da disciplina (Projeto "Regime financeiro - IA"):
  - TopIA_01 — avaliação do projeto: monografia + apresentação oral + implementação; tema precisa ser aprovado antes; repositório GitHub enviado à professora; relatório técnico segue modelo da professora.
  - TopIA_02 — cadeias de Markov e matriz de transição (linhas = estado atual, colunas = estado futuro). O projeto adota a mesma convenção.
  - TopIA_04 — PCA (combinações lineares, rotação do sistema de coordenadas, retenção de variância).
  - TopIA_05 — K-Means (passos do algoritmo e comparação de agrupamentos).

## 3. Escopo

| Dentro do escopo | Fora do escopo (até autorização explícita) |
|---|---|
| Um ativo (BOVA11.SA, proposta), dados diários | LSTM e qualquer rede neural (fica para o TCC) |
| Indicadores causais (dependem só de datas ≤ t) | Previsão de preço ou de retorno |
| K-Means com e sem PCA | Estratégia, sinal ou automação de trading |
| Cadeia de Markov de 1ª ordem, horizontes 1/5/10/20 pregões | AWS/SageMaker ou qualquer execução em nuvem (D-002) |
| Avaliação fora da amostra contra baselines | Múltiplos ativos, dados intradiários |
| Dashboard Streamlit local, somente leitura de artefatos | Recomendação de compra/venda |

## 4. Formulação do problema

Notação para todas as etapas seguintes:

1. **Observação.** Para cada pregão `t`, um vetor de indicadores `x_t` calculado apenas com dados disponíveis no fechamento de `t`.
2. **Representação.** `z_t = scaler(x_t)` (variante base) ou `z_t = PCA(scaler(x_t))` (variante PCA). `scaler` e `PCA` ajustados só no treino.
3. **Estado.** `s_t = KMeans.predict(z_t) ∈ {0, …, K−1}`, com o K-Means ajustado só no treino.
4. **Modelo de transição.** Matriz `P̂` (K×K) estimada pelos pares consecutivos `(s_t, s_{t+1})` com ambos os pregões no treino; cada linha soma 1.
5. **Previsão.** Para horizonte `h`: `p̂(s_{t+h} | s_t) = e_{s_t} · P̂^h` (linha `s_t` de `P̂^h`).
6. **Alvo.** O estado observado `s_{t+h}`. Uma origem `t` só é avaliada se `t` e `t+h` pertencem ao **mesmo** conjunto (validação ou teste). Origens cujo alvo cairia no conjunto seguinte são descartadas.
7. **O que não é previsto:** preço, retorno ou direção do mercado. O alvo é a distribuição sobre estados definidos pelo próprio modelo.

## 5. Análise da adequação da metodologia

### 5.1 Avaliação geral

A cadeia proposta é adequada para o objetivo **acadêmico** declarado — discretizar o comportamento do mercado em estados interpretáveis e estudar a dinâmica entre eles — e conversa diretamente com o conteúdo dos slides (PCA, K-Means, Markov). Ela tem, porém, três limitações estruturais que precisam aparecer no relatório e orientam os ajustes abaixo:

- **K-Means não é um modelo de regimes temporais.** Cada pregão é classificado isoladamente; não há termo que favoreça permanência no estado. A persistência observada virá apenas da suavidade dos indicadores (janelas de 20 pregões). Modelos que estimam estados e transições conjuntamente (HMM gaussiano, Markov-switching) são a referência natural da literatura e devem ser citados como alternativa/trabalho futuro, sem entrar no escopo.
- **A previsibilidade é parcialmente mecânica.** Com janelas de 20 pregões, o estado em `t+5` compartilha 15 dos 20 pregões usados no estado em `t`. Parte da "acurácia" em horizontes curtos vem da sobreposição, não de dinâmica do mercado. Por isso a comparação com a **persistência** é obrigatória e é a referência mais exigente em `h` curto.
- **Markov de 1ª ordem homogênea converge para a distribuição estacionária.** Para `h` grande, `P̂^h` tende a `π̂` e a previsão se aproxima da marginal do treino. Perda de vantagem em `h = 20` é resultado esperado e deve ser relatado como tal, não como falha de implementação.

### 5.2 K-Means

Adequado como discretizador, desde que:

- os dados sejam padronizados (distância euclidiana é sensível à escala);
- se reconheça a suposição implícita de grupos aproximadamente esféricos e de variância similar — indicadores financeiros têm caudas pesadas e episódios extremos (ex.: 2020) podem formar um cluster pequeno ou "puxar" centroides;
- a escolha de K não use só silhouette/inércia. Critério proposto (D-015): silhouette e inércia no treino **e** tamanho mínimo de cluster **e** duração média dos regimes **e** estabilidade entre sementes (ARI) **e** interpretabilidade dos centroides em unidades originais. O teste não participa;
- comparações entre ajustes alinhem rótulos por correspondência de centroides (algoritmo húngaro, `scipy.optimize.linear_sum_assignment`), nunca pela numeração.

### 5.3 PCA — ajuste importante

PCA sem *whitening* é uma **rotação ortogonal** do espaço padronizado: distâncias euclidianas são preservadas. Consequência: se o número de componentes retidos for igual ao número de indicadores, o K-Means "com PCA" produz a mesma partição que o K-Means "sem PCA" (a menos de inicialização). Com apenas 4 indicadores, a regra de 90–95% de variância pode reter 3 ou 4 componentes e a comparação ficaria vazia.

Ajustes propostos (D-012):

- registrar sempre quantas componentes foram retidas e se houve descarte; só há comparação informativa quando há descarte;
- incluir, além da regra de variância, uma variante com 2 componentes (útil também para visualização);
- avaliar na Etapa 02 um conjunto um pouco maior de indicadores correlacionados (ex.: retorno acumulado de 5 e 20 pregões, volatilidade de 5 e 20, razão entre volatilidades, distância às médias de 20 e 60), para que a redução de dimensão tenha papel real. Decisão final fica para a Etapa 02;
- lembrar no relatório que PCA maximiza variância, não poder discriminativo nem previsibilidade, e que outliers do treino podem dominar as primeiras componentes (sensibilidade com `RobustScaler` ou winsorização ajustada no treino).

### 5.4 Cadeia de Markov

Adequada e alinhada aos slides. Ajustes propostos:

- **Suavização das linhas** (D-011): contagens com pseudocontagem aditiva `α` (Laplace/Dirichlet) para estados raros; `α` escolhido na validação; contagens brutas por linha sempre reportadas. Estados sem nenhuma transição de saída recebem tratamento documentado (linha uniforme ou marginal), nunca descartados silenciosamente.
- **Matriz direta de h passos** (D-010): além de `P̂^h` (Chapman–Kolmogorov), estimar `Q̂_h` contando pares `(s_t, s_{t+h})` no treino. Se `Q̂_h` superar `P̂^h` fora da amostra, há evidência contra a hipótese de 1ª ordem — achado interpretável, que responde parte da pergunta de pesquisa.
- **Homogeneidade temporal:** comparar descritivamente a matriz do treino com a matriz que seria observada na validação (só como diagnóstico, nunca para ajustar o modelo) e discutir a mudança.
- **Distribuição estacionária** `π̂` (autovetor à esquerda) e **duração esperada** de cada regime `1/(1 − P̂_kk)` como resultados interpretáveis.

### 5.5 Indicadores (proposta para a Etapa 02)

| Indicador | Fórmula proposta | Observação |
|---|---|---|
| Log-retorno | `r_t = ln(C_t / C_{t−1})` | Muito ruidoso como atributo de cluster; tende a fazer o estado "piscar" dia a dia. Avaliar retorno acumulado em janela como alternativa. |
| Volatilidade 20 | desvio-padrão de `r` em `t−19…t` | Unidade: log-retorno diário (anualização opcional, documentada). |
| Distância da MM20 | `C_t / MM20_t − 1` | MM20 inclui `t`; é observável no fechamento de `t`. |
| Volume relativo | `ln(V_t / mediana(V_{t−20…t−1}))` | Volume de ETF tem crescimento estrutural; nível absoluto não é comparável entre anos. |

`C` = fechamento ajustado (definir na Etapa 01 se há eventos corporativos relevantes no período). Todo indicador precisa de teste mostrando que alterar dados posteriores a `t` não altera o valor em `t`.

### 5.6 Avaliação (proposta para a Etapa 07)

- **Métrica principal:** Brier multiclasse `BS = (1/N) Σ_t Σ_k (p̂_{t,k} − y_{t,k})²`, por horizonte (D-009).
- **Secundária:** log loss com probabilidades limitadas a `[ε, 1]`, documentado.
- **Baselines:** (a) marginal `π̂` das frequências de estado no treino; (b) persistência `e_{s_t}` (one-hot). A persistência tem log loss infinito quando erra, por isso é comparada pelo Brier, ou por uma versão suavizada `(1−ε)` no estado atual com `ε` escolhido na validação.
- **Comparação entre configurações** (K diferentes, com/sem PCA) **pelo Brier Skill Score** `BSS = 1 − BS_modelo / BS_referência`, não pelo Brier bruto: com outro K ou outra partição o problema de classificação muda e os valores absolutos não são comparáveis.
- **Incerteza:** origens sobrepostas tornam os erros autocorrelacionados; intervalos por *block bootstrap* com bloco ≥ h.
- **Diagnósticos adicionais:** distribuição de estados por conjunto, calibração (diagrama de confiabilidade) e desempenho por subperíodo.

### 5.7 Divisão temporal

70/15/15 cronológico após o cálculo dos indicadores e remoção do aquecimento das janelas. Estimativa de calendário (não é resultado): ~250 pregões/ano, então cada conjunto de 15% corresponde a pouco mais de 1 ano a partir de 2018. Riscos: o choque de 2020 fica no treino e validação/teste podem ter distribuição de estados bem diferente. Alternativa a decidir na Etapa 01 após checar a qualidade dos dados: iniciar a série mais cedo (BOVA11 existe desde o fim de 2008) para que o treino cubra mais tipos de regime. Avaliação com origem móvel (*rolling origin*), sempre reajustando só com o passado, fica como análise de sensibilidade opcional na Etapa 07.

## 6. Riscos metodológicos

| # | Risco | Efeito | Mitigação | Etapa |
|---|---|---|---|---|
| R1 | Vazamento temporal em indicadores (janelas centradas, `shift` errado) | Desempenho artificial | Funções puras + teste de invariância a dados futuros | 02 |
| R2 | `fit` de scaler/PCA/K-Means fora do treino | Vazamento de distribuição | Objetos ajustados uma vez e salvos; teste que falha se houver `fit` fora do treino | 03–05 |
| R3 | Transições cruzando a fronteira treino/validação ou alvo `t+h` em outro conjunto | Vazamento de alvo | Máscaras de origem válida por horizonte; teste unitário | 06–07 |
| R4 | PCA sem descarte de componentes | Comparação com/sem PCA vazia | D-012 | 04–05 |
| R5 | Estado "pisca" por retorno diário ruidoso | Matriz quase sem persistência, regimes não interpretáveis | Medir duração média; avaliar atributos suavizados | 02, 05 |
| R6 | Previsibilidade mecânica por janelas sobrepostas | Superestimar a "dinâmica" | Baseline de persistência; discussão explícita | 07 |
| R7 | Clusters raros (crises) com poucas transições | Linhas instáveis em `P̂` | Tamanho mínimo, suavização `α`, contagens reportadas | 05–06 |
| R8 | Não estacionariedade / mudança estrutural | Matriz do treino não vale no teste | Diagnóstico de homogeneidade, análise por subperíodo, limitação declarada | 07 |
| R9 | Permutação de rótulos entre ajustes | Comparações erradas | Alinhamento húngaro por centroides | 05 |
| R10 | Comparar Brier bruto entre K diferentes | Conclusão inválida | Usar BSS contra baselines no mesmo protocolo | 07 |
| R11 | Seleção de K/α/PCA olhando o teste | Otimismo | Teste aberto uma única vez, após congelar configuração | 07 |
| R12 | Fonte de dados não oficial ou alterada após coleta | Irreprodutibilidade | Snapshot bruto + metadados (data/hora, fonte, hash) | 01 |
| R13 | Volume com tendência estrutural | Cluster separado por época, não por regime | Volume relativo normalizado localmente | 02 |
| R14 | Interpretar clusters como "alta/queda/lateral" a priori | Viés de confirmação | Rótulos só após análise dos centroides | 05 |

## 7. Arquitetura proposta

```
Trabalho final/
├── README.md
├── CLAUDE.md
├── requirements.txt          # versões fixadas (Etapa 01)
├── config/
│   └── config.yaml           # ativo, datas, janelas, split, K, horizontes, sementes
├── data/
│   ├── raw/                  # snapshot bruto + metadata.json (nunca editado)
│   └── processed/            # indicadores e splits gerados
├── src/regimes/
│   ├── __init__.py
│   ├── config.py             # leitura/validação da configuração
│   ├── data.py               # coleta, verificação de integridade, snapshot + metadados
│   ├── features.py           # indicadores causais (funções puras)
│   ├── split.py              # divisão cronológica; máscaras de origem válida por horizonte
│   ├── preprocessing.py      # scaler e PCA ajustados só no treino
│   ├── clustering.py         # K-Means, métricas de seleção, alinhamento de rótulos, perfis
│   ├── markov.py             # contagens, normalização, suavização, P^h, Q_h, estacionária
│   ├── evaluation.py         # Brier, log loss, BSS, baselines, block bootstrap
│   ├── plots.py              # figuras do relatório
│   └── pipeline.py           # orquestra o experimento e salva artefatos
├── app/
│   └── streamlit_app.py      # lê artefatos; nunca ajusta modelos
├── tests/                    # pytest: features, split, markov, evaluation, anti-vazamento
├── notebooks/                # exploração apenas
├── artifacts/                # modelos (joblib), resultados (json/csv) com metadados
├── reports/figures/
└── docs/
    ├── PLANO_EXECUCAO.md
    ├── DECISOES.md
    ├── EXPERIMENTOS.md
    └── etapas/
```

Princípios:

- **Funções puras** em `features.py`, `markov.py` e `evaluation.py` — recebem arrays/DataFrames e devolvem resultados, sem I/O. São as mais testadas.
- **Objetos ajustados uma vez** (scaler, PCA, K-Means) no treino, serializados e reutilizados; validação e teste só chamam `transform`/`predict`.
- **Uma única configuração** (`config.yaml`) como fonte de parâmetros; cada artefato salvo junto com a configuração, versão das bibliotecas, data e hash dos dados.
- **Dashboard somente leitura:** consome `artifacts/`; nenhuma interação reajusta modelos.
- Módulos e testes são criados **na etapa em que forem necessários**, não antecipadamente.

## 8. Dependências

| Pacote | Uso | Observação |
|---|---|---|
| Python ≥ 3.10 | — | 3.11 recomendado; versão exata registrada na Etapa 01 |
| numpy, pandas | séries e matrizes | — |
| scikit-learn | `StandardScaler`, `PCA`, `KMeans`, `silhouette_score`, `adjusted_rand_score` | `random_state=42`, `n_init` explícito |
| scipy | `linear_sum_assignment` (alinhamento de rótulos) | já é dependência do scikit-learn |
| matplotlib (seaborn opcional) | figuras | — |
| pyyaml | `config.yaml` | — |
| joblib | serialização de modelos | já vem com scikit-learn |
| streamlit | dashboard (Etapa 08) | instalar só na Etapa 08 |
| pytest | testes | — |
| Fonte de dados | coleta (Etapa 01) | candidatas: `yfinance` (API não oficial, sujeita a termos do Yahoo) ou séries históricas da B3 (oficiais, sem ajuste, formato posicional). Decidir na Etapa 01. |

Versões serão fixadas com `==` em `requirements.txt` na Etapa 01, a partir de um ambiente virtual criado para o projeto.

## 9. Critérios de validação

### 9.1 Testes automatizados previstos

| Teste | Verifica | Etapa |
|---|---|---|
| Invariância a dados futuros | alterar preços após `t` não muda indicadores em `≤ t` | 02 |
| Série sintética pequena | fórmulas dos indicadores com valores calculados à mão | 02 |
| Split cronológico | datas estritamente crescentes, sem interseção, sem lacunas entre conjuntos | 03 |
| `fit` apenas no treino | médias do scaler = médias do treino; validação/teste não alteram objetos | 03–05 |
| Matriz conhecida | sequência sintética → `P̂` esperada; linhas somam 1; estados sem saída tratados | 06 |
| Fronteiras | nenhuma transição/origem cruza conjuntos; origens descartadas corretamente por `h` | 06–07 |
| `P̂^h` | `h = 1` reproduz `P̂`; convergência para `π̂` em matriz ergódica sintética | 06 |
| Métricas | Brier/BSS em casos triviais (previsão perfeita = 0; igual à referência ⇒ BSS = 0) | 07 |
| Alinhamento de rótulos | permutação conhecida é recuperada | 05 |

### 9.2 Hipóteses pré-registradas (antes de ver os dados)

- **H1:** para `h = 1`, a previsão de Markov tem BSS > 0 em relação à marginal **e** à persistência na validação.
- **H2:** a vantagem sobre a marginal diminui com `h`, aproximando-se de zero em `h = 20`.
- **H3:** com descarte de componentes, a variante PCA produz partição diferente (ARI < 1) da variante base; o efeito sobre o BSS pode ser positivo, nulo ou negativo.

Qualquer resultado — inclusive a rejeição de H1 — é reportado. A configuração (indicadores, K, representação, `α`) é congelada na validação; o teste é executado uma única vez.

### 9.3 Reprodutibilidade

Um terceiro, com `requirements.txt`, `config.yaml` e o snapshot bruto, deve reproduzir artefatos e métricas com um único comando documentado.

## 10. Decisões desta etapa

Registradas em `docs/DECISOES.md` como D-008 a D-016:

- D-008 Formulação formal do problema (seção 4).
- D-009 Brier multiclasse + BSS como métrica principal; comparação entre configurações via BSS.
- D-010 Comparar `P̂^h` com matriz direta de h passos `Q̂_h`.
- D-011 Suavização aditiva nas linhas de `P̂`, `α` escolhido na validação.
- D-012 Comparação com/sem PCA exige descarte de componentes; avaliar ampliar indicadores na Etapa 02.
- D-013 Origens e transições válidas apenas dentro de um mesmo conjunto.
- D-014 Arquitetura de módulos `src/regimes/` com configuração única e dashboard somente leitura.
- D-015 Protocolo de seleção de K multicritério, sem uso do teste.
- D-016 Hipóteses H1–H3 pré-registradas.

Permanecem como estavam: D-004 (BOVA11) e D-005 (diário; 1/5/10/20) como propostas a confirmar na Etapa 01; D-002 (AWS) pendente.

## 11. Resultados efetivos

Nenhum. Esta etapa não executou experimentos, não coletou dados e não produziu métricas.

## 12. Limitações

- Revisão baseada em documentos; nenhuma suposição sobre os dados foi verificada (disponibilidade, ajustes, faltantes).
- `CLAUDE.md` e outros arquivos citados no README não estavam na pasta do repositório.
- O modelo de relatório técnico da professora ainda não foi incorporado.

## 13. Verificações executadas

- Leitura de README, plano, decisões e instruções do Projeto.
- Consulta aos slides TopIA_01, 02, 04 e 05 para alinhar requisitos e convenções.
- Inspeção da pasta do repositório: estava vazia no momento da revisão (sem `.git`). Os três documentos enviados foram gravados nela.
- Nenhum teste automatizado (não há código).

## 14. Pendências

1. Adicionar à pasta `CLAUDE.md`, `docs/CONFIGURAR_CLAUDE.md`, `prompts/ETAPAS_CLAUDE.md`, `docs/EXPERIMENTOS.md` e `docs/MODELO_RELATORIO_ETAPA.md` (links do README ainda quebrados).
2. Inicializar o repositório Git local (`git init`) e o remoto no GitHub que será enviado à professora.
3. Confirmar com a professora a aprovação do tema e a exigência (ou não) de AWS (D-002).
4. Etapa 01: escolher a fonte de dados, verificar eventos corporativos do BOVA11, decidir data inicial (2018 vs. mais cedo) e fixar versões das bibliotecas.
5. Etapa 02: decidir conjunto final de indicadores (incluindo a ampliação sugerida em D-012).
