# Regimes do Mercado

Projeto acadêmico para agrupar períodos de ações da B3 por **retorno e amplitude**,
com K-Means de três grupos, e estudar as transições entre grupos com Cadeias de
Markov. O estudo de referência usa PETR4 semanal; o painel Streamlit permite
escolher outras ações, períodos e frequências (semanal, mensal ou anual).
Sem PCA, LSTM, AWS ou indicadores adicionais.

## Execução

Use Python 3.12 ou superior. No PowerShell, na pasta do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py
```

O painel baixa os dados e executa a análise sozinho; não depende dos CSVs.
Com o ambiente ativado, basta `streamlit run app.py`. Bibliotecas (em
`requirements.txt`): pandas, yfinance, numpy, scikit-learn, matplotlib,
streamlit e plotly.

Para reproduzir o estudo de referência (PETR4 semanal) com CSVs e gráficos PNG:

```powershell
.\.venv\Scripts\python.exe baixar_dados.py
.\.venv\Scripts\python.exe analisar_dados.py
.\.venv\Scripts\python.exe calcular_markov.py
```

Se o ambiente já existe, não é necessário recriá-lo. Aguarde o término de cada
script e avance somente se ele concluir com sucesso. A coleta exige internet;
os demais scripts usam CSVs locais. Os caminhos são relativos aos scripts.
Falhas podem deixar resultados de uma execução anterior no disco.

## Formação das semanas

São consultados preços diários de [PETR4.SA no Yahoo Finance](https://finance.yahoo.com/quote/PETR4.SA/history/)
com `auto_adjust=True`, desde 01/01/2020, limitados a 2026. Fechamento, máxima e
mínima usam o mesmo ajuste por dividendos e desdobramentos. Não representam
necessariamente as cotações nominais negociadas na época; o histórico pode
ser revisado pelo fornecedor.

A agregação usa [resample do pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html)
com `W-FRI`, `closed="right"` e `label="right"`:

- Fechamento: último fechamento disponível na semana.
- Máxima: maior máxima dos pregões da semana.
- Mínima: menor mínima dos pregões da semana.
- Data: sexta-feira de encerramento, mesmo quando ela não teve negociação.

Não se criam preços para feriados. A semana só entra depois de sua sexta-feira
ter terminado. Para o dia atual, o corte conservador é 19h de São Paulo: antes
disso, a sexta atual ainda não é aceita. Rótulos posteriores a 2026 são excluídos.

Preços ausentes, não finitos, não positivos ou inconsistentes são informados
com suas datas. Se houver um preço diário explicitamente inválido, a semana
correspondente é excluída para não apresentar extremos parciais. Também são
excluídas semanas sem dados e semanas sem fechamento anterior válido.
Nenhum retorno atravessa uma semana ausente. A consistência dos preços tolera
apenas erro numérico proporcional a `abs(fechamento) * 1e-10`, sem alterá-los.
Não há consulta a calendário externo para detectar pregões omitidos pela fonte.

As fórmulas, calculadas após agregar, são:

```text
variacao_percentual = (fechamento_semana / fechamento_semana_anterior - 1) * 100
amplitude_percentual = (maxima_semana - minima_semana) / fechamento_semana_anterior * 100
```

A primeira semana é usada apenas como referência de fechamento, sem entrar na
análise por não ter retorno anterior. Assim, o recorte inicial de 01/01/2020 não
introduz uma semana parcial como observação do modelo. Uma coleta válida guarda
uma cópia datada do CSV anterior antes da substituição.

## Min-Max e K-Means

Os primeiros `int(n * 0.8)` registros ordenados formam o treino; os restantes,
o teste. Cada variável tem mínimo e máximo próprios, calculados **só no treino**:

`valor_normalizado = (valor - minimo) / (maximo - minimo)`

O teste usa esses mesmos parâmetros, sem recorte para [0, 1]. Para variável
constante no treino, o denominador é 1: o treino fica em zero e o teste mantém
sua diferença ao mínimo. Menos de três pares distintos impede o agrupamento.

O [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
recebe exclusivamente `variacao_normalizada` e `amplitude_normalizada`, com
`n_clusters=3`, `random_state=42`, `n_init=10`. O ajuste usa apenas o treino;
todos os registros são classificados pelo modelo ajustado. Os grupos são
ordenados pelos centroides do retorno original, do menor para o maior.
Cada coordenada do centroide é convertida com `valor_normalizado * escala + minimo`.

O resumo descreve a base completa; os centroides são do treino. O gráfico usa
retorno semanal no eixo X, amplitude semanal no eixo Y e centroides marcados
com X. Também há o histórico do fechamento semanal e o mapa de calor de Markov.

## Markov semanal

São contados somente pares separados por **sete dias**, dentro do mesmo
conjunto. Não se atravessam lacunas nem a fronteira treino/teste. As linhas
representam a semana atual e as colunas, a próxima semana:

`probabilidade[i, j] = contagem[i, j] / total_de_saidas[i]`

A matriz estimada utiliza apenas o treino. O teste gera frequências separadas
e diferenças em pontos percentuais, sem alterar a matriz estimada. Linhas sem
saídas ficam indefinidas, sem probabilidades inventadas. O terminal informa
contagens, permanência e transição mais frequente de cada grupo. O aviso de
menos de dez saídas é uma cautela descritiva, não um teste estatístico.

Cada observação agora resume uma semana: o retorno compara dois fechamentos
semanais, a amplitude abrange todos os pregões daquela semana e uma transição
representa uma semana de horizonte. A trajetória intrassemanal não é preservada
nesses três preços agregados. Não foi feita comparação de desempenho com outra
frequência. Markov não garante estabilidade temporal ou previsão de preços.

## Resultados em 10/11/2026

Foram recebidos **1.707 pregões válidos**, sem descartes por preço inválido nesta
consulta. Uma semana ainda em andamento foi excluída. Havia **358 semanas
encerradas** com preços; a primeira, rotulada 03/01/2020, serviu de referência.

Foram analisadas **357 semanas**, de **10/01/2020 a 06/11/2026**:
**285 de treino** e **72 de teste**, iniciando em **27/06/2025**.

| Grupo | Semanas | Retorno médio (%) | Amplitude média (%) | Centroide retorno (%) | Centroide amplitude (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 27 | -10,032890 | 13,493227 | -10,172956 | 13,834709 |
| 2 | 285 | 0,324328 | 5,419920 | 0,320801 | 5,535980 |
| 3 | 45 | 8,981083 | 12,899252 | 9,134171 | 13,468279 |

Na base observada, o grupo 1 reúne retornos negativos e amplitudes elevadas.
O grupo 2 tem retorno médio próximo de zero e menor amplitude média, mas inclui
semanas positivas e negativas. O grupo 3 reúne retornos positivos e amplitudes
elevadas. Essas interpretações decorrem dos dados, não de rótulos pré-definidos.

Matriz estimada no treino, **284 transições**, em porcentagem:

| Atual → próxima | Grupo 1 | Grupo 2 | Grupo 3 | Saídas |
| --- | ---: | ---: | ---: | ---: |
| Grupo 1 | 29,17% | 41,67% | 29,17% | 24 |
| Grupo 2 | 5,41% | 82,88% | 11,71% | 222 |
| Grupo 3 | 13,16% | 73,68% | 13,16% | 38 |

As transições mais frequentes por grupo foram **1→2 (10)**, **2→2 (184)** e
**3→2 (28)**. O grupo 2 apresentou a maior permanência, **82,88%**.
As somas exibidas podem diferir de 100% pelo arredondamento de duas casas.

O teste tem **71 transições**. A permanência do grupo 2 foi **85,48%**,
contra 82,88% no treino. Os grupos 1 e 3 possuem apenas **2 e 7 saídas** no
teste, respectivamente; suas frequências exigem cautela. As contagens,
matrizes completas e diferenças são impressas por `calcular_markov.py`.

## Painel interativo (Streamlit + Plotly)

`app.py` importa as funções dos três scripts e executa o mesmo fluxo para a
seleção feita na barra lateral:

1. coleta diária no Yahoo Finance (preços ajustados);
2. agregação semanal (sexta), mensal (fim do mês) ou anual (fim do ano), com
   último fechamento, maior máxima e menor mínima;
3. retorno e amplitude percentuais, com as fórmulas acima;
4. divisão cronológica 80/20;
5. Min-Max ajustado só no treino;
6. K-Means com três grupos ajustado só no treino;
7. matriz de Markov só com transições consecutivas do treino.

**Filtros:** ação (PETR4, VALE3, ITUB4, BBAS3, BBDC4, ABEV3 ou outro código da
B3 digitado), data inicial, data final e frequência. Nada é recalculado até o
clique em **Analisar ativo**. São validados: data inicial anterior à final,
ausência de datas futuras, existência de dados e tamanho da amostra. Períodos
em andamento (semana, mês ou ano atual) e o que passar da data final escolhida
são excluídos.

**Amostra mínima:** com menos de 30 períodos completos (`MINIMO_OBSERVACOES`
em `analisar_dados.py`), o painel mostra só o gráfico e a tabela histórica,
sem K-Means nem Markov. Na prática, a frequência anual fica sempre nesse caso
(2020–2025 são 5 anos analisáveis). Linhas da matriz com menos de 10 saídas
recebem aviso; linhas sem saídas ficam indefinidas.

**Abas:**

- **Visão geral:** ativo, período, frequência, observações, último fechamento,
  retorno acumulado (primeiro ao último fechamento exibido), divisão
  treino/teste e gráfico interativo do fechamento (zoom e valores ao passar o
  mouse).
- **K-Means:** dispersão retorno × amplitude com os três grupos e centroides em
  X (data, valores, grupo e conjunto ao passar o mouse) e tabela-resumo.
- **Cadeias de Markov:** mapa de calor da matriz do treinamento, contagens de
  transições, avisos e consulta por grupo atual com gráfico de barras.
- **Dados e normalização:** dados históricos, parâmetros e dados Min-Max,
  classificação de cada período e botão para baixar a tabela final em CSV.

**Cache:** a consulta ao Yahoo e a análise ficam em `st.cache_data` por uma
hora. Trocar de aba ou de grupo na consulta não baixa nem treina novamente.

Os resultados são probabilidades históricas e não garantem o comportamento
futuro dos ativos. Não há recomendação de compra ou venda.

## Arquivos e verificação

- `baixar_dados.py`: consulta diária e agregação (`baixar_precos_diarios`,
  `agregar_periodos`); executado sozinho, gera a série semanal da PETR4.
- `analisar_dados.py`: Min-Max e K-Means (`normalizar_e_agrupar`) e dois gráficos.
- `calcular_markov.py`: transições (`estimar_markov`), comparação e mapa de calor.
- `app.py`: painel interativo que usa essas funções.
- `dados/petrobras.csv`: `data`, `fechamento`, `maxima`, `minima`,
  `variacao_percentual`, `amplitude_percentual`.
- `dados/petrobras_grupos.csv`: as seis colunas originais, mais
  `variacao_normalizada`, `amplitude_normalizada`, `grupo` e `conjunto`.
- `dados/resumo_grupos.csv`: semanas, médias e centroides (treino) por grupo.
- `dados/markov_treinamento.csv`: matriz de transição estimada no treinamento.
  (Os CSVs são produtos dos scripts; o painel não os lê.)
- Gráficos em `dados/`: `evolucao_petrobras.png`, `grupos_petrobras.png` e
  `markov_petrobras.png`.

Toda a pasta `dados/` é ignorada pelo Git. Arquivos históricos não são entradas
do fluxo atual. Prompts, decisões e verificações auxiliares permanecem em
`../Instrucoes_Codex/`, fora do repositório.

Verificações aprovadas: agregação conhecida, sexta sem pregão, corte de semana
aberta, ausências sem preenchimento, fórmulas, duas escalas só do treino,
três grupos, constantes, teste sem influência no ajuste, transições conhecidas,
exclusão de lacunas e fronteira, contagens reais independentes e três PNGs
inspecionados visualmente. O painel foi testado com PETR4 e VALE3 semanais,
PETR4 mensal e anual, período alterado, datas inválidas, código digitado,
código inexistente e troca de grupo sem novo treino; a normalização exibida
confere com a fórmula e o treino da PETR4 semanal reproduz a matriz acima.
Finalidade acadêmica.
