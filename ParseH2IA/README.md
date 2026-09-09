# ParseH2IA — Parsing de Dependências para o Português Brasileiro

Código da dissertação de mestrado que reimplementa e estende o
[PortParser](https://aclanthology.org/2024.propor-1.1/) para o português brasileiro,
sobre o corpus **Porttinari**: modelos multi-tarefa (UPOS + DEPREL + HEAD) com
encoders BERT e dois cabeçotes de predição (linear e biaffine), acompanhados de um
estudo de explicabilidade por camada (logit lens, probes, early exit, block skip).

*Master's dissertation code: multi-task BERT dependency parsers for Brazilian
Portuguese (Porttinari treebank) with linear and biaffine heads, plus a layer-wise
explainability study.*

## Resultados principais

| Modelo | UAS | LAS | UPOS |
|---|---|---|---|
| **BtL-Lin-MTL** (BERTimbau Large, linear, MTL) | **94,99** | **93,84** | 99,28 |
| PortParser (referência, Lopes et al. 2024) | 96,08 | 94,61 | 99,09 |

Encoders avaliados: BERTimbau Base/Large, mBERT, JabuticaBERT.
Variantes: cabeçote {linear, biaffine} × {MTL com UPOS (cU), ablação sem UPOS (sU)}.

Do estudo de explicabilidade: a hierarquia UPOS→DEPREL→HEAD emerge em profundidade
(informação de UPOS satura na camada ~3; ligações sintáticas só se resolvem no quarto
superior da rede); um etiquetador UPOS com encoder truncado na camada 3 mantém 98,35%
de acurácia com speedup de 3,67×.

## Estrutura

```
ParseH2IA/
├── notebooks/
│   ├── 01_otimizacao_treino_linear.ipynb      # Optuna + treino, cabeçote linear
│   ├── 02_otimizacao_treino_biaffine.ipynb    # Optuna + treino, cabeçote biaffine
│   ├── 03_otimizacao_ablacao.ipynb            # Optuna das ablações sem UPOS
│   ├── 04_treino_ablacao_sem_upos_linear.ipynb
│   ├── 05_treino_ablacao_sem_upos_biaffine.ipynb
│   ├── 06_inferencia_linear.ipynb             # inferência + métricas (linear)
│   ├── 07_inferencia_biaffine.ipynb           # inferência + métricas (biaffine)
│   ├── 08_inferencia_ablacao_linear.ipynb
│   ├── 09_inferencia_ablacao_biaffine.ipynb
│   ├── 10_reproducao_portparser.ipynb         # baseline PortParser
│   └── 11_explicabilidade.ipynb               # AUTOCONTIDO: logit lens, early exit,
│                                              #   block skip, probes, exit UPOS
├── scripts/                                   # análises da dissertação e do artigo
│   ├── layerwise_probes.py                    # probes por camada (5 encoders, 5 seeds)
│   ├── layerwise_frozen_heads.py              # cabeças re-treinadas no encoder congelado
│   ├── layerwise_mbert_analysis.py            # perfil por camada do mBERT
│   ├── layerwise_large_analysis.py            # perfil do BERTimbau-large (24 camadas)
│   ├── layerwise_independent_runs.py          # replicação em runs independentes
│   ├── layerwise_comparison_aaai.py           # figuras/tabelas + bootstrap pareado
│   ├── upos_early_exit.py                     # tagger UPOS truncado + benchmark
│   ├── gerar_graficos.py                      # figuras da dissertação
│   ├── label_instability_analysis.py          # instabilidade por rótulo
│   └── latex_*.py                             # tabelas LaTeX
├── artifacts/
│   ├── results/                               # JSONL de métricas por época (todos os modelos)
│   ├── layerwise/                             # CSVs da análise por camada e early exit
│   ├── figures/                               # figuras (PDF) da dissertação e do artigo
│   ├── instability/                           # resultados de instabilidade de treino
│   └── latex_tables/                          # tabelas geradas
├── requirements.txt
└── README.md
```

## Reprodução

### 1. Ambiente

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # ambiente completo
# pip install -r requirements_optuna.txt # apenas otimização/treino
```

GPU recomendada: os treinos e a análise por camada foram executados em uma
NVIDIA RTX 4090 (24 GB); os notebooks de inferência rodam em GPUs menores.

### 2. Dados e checkpoints (não versionados)

O corpus **Porttinari** deve ser obtido junto aos autores/portal do
[NILC](http://www.nilc.icmc.usp.br/) e processado para o formato HuggingFace
`datasets` esperado pelos notebooks
(`data_dois/complaints_dataset_obj_outxpos`, splits train/val/test com colunas
`tokens`, `upos`, `deprel`, `head_tags`). Os checkpoints dos modelos treinados
(`linear_BERTimbau_base/`, `biaffine_BERTImbau_base/`, etc.) são grandes demais
para o GitHub — treine-os com os notebooks 01–05 ou solicite aos autores.

Os notebooks e scripts localizam dados e checkpoints pela variável de ambiente
`PARSEH2IA_DATA` (default: diretório pai do repositório — isto é, clone o
repositório dentro da pasta que contém `data_dois/` e os checkpoints, ou exporte
`PARSEH2IA_DATA=/caminho/para/dados`).

### 3. Ordem de execução

1. **Treino**: notebooks 01–05 (Optuna: 10 trials TPE; treino final: 40 épocas).
   Logging no Weights & Biases é opcional (`WANDB_API_KEY`).
2. **Avaliação**: notebooks 06–09 reproduzem as métricas da dissertação
   (acurácia UPOS/DEPREL, UAS, LAS; 1º subtoken, decodificação gulosa, sem MST).
3. **Baseline**: notebook 10 reproduz o PortParser no mesmo split.
4. **Explicabilidade**: o notebook **11** é autocontido — reproduz logit lens,
   early exit, block skip, probes por camada e o etiquetador UPOS com saída
   antecipada, gravando os CSVs em `outputs_explicabilidade/` (os valores de
   referência estão em `artifacts/layerwise/`). Os experimentos mais pesados
   (probes com 5 encoders, cabeças congeladas no modelo de 24 camadas, análises
   mBERT/large) ficam nos scripts de `scripts/`.

## Citação

Dissertação de mestrado, Universidade Federal de Pelotas (UFPel), 2026.
Artigo do estudo por camada: *"Information, Behavior, and Format: Where Syntax
Lives in BERT Depends on Who Is Asking"* (em submissão).

```bibtex
@mastersthesis{lima2026parseh2ia,
  title  = {ParseH2IA: parsing de dependências multi-tarefa para o português
            brasileiro com encoders BERT},
  author = {Lima, Guilherme Dallman},
  school = {Universidade Federal de Pelotas},
  year   = {2026}
}
```
