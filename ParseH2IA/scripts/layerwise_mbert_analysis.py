"""
Análise por camada para os modelos mBERT (linear e biaffine) — eixo mono vs. multilíngue.

Reutiliza as funções já validadas nos notebooks (fonte única de verdade):
  inference_notebook_linear.ipynb    — células 3 (labels), 6 (classe), 54/55 (per-label),
                                       56 (impacto), 60 (collect), 65/66 (block skip)
  inference_notebook_biaffine.ipynb  — células 3, 6, 7, 9 (classes), 46 (heads),
                                       51 (per-label), 52 (impacto), 54 (early exit), 56 (block skip)

As células que salvam CSVs com nomes legados (BERTimbau) são executadas em um diretório
scratch; os objetos resultantes são então salvos aqui com sufixo do modelo:

  layerwise_preds_store_{tag}.pkl
  layerwise_per_label_stats_{task}_{tag}.csv
  layerwise_impact_summary_{task}_{tag}.csv
  layerwise_early_exit_comparison_{tag}.csv   (métricas por camada L=0..12)
  layerwise_block_skip_comparison_{tag}.csv

tags: linear_mbert, biaffine_mbert

Uso:  .venv/bin/python layerwise_mbert_analysis.py
"""
import json
import os
import pickle
import tempfile

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import torch

MSC = os.environ.get('PARSEH2IA_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
MBERT_TOKENIZER = 'google-bert/bert-base-multilingual-cased'

DATA_SETUP = """
from datasets import load_from_disk
dataset = load_from_disk(MSC + '/data_dois/complaints_dataset_obj_outxpos')
test_dataset = dataset['test']
test_sentences = test_dataset['tokens']; test_upos = test_dataset['upos']
test_deprel = test_dataset['deprel']; test_head = test_dataset['head_tags']
"""


def cells(path):
    nb = json.load(open(path))
    return lambda i: ''.join(nb['cells'][i]['source'])


def run_in_scratch(ns, code, tag):
    """Executa código que salva nomes legados dentro de um scratch dir descartável."""
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as scratch:
        os.chdir(scratch)
        try:
            exec(compile(code, tag, 'exec'), ns)
        finally:
            os.chdir(cwd)


def save_outputs(ns, tag):
    with open(f'layerwise_preds_store_{tag}.pkl', 'wb') as f:
        pickle.dump({'preds': ns['preds_store'], 'golds': ns['golds_store']}, f)
    for task, df in ns['perlabel_stats'].items():
        df.to_csv(f'layerwise_per_label_stats_{task}_{tag}.csv', index=False)
    for task, df in ns['impact_summary'].items():
        df.to_csv(f'layerwise_impact_summary_{task}_{tag}.csv', index=False)
    ns['per_layer_acc'].to_csv(f'layerwise_early_exit_comparison_{tag}.csv', index=False)
    ns['block_comparison'].to_csv(f'layerwise_block_skip_comparison_{tag}.csv', index=False)
    print(f'[{tag}] saídas salvas.')


def eval_all_layers(ns):
    """Métricas por camada a partir do store coletado (usa eval_layer_combo do notebook)."""
    n_layers = ns['preds_store'][0]['upos'].shape[0]
    df = pd.DataFrame([
        ns['eval_layer_combo'](ns['preds_store'], ns['golds_store'], L, L, L)
        for L in range(n_layers)
    ]).rename(columns={'upos_layer': 'layer'}).drop(columns=['deprel_layer', 'head_layer'])
    ns['per_layer_acc'] = df
    print(df.round(4).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# LINEAR — mBERT
# ══════════════════════════════════════════════════════════════════════════════
def run_linear_mbert():
    tag = 'linear_mbert'
    print(f'\n════════ {tag} ════════')
    src = cells(f'{MSC}/inference_notebook_linear.ipynb')
    ns = {'display': print}
    for i in [0, 1, 3, 6]:                       # seed, imports, labels, classe
        exec(compile(src(i), f'lin_{i}', 'exec'), ns)
    exec(DATA_SETUP, ns)

    exec("""
from transformers import AutoConfig, AutoTokenizer
layerwise_config    = AutoConfig.from_pretrained('%s/linear_mbert_base')
layerwise_tokenizer = AutoTokenizer.from_pretrained('%s')
layerwise_model = MultiTaskSentencePredictionEncoder.from_pretrained(
    '%s/linear_mbert_base', config=layerwise_config,
    num_deprel_labels=len(DEPREL_LABELS), num_upos_labels=len(UPOS_LABELS)
).to('cuda' if torch.cuda.is_available() else 'cpu')
layerwise_model.eval()
print('modelo linear mBERT carregado')
""" % (MSC, MBERT_TOKENIZER, MSC), ns)

    exec(compile(src(54), 'lin_54', 'exec'), ns)             # fn per-label
    run_in_scratch(ns, src(55), 'lin_55')                    # roda per-label (salva legado no scratch)
    run_in_scratch(ns, src(56), 'lin_56')                    # summarize impact
    run_in_scratch(ns, src(60), 'lin_60')                    # collect all layers
    exec(compile(src(61), 'lin_61', 'exec'), ns)             # eval_layer_combo + per_layer_acc
    eval_all_layers(ns)
    exec(compile(src(65), 'lin_65', 'exec'), ns)             # fn block skip
    run_in_scratch(ns, src(66), 'lin_66')                    # roda block skip
    print(ns['block_comparison'].round(4).to_string(index=False))

    save_outputs(ns, tag)
    del ns
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# BIAFFINE — mBERT
# ══════════════════════════════════════════════════════════════════════════════
def run_biaffine_mbert():
    tag = 'biaffine_mbert'
    print(f'\n════════ {tag} ════════')
    src = cells(f'{MSC}/inference_notebook_biaffine.ipynb')
    ns = {'display': print}
    for i in [0, 1, 3, 6, 7, 9]:                 # seed, imports, labels, MLP/Biaffine, classe, build_model
        exec(compile(src(i), f'bia_{i}', 'exec'), ns)
    exec(DATA_SETUP, ns)

    exec("""
from transformers import AutoTokenizer
layerwise_tokenizer = AutoTokenizer.from_pretrained('%s')
layerwise_model = build_model(
    '%s/biaffine_mbert_base/checkpoint-29480',
    num_deprel_labels=len(DEPREL_LABELS), num_upos_labels=len(UPOS_LABELS),
)
layerwise_model.eval()
print('modelo biaffine mBERT carregado')
""" % (MBERT_TOKENIZER, MSC), ns)

    exec(compile(src(46), 'bia_46', 'exec'), ns)             # biaffine_heads_from_hidden + logit lens
    run_in_scratch(ns, src(51), 'bia_51')                    # per-label (salva legado no scratch)
    run_in_scratch(ns, src(52), 'bia_52')                    # summarize impact
    run_in_scratch(ns, src(54), 'bia_54')                    # early exit collect + per_layer_acc
    eval_all_layers(ns)
    run_in_scratch(ns, src(56), 'bia_56')                    # block skip
    print(ns['block_comparison'].round(4).to_string(index=False))

    save_outputs(ns, tag)
    del ns
    torch.cuda.empty_cache()


if __name__ == '__main__':
    os.chdir(MSC)
    pd.set_option('display.width', 220)
    run_linear_mbert()
    run_biaffine_mbert()
    print('\nDONE')
