"""
Análise por camada para o biaffine BERTimbau-large (24 camadas) — generalização de escala.

Reutiliza as funções do inference_notebook_biaffine.ipynb (fonte única de verdade).
Não existe checkpoint linear BERTimbau-large com UPOS em disco (apenas a variante de
ablação sem UPOS) — o lado linear da escala é coberto por layerwise_independent_runs.py.

Saídas (tag = biaffine_bertimbau_large):
  layerwise_preds_store_{tag}.pkl
  layerwise_per_label_stats_{task}_{tag}.csv
  layerwise_impact_summary_{task}_{tag}.csv
  layerwise_early_exit_comparison_{tag}.csv   (métricas por camada L=0..24)
  layerwise_block_skip_comparison_{tag}.csv   (configs adaptadas a 24 blocos)

Uso:  .venv/bin/python layerwise_large_analysis.py
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
MODEL_PATH = f'{MSC}/biaffine_BERTimbau_large'
TOKENIZER_NAME = 'neuralmind/bert-large-portuguese-cased'
TAG = 'biaffine_bertimbau_large'

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
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as scratch:
        os.chdir(scratch)
        try:
            exec(compile(code, tag, 'exec'), ns)
        finally:
            os.chdir(cwd)


if __name__ == '__main__':
    os.chdir(MSC)
    pd.set_option('display.width', 220)

    src = cells(f'{MSC}/inference_notebook_biaffine.ipynb')
    ns = {'display': print}
    for i in [0, 1, 3, 6, 7, 9]:      # seed, imports, labels, MLP/Biaffine, classe, build_model
        exec(compile(src(i), f'bia_{i}', 'exec'), ns)
    exec(DATA_SETUP, ns)

    exec(f"""
from transformers import AutoTokenizer
layerwise_tokenizer = AutoTokenizer.from_pretrained('{TOKENIZER_NAME}')
layerwise_model = build_model(
    '{MODEL_PATH}',
    num_deprel_labels=len(DEPREL_LABELS), num_upos_labels=len(UPOS_LABELS),
)
layerwise_model.eval()
print('camadas:', layerwise_model.config.num_hidden_layers,
      '| hidden:', layerwise_model.config.hidden_size)
""", ns)

    exec(compile(src(46), 'bia_46', 'exec'), ns)   # biaffine_heads_from_hidden + logit lens
    run_in_scratch(ns, src(51), 'bia_51')          # per-label stats (nomes legados no scratch)
    run_in_scratch(ns, src(52), 'bia_52')          # summarize impact
    run_in_scratch(ns, src(54), 'bia_54')          # collect all layers + per_layer_acc
    print(ns['per_layer_acc'].round(4).to_string(index=False))

    # block skip: reutiliza APENAS a função da célula 56 (configs de 12 blocos não se
    # aplicam a 24 camadas) e define configs próprias
    fn_only = src(56).split('block_configs =')[0]
    exec(compile(fn_only, 'bia_56_fn', 'exec'), ns)
    block_configs = {
        'sanity: todos os blocos (== inferência normal)': list(range(24)),
        'APENAS bloco 24':                                [23],
        'APENAS bloco 24, aplicado 24x':                  [23] * 24,
        'blocos 23 e 24':                                 [22, 23],
        'blocos 21 a 24':                                 [20, 21, 22, 23],
        'blocos 13 a 24 (pula metade inferior)':          list(range(12, 24)),
        'nenhum bloco (embeddings puras)':                [],
    }
    block_rows = []
    for name, blocks in block_configs.items():
        r = ns['eval_block_subset'](
            blocks, ns['test_sentences'], ns['test_upos'], ns['test_deprel'], ns['test_head'],
            ns['layerwise_model'], ns['layerwise_tokenizer'], desc=name)
        r['config'] = name
        block_rows.append(r)
    block_comparison = pd.DataFrame(block_rows)[
        ['config', 'blocos_ativos', 'n_blocos', 'upos_acc', 'deprel_acc', 'uas', 'las']
    ]
    print(block_comparison.round(4).to_string(index=False))

    # salva com o sufixo do modelo
    with open(f'layerwise_preds_store_{TAG}.pkl', 'wb') as f:
        pickle.dump({'preds': ns['preds_store'], 'golds': ns['golds_store']}, f)
    for task, df in ns['perlabel_stats'].items():
        df.to_csv(f'layerwise_per_label_stats_{task}_{TAG}.csv', index=False)
    for task, df in ns['impact_summary'].items():
        df.to_csv(f'layerwise_impact_summary_{task}_{TAG}.csv', index=False)
    ns['per_layer_acc'].to_csv(f'layerwise_early_exit_comparison_{TAG}.csv', index=False)
    block_comparison.to_csv(f'layerwise_block_skip_comparison_{TAG}.csv', index=False)
    print(f'[{TAG}] saídas salvas.\nDONE')
