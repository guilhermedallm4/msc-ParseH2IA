"""
Runs de treino INDEPENDENTES — robustez do perfil de profundidade (ponto 1 do plano AAAI).

Os checkpoints dos folds do CV principal não foram mantidos em disco (apenas os CSVs de
métricas). Como runs de treino independentes usamos os checkpoints da ABLAÇÃO sem UPOS
(mesmos encoders, mesmo corpus, mistura de tarefas deprel+head): se o perfil de
profundidade (convergência tardia de HEAD, deferral do mBERT etc.) reaparecer nesses
runs, ele não é um artefato de um treino específico.

Também cobre o lado LINEAR na escala large: não existe checkpoint linear
BERTimbau-large com UPOS, mas existe o linear de ablação large.

Modelos avaliados (logit lens deprel/head por camada, test set completo):
  ablacao_lin_bertimbau         best_models_ablacao_linear/neuralmind_bert-base-portuguese-cased
  ablacao_bia_bertimbau         best_models_ablacao_biaffine/neuralmind_bert-base-portuguese-cased
  ablacao_lin_bertimbau_large   best_models_ablacao_linear/neuralmind_bert-large-portuguese-cased
  ablacao_lin_mbert             best_models_ablacao_linear/google-bert_bert-base-multilingual-cased

Saída: layerwise_early_exit_comparison_{tag}.csv  (deprel_acc, uas, las por camada)

Uso:  .venv/bin/python layerwise_independent_runs.py
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

MSC = '/home/guilhermelima/msc'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODELS = {
    'ablacao_lin_bertimbau': ('linear',
        f'{MSC}/best_models_ablacao_linear/neuralmind_bert-base-portuguese-cased',
        'neuralmind/bert-base-portuguese-cased'),
    'ablacao_bia_bertimbau': ('biaffine',
        f'{MSC}/best_models_ablacao_biaffine/neuralmind_bert-base-portuguese-cased',
        'neuralmind/bert-base-portuguese-cased'),
    'ablacao_lin_bertimbau_large': ('linear',
        f'{MSC}/best_models_ablacao_linear/neuralmind_bert-large-portuguese-cased',
        'neuralmind/bert-large-portuguese-cased'),
    'ablacao_lin_mbert': ('linear',
        f'{MSC}/best_models_ablacao_linear/google-bert_bert-base-multilingual-cased',
        'google-bert/bert-base-multilingual-cased'),
}

DEPREL_LABELS = ['det', 'nsubj', 'root', 'obj', 'xcomp', 'punct', 'mark', 'advcl', 'case',
                 'obl', 'amod', 'conj', 'cc', 'nmod', 'advmod', 'flat:name', 'ccomp', 'cop',
                 'acl', 'nummod', 'acl:relcl', 'ccomp:speech', 'parataxis', 'csubj',
                 'aux:pass', 'appos', 'fixed', 'nsubj:pass', 'aux', 'nsubj:outer',
                 'obl:agent', 'expl:impers', 'expl', 'discourse', 'orphan', 'dislocated',
                 'flat', 'flat:foreign', 'iobj', 'vocative', 'csubj:outer', 'list',
                 'reparandum', 'csubj:pass']
DEPREL_TO_IDX = {l: i for i, l in enumerate(DEPREL_LABELS)}


def nb_cells(path):
    nb = json.load(open(path))
    return lambda i: ''.join(nb['cells'][i]['source'])


def load_ablation_linear(path):
    """Classe da célula 10 do ablacao_sem_upos_linear.ipynb (deprel + head lineares)."""
    src = nb_cells(f'{MSC}/ablacao_sem_upos_linear.ipynb')
    ns = {}
    exec('import torch\nimport torch.nn as nn\nimport numpy as np\n'
         'from typing import Optional\n'
         'from transformers import BertPreTrainedModel, AutoModel, AutoConfig', ns)
    exec(compile(src(10), 'ablacao_lin_class', 'exec'), ns)
    config = ns['AutoConfig'].from_pretrained(path)
    model = ns['MultiTaskSentencePredictionEncoderAblacao'].from_pretrained(
        path, config=config, num_deprel_labels=len(DEPREL_LABELS)).to(DEVICE)
    model.eval()

    def heads(hs):
        # hs: [N, L, H] → deprel [N, L, C], head [N, L, num_head_labels]
        return model.deprel_classifier(hs), model.head_classifier(hs)
    return model, heads


def load_ablation_biaffine(path):
    """Classes das células 10–11 do ablacao_sem_upos_biaffine.ipynb (arc/rel, sem upos)."""
    src = nb_cells(f'{MSC}/ablacao_sem_upos_biaffine.ipynb')
    ns = {}
    exec('import torch\nimport torch.nn as nn\nimport numpy as np\n'
         'from typing import Optional\n'
         'from transformers import BertModel, BertPreTrainedModel, AutoConfig', ns)
    for i in (10, 11):
        exec(compile(src(i), f'ablacao_bia_{i}', 'exec'), ns)
    config = ns['AutoConfig'].from_pretrained(path)
    model = ns['MultiTaskSentencePredictionEncoder'].from_pretrained(
        path, config=config, num_deprel_labels=len(DEPREL_LABELS)).to(DEVICE)
    model.eval()

    def heads(hs):
        # mesmo protocolo do forward: rel avaliado na head predita pela camada
        h_arc_dep, h_arc_head = model.arc_dep_mlp(hs), model.arc_head_mlp(hs)
        logits_head = model.arc_biaffine(h_arc_dep, h_arc_head).squeeze(1)   # [N, L, L]
        h_rel_dep, h_rel_head = model.rel_dep_mlp(hs), model.rel_head_mlp(hs)
        logits_rel = model.rel_biaffine(h_rel_dep, h_rel_head)               # [N, C, L, L]
        N, L, _ = logits_head.shape
        arc_preds = logits_head.argmax(-1).clamp(0, L - 1)
        idx = arc_preds.unsqueeze(-1).unsqueeze(-1).expand(N, L, 1, len(DEPREL_LABELS))
        logits_deprel = (logits_rel.permute(0, 2, 3, 1).contiguous()
                         .gather(2, idx).squeeze(2))                         # [N, L, C]
        return logits_deprel, logits_head
    return model, heads


def evaluate(model, heads_fn, tokenizer, sentences, deprel_list, head_list):
    n_layers = model.config.num_hidden_layers + 1
    total = 0
    deprel_c = np.zeros(n_layers)
    uas_c = np.zeros(n_layers)
    las_c = np.zeros(n_layers)

    for i in tqdm(range(len(sentences))):
        tokens = sentences[i]
        g_deprel = np.array([DEPREL_TO_IDX[d] for d in deprel_list[i]])
        g_head = np.array([int(h) for h in head_list[i]])

        inputs = tokenizer(tokens, is_split_into_words=True, return_tensors='pt',
                           padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            out = model.bert(input_ids=inputs['input_ids'],
                             attention_mask=inputs['attention_mask'],
                             token_type_ids=inputs.get('token_type_ids'),
                             output_hidden_states=True)
            hs = torch.stack(out.hidden_states, dim=0)[:, 0]                 # [N, L, H]
            logits_deprel, logits_head = heads_fn(hs)

        word_ids = inputs.word_ids(batch_index=0)
        first_subs, tok_idxs = [], []
        seen = set()
        for pos, w in enumerate(word_ids):
            if w is not None and w not in seen and w < len(tokens):
                seen.add(w)
                first_subs.append(pos)
                tok_idxs.append(w)
        if not first_subs:
            continue
        fs = torch.tensor(first_subs, device=DEVICE)

        p_deprel = logits_deprel[:, fs].argmax(-1).cpu().numpy()             # [N, n_tok]
        p_head = logits_head[:, fs].argmax(-1).cpu().numpy()

        gd, gh = g_deprel[tok_idxs], g_head[tok_idxs]
        total += len(tok_idxs)
        d_ok = p_deprel == gd[None, :]
        h_ok = p_head == gh[None, :]
        deprel_c += d_ok.sum(1)
        uas_c += h_ok.sum(1)
        las_c += (d_ok & h_ok).sum(1)

    return pd.DataFrame({'layer': np.arange(n_layers),
                         'deprel_acc': deprel_c / total,
                         'uas': uas_c / total, 'las': las_c / total})


if __name__ == '__main__':
    os.chdir(MSC)
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    test = load_from_disk(f'{MSC}/data_dois/complaints_dataset_obj_outxpos')['test']

    for tag, (arch, path, tok_name) in MODELS.items():
        print(f'\n════════ {tag} ════════')
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        loader = load_ablation_linear if arch == 'linear' else load_ablation_biaffine
        model, heads_fn = loader(path)

        df = evaluate(model, heads_fn, tokenizer,
                      test['tokens'], test['deprel'], test['head_tags'])
        df.to_csv(f'layerwise_early_exit_comparison_{tag}.csv', index=False)
        print(df.round(4).to_string(index=False))

        del model
        torch.cuda.empty_cache()
    print('\nDONE')
