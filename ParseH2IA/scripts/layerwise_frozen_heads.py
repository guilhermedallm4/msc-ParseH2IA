"""
Cabeças re-treinadas sobre o encoder CONGELADO do biaffine BERTimbau-large.

Pergunta: se lermos a camada onde cada tarefa SATURA — com uma cabeça treinada para
essa camada (em vez do logit lens, que usa a cabeça da camada final) — recuperamos o
desempenho do MTL implantado?

Camadas de saturação (derivadas de layerwise_probe_results.csv e do logit lens):
  L=10  saturação informacional de UPOS   (probe >= 99% do final)
  L=18  saturação informacional de DEPREL (probe >= 99% do final)
  L=23  saturação comportamental de HEAD  (UAS >= 95% do final no MTL)
  L=24  camada final (referência superior)

Para cada camada treinamos DUAS famílias de cabeça (3 seeds cada) sobre features
congeladas do encoder:
  linear   — upos/deprel/head = nn.Linear por token (head posicional, 200 classes)
  biaffine — upos linear + arc/rel MLPs + biaffine (Dozat & Manning), protocolo de
             treino idêntico ao _compute_loss do modelo original (teacher forcing)

Configuração "roteada": cada tarefa lê da SUA camada de saturação com a SUA cabeça
(UPOS@10, DEPREL@18, HEAD@23). Comparação final contra o MTL implantado fine-tunado
ponta a ponta (linha de referência).

Saída: layerwise_frozen_heads_results.csv  (family, layer, seed, métricas)

Uso:  .venv/bin/python layerwise_frozen_heads.py
"""
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

MSC = os.environ.get('PARSEH2IA_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = f'{MSC}/biaffine_BERTimbau_large'
TOKENIZER_NAME = 'neuralmind/bert-large-portuguese-cased'
LAYERS = (10, 18, 23, 24)          # saturação UPOS / DEPREL / HEAD / final
SEEDS = (0, 1, 2)
EPOCHS_LINEAR = 30
EPOCHS_BIAFFINE = 15
BATCH_SENTS = 32
LR = 1e-3
WD = 1e-4
NUM_HEAD_LABELS = 200              # mesma formulação posicional da arquitetura linear

UPOS_LABELS = ['DET', 'NOUN', 'VERB', 'PUNCT', 'SCONJ', 'ADP', 'ADJ', 'CCONJ', 'ADV',
               'PROPN', 'AUX', 'NUM', 'PRON', 'SYM', 'X', 'INTJ']
DEPREL_LABELS = ['det', 'nsubj', 'root', 'obj', 'xcomp', 'punct', 'mark', 'advcl', 'case',
                 'obl', 'amod', 'conj', 'cc', 'nmod', 'advmod', 'flat:name', 'ccomp', 'cop',
                 'acl', 'nummod', 'acl:relcl', 'ccomp:speech', 'parataxis', 'csubj',
                 'aux:pass', 'appos', 'fixed', 'nsubj:pass', 'aux', 'nsubj:outer',
                 'obl:agent', 'expl:impers', 'expl', 'discourse', 'orphan', 'dislocated',
                 'flat', 'flat:foreign', 'iobj', 'vocative', 'csubj:outer', 'list',
                 'reparandum', 'csubj:pass']
UPOS_TO_IDX = {l: i for i, l in enumerate(UPOS_LABELS)}
DEPREL_TO_IDX = {l: i for i, l in enumerate(DEPREL_LABELS)}


def nb_cells(path):
    nb = json.load(open(path))
    return lambda i: ''.join(nb['cells'][i]['source'])


# MLP e Biaffine: mesmas classes do notebook (fonte única de verdade)
_ns = {}
exec('import torch\nimport torch.nn as nn\nimport numpy as np\n'
     'from typing import Optional\nfrom transformers import BertModel, BertPreTrainedModel',
     _ns)
_NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'notebooks',
                   '07_inferencia_biaffine.ipynb')
exec(compile(nb_cells(_NB)(7), 'nb_cell7', 'exec'), _ns)  # célula 7: classes MLP/Biaffine
MLP, Biaffine = _ns['MLP'], _ns['Biaffine']


class LinearHeads(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.upos = nn.Linear(hidden, len(UPOS_LABELS))
        self.deprel = nn.Linear(hidden, len(DEPREL_LABELS))
        self.head = nn.Linear(hidden, NUM_HEAD_LABELS)

    def forward(self, hs):                     # hs [B, L, H]
        return self.upos(hs), self.deprel(hs), self.head(hs)


class BiaffineHeads(nn.Module):
    def __init__(self, hidden, arc_hidden=500, rel_hidden=100, mlp_dropout=0.33):
        super().__init__()
        self.upos = nn.Linear(hidden, len(UPOS_LABELS))
        self.arc_dep, self.arc_head = MLP(hidden, arc_hidden, mlp_dropout), MLP(hidden, arc_hidden, mlp_dropout)
        self.rel_dep, self.rel_head = MLP(hidden, rel_hidden, mlp_dropout), MLP(hidden, rel_hidden, mlp_dropout)
        self.arc_biaffine = Biaffine(arc_hidden, out_features=1, bias_x=True, bias_y=False)
        self.rel_biaffine = Biaffine(rel_hidden, out_features=len(DEPREL_LABELS),
                                     bias_x=True, bias_y=True)

    def forward(self, hs, pad_mask=None):      # hs [B, L, H]; pad_mask [B, L] True=pad
        logits_upos = self.upos(hs)
        logits_head = self.arc_biaffine(self.arc_dep(hs), self.arc_head(hs)).squeeze(1)
        if pad_mask is not None:
            logits_head = logits_head.masked_fill(pad_mask.unsqueeze(1), -1e4)
        logits_rel = self.rel_biaffine(self.rel_dep(hs), self.rel_head(hs))  # [B, C, L, L]
        return logits_upos, logits_head, logits_rel


def extract_features(sentences, upos, deprel, head, layers):
    """Features congeladas [n_layers_sel, L_i, H] + labels alinhadas por sentença."""
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    encoder = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE)
    encoder.eval()

    feats, labels = [], []
    for i in tqdm(range(len(sentences)), desc='extract'):
        tokens = sentences[i]
        inputs = tokenizer(tokens, is_split_into_words=True, return_tensors='pt',
                           padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            out = encoder(input_ids=inputs['input_ids'],
                          attention_mask=inputs['attention_mask'],
                          token_type_ids=inputs.get('token_type_ids'),
                          output_hidden_states=True)
            hs = torch.stack([out.hidden_states[l] for l in layers], dim=0)[:, 0]

        word_ids = inputs.word_ids(batch_index=0)
        L = hs.shape[1]
        lab = {'upos': np.full(L, -100), 'deprel': np.full(L, -100),
               'head': np.full(L, -100), 'first_subs': [], 'tok_idxs': []}
        seen = set()
        for pos, w in enumerate(word_ids):
            if w is not None and w not in seen and w < len(tokens):
                seen.add(w)
                lab['upos'][pos] = UPOS_TO_IDX[upos[i][w]]
                lab['deprel'][pos] = DEPREL_TO_IDX[deprel[i][w]]
                h = int(head[i][w])
                lab['head'][pos] = h if h < L and h < NUM_HEAD_LABELS else -100
                lab['first_subs'].append(pos)
                lab['tok_idxs'].append(w)
        if not lab['first_subs']:
            continue
        feats.append(hs.half().cpu())
        labels.append(lab)

    del encoder
    torch.cuda.empty_cache()
    return feats, labels


def make_batch(feats, labels, idxs, layer_pos):
    """Padding de um lote de sentenças para a camada selecionada."""
    L_max = max(feats[j].shape[1] for j in idxs)
    H = feats[idxs[0]].shape[2]
    hs = torch.zeros(len(idxs), L_max, H)
    pad = torch.ones(len(idxs), L_max, dtype=torch.bool)
    y = {t: torch.full((len(idxs), L_max), -100, dtype=torch.long)
         for t in ('upos', 'deprel', 'head')}
    for b, j in enumerate(idxs):
        L = feats[j].shape[1]
        hs[b, :L] = feats[j][layer_pos].float()
        pad[b, :L] = False
        for t in ('upos', 'deprel', 'head'):
            y[t][b, :L] = torch.tensor(labels[j][t])
    return hs.to(DEVICE), pad.to(DEVICE), {t: v.to(DEVICE) for t, v in y.items()}


def train_heads(family, feats, labels, layer_pos, hidden, seed):
    torch.manual_seed(seed)
    heads = (LinearHeads(hidden) if family == 'linear' else BiaffineHeads(hidden)).to(DEVICE)
    opt = torch.optim.AdamW(heads.parameters(), lr=LR, weight_decay=WD)
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    n = len(feats)
    g = torch.Generator().manual_seed(seed)
    epochs = EPOCHS_LINEAR if family == 'linear' else EPOCHS_BIAFFINE

    heads.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).tolist()
        for s in range(0, n, BATCH_SENTS):
            hs, pad, y = make_batch(feats, labels, perm[s:s + BATCH_SENTS], layer_pos)
            B, L, _ = hs.shape
            opt.zero_grad()
            if family == 'linear':
                lu, ld, lh = heads(hs)
                loss = (loss_fct(lu.view(-1, lu.shape[-1]), y['upos'].view(-1))
                        + loss_fct(ld.view(-1, ld.shape[-1]), y['deprel'].view(-1))
                        + loss_fct(lh.view(-1, lh.shape[-1]), y['head'].view(-1)))
            else:
                lu, lhd, lrel = heads(hs, pad)
                # protocolo idêntico ao _compute_loss original (teacher forcing)
                safe = y['head'].clamp(0, L - 1)
                idx = safe.unsqueeze(-1).unsqueeze(-1).expand(B, L, 1, len(DEPREL_LABELS))
                lrel_gold = (lrel.permute(0, 2, 3, 1).contiguous()
                             .gather(2, idx).squeeze(2))
                loss = (loss_fct(lu.view(-1, lu.shape[-1]), y['upos'].view(-1))
                        + loss_fct(lhd.reshape(B * L, L), y['head'].view(-1))
                        + loss_fct(lrel_gold.reshape(B * L, -1), y['deprel'].view(-1)))
            loss.backward()
            opt.step()

    heads.eval()
    return heads


def evaluate_heads(family, heads, feats, labels, layer_pos):
    """Métricas no protocolo padrão (1º subtoken, golds word-level)."""
    total = upos_c = deprel_c = uas_c = las_c = 0
    with torch.no_grad():
        for j in range(len(feats)):
            hs = feats[j][layer_pos].float().unsqueeze(0).to(DEVICE)
            fs = torch.tensor(labels[j]['first_subs'], device=DEVICE)
            if family == 'linear':
                lu, ld, lh = heads(hs)
                p_head = lh[0, fs].argmax(-1).cpu().numpy()
            else:
                lu, lhd, lrel = heads(hs)
                L = hs.shape[1]
                arc = lhd.argmax(-1).clamp(0, L - 1)
                idx = arc.unsqueeze(-1).unsqueeze(-1).expand(1, L, 1, len(DEPREL_LABELS))
                ld = lrel.permute(0, 2, 3, 1).contiguous().gather(2, idx).squeeze(2)
                p_head = lhd[0, fs].argmax(-1).cpu().numpy()
            p_upos = lu[0, fs].argmax(-1).cpu().numpy()
            p_deprel = ld[0, fs].argmax(-1).cpu().numpy()

            g_upos = np.array(labels[j]['upos'])[labels[j]['first_subs']]
            g_deprel = np.array(labels[j]['deprel'])[labels[j]['first_subs']]
            g_head = np.array(labels[j]['head'])[labels[j]['first_subs']]

            valid = g_head != -100
            total += valid.sum()
            upos_c += ((p_upos == g_upos) & valid).sum()
            d_ok = (p_deprel == g_deprel) & valid
            h_ok = (p_head == g_head) & valid
            deprel_c += d_ok.sum()
            uas_c += h_ok.sum()
            las_c += (d_ok & h_ok).sum()
    return {'upos_acc': upos_c / total, 'deprel_acc': deprel_c / total,
            'uas': uas_c / total, 'las': las_c / total}


if __name__ == '__main__':
    os.chdir(MSC)
    from datasets import load_from_disk
    ds = load_from_disk(f'{MSC}/data_dois/complaints_dataset_obj_outxpos')
    tr, te = ds['train'], ds['test']

    print(f'Extraindo features congeladas das camadas {LAYERS}...')
    f_tr, l_tr = extract_features(tr['tokens'], tr['upos'], tr['deprel'], tr['head_tags'], LAYERS)
    f_te, l_te = extract_features(te['tokens'], te['upos'], te['deprel'], te['head_tags'], LAYERS)
    hidden = f_tr[0].shape[2]

    rows = []
    for layer_pos, layer in enumerate(LAYERS):
        for family in ('linear', 'biaffine'):
            for seed in SEEDS:
                heads = train_heads(family, f_tr, l_tr, layer_pos, hidden, seed)
                m = evaluate_heads(family, heads, f_te, l_te, layer_pos)
                m.update({'family': family, 'layer': layer, 'seed': seed})
                rows.append(m)
                print(f'L={layer:2d} {family:8s} seed={seed} | ' +
                      ' '.join(f'{k}={v:.4f}' for k, v in m.items() if k not in
                               ('family', 'layer', 'seed')))
                del heads
                torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv('layerwise_frozen_heads_results.csv', index=False)

    df = pd.DataFrame(rows)
    df.to_csv('layerwise_frozen_heads_results.csv', index=False)
    agg = (df.groupby(['family', 'layer'])[['upos_acc', 'deprel_acc', 'uas', 'las']]
             .agg(['mean', 'std']))
    print('\n── Resumo (média ± dp sobre 3 seeds) ──')
    print(agg.round(4).to_string())
    print('\nReferência MTL implantado (fine-tune ponta a ponta): '
          'UPOS 0.9934 | DEPREL 0.9779 | UAS 0.9152 | LAS 0.9056')
    print('DONE')
