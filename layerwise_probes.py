"""
Probes lineares por camada (5 seeds) sobre os encoders FINE-TUNADOS — ponto 2 do plano AAAI.

Diferença para o logit lens: em vez de ler camadas intermediárias com a cabeça treinada
na camada final (limite inferior), treinamos um probe linear POR CAMADA sobre o encoder
congelado. Mede diretamente a informação linearmente decodificável em cada profundidade,
com desvio-padrão entre seeds (ponto 1: variância de treinamento no nível do probe).

Encoders (o encoder é fine-tunado junto com a cabeça, logo difere entre arquiteturas —
comparar probes lin vs bia testa a claim "o decoder remodela o encoder" sem o viés da
cabeça de leitura):

  lin_bertimbau        linear_BERTimbau_base
  bia_bertimbau        biaffine_BERTImbau_base/checkpoint-24321
  lin_mbert            linear_mbert_base
  bia_mbert            biaffine_mbert_base/checkpoint-29480
  bia_bertimbau_large  biaffine_BERTimbau_large

Protocolo: probe = nn.Linear(hidden, C) sobre o 1º subtoken de cada palavra;
treino em 3.000 sentenças do split train (amostradas com seed 42), avaliação no test
completo; AdamW lr=1e-3, wd=1e-4, 30 épocas, batch 8192; 5 seeds (0..4).
Tarefas: UPOS (16), DEPREL (44), HEAD (posicional, como na arquitetura linear).

Saída: layerwise_probe_results.csv  (encoder, layer, task, seed, acc)

Uso:  .venv/bin/python layerwise_probes.py
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

MSC = '/home/guilhermelima/msc'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_TRAIN_SENTS = 3000
SEEDS = (0, 1, 2, 3, 4)
EPOCHS = 30
BATCH = 8192
LR = 1e-3
WD = 1e-4

ENCODERS = {
    'lin_bertimbau':       (f'{MSC}/linear_BERTimbau_base',
                            'neuralmind/bert-base-portuguese-cased'),
    'bia_bertimbau':       (f'{MSC}/biaffine_BERTImbau_base/checkpoint-24321',
                            'neuralmind/bert-base-portuguese-cased'),
    'lin_mbert':           (f'{MSC}/linear_mbert_base',
                            'google-bert/bert-base-multilingual-cased'),
    'bia_mbert':           (f'{MSC}/biaffine_mbert_base/checkpoint-29480',
                            'google-bert/bert-base-multilingual-cased'),
    'bia_bertimbau_large': (f'{MSC}/biaffine_BERTimbau_large',
                            'neuralmind/bert-large-portuguese-cased'),
}

# mesmos vocabulários de labels dos notebooks
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


def extract_hidden_states(encoder, tokenizer, sentences, upos, deprel, head):
    """Estados ocultos (todas as camadas, fp16, CPU) no 1º subtoken de cada palavra."""
    feats, golds = [], {'upos': [], 'deprel': [], 'head': []}
    encoder.eval()
    for i in tqdm(range(len(sentences)), desc='extract'):
        tokens = sentences[i]
        inputs = tokenizer(tokens, is_split_into_words=True, return_tensors='pt',
                           padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            out = encoder(input_ids=inputs['input_ids'],
                          attention_mask=inputs['attention_mask'],
                          token_type_ids=inputs.get('token_type_ids'),
                          output_hidden_states=True)
            hs = torch.stack(out.hidden_states, dim=0)[:, 0]      # [n_layers, L, H]

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

        feats.append(hs[:, torch.tensor(first_subs, device=DEVICE)].half().cpu())
        golds['upos'].append(np.array([UPOS_TO_IDX[upos[i][t]] for t in tok_idxs]))
        golds['deprel'].append(np.array([DEPREL_TO_IDX[deprel[i][t]] for t in tok_idxs]))
        golds['head'].append(np.array([int(head[i][t]) for t in tok_idxs]))

    X = torch.cat(feats, dim=1)                                   # [n_layers, N, H]
    y = {t: torch.tensor(np.concatenate(golds[t]), dtype=torch.long) for t in golds}
    return X, y


def train_probe(X_tr, y_tr, X_te, y_te, n_classes, seed):
    """Treina um probe linear e retorna acurácia no teste."""
    torch.manual_seed(seed)
    probe = nn.Linear(X_tr.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=WD)
    loss_fct = nn.CrossEntropyLoss()

    n = X_tr.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            opt.zero_grad()
            loss = loss_fct(probe(X_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        preds = torch.cat([probe(X_te[s:s + BATCH]).argmax(-1)
                           for s in range(0, X_te.shape[0], BATCH)])
    return (preds == y_te).float().mean().item()


def main():
    os.chdir(MSC)
    from datasets import load_from_disk
    from transformers import AutoModel, AutoTokenizer

    dataset = load_from_disk(f'{MSC}/data_dois/complaints_dataset_obj_outxpos')
    rng = np.random.default_rng(42)
    tr_idx = rng.choice(len(dataset['train']), size=N_TRAIN_SENTS, replace=False)
    train = dataset['train'].select(tr_idx)
    test = dataset['test']

    rows = []
    for enc_name, (path, tok_name) in ENCODERS.items():
        print(f'\n════════ {enc_name} ════════')
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        # AutoModel extrai o encoder do checkpoint MTL (ignora cabeças, remove prefixo)
        encoder = AutoModel.from_pretrained(path).to(DEVICE)

        X_tr, y_tr = extract_hidden_states(
            encoder, tokenizer, train['tokens'], train['upos'],
            train['deprel'], train['head_tags'])
        X_te, y_te = extract_hidden_states(
            encoder, tokenizer, test['tokens'], test['upos'],
            test['deprel'], test['head_tags'])
        del encoder
        torch.cuda.empty_cache()

        n_head_classes = int(max(y_tr['head'].max(), y_te['head'].max())) + 1
        n_classes = {'upos': len(UPOS_LABELS), 'deprel': len(DEPREL_LABELS),
                     'head': n_head_classes}
        n_layers = X_tr.shape[0]
        print(f'{n_layers} camadas | train tokens: {X_tr.shape[1]} | test tokens: {X_te.shape[1]}')

        for layer in tqdm(range(n_layers), desc='probes'):
            Xl_tr = X_tr[layer].float().to(DEVICE)
            Xl_te = X_te[layer].float().to(DEVICE)
            for task in ('upos', 'deprel', 'head'):
                yl_tr = y_tr[task].to(DEVICE)
                yl_te = y_te[task].to(DEVICE)
                for seed in SEEDS:
                    acc = train_probe(Xl_tr, yl_tr, Xl_te, yl_te, n_classes[task], seed)
                    rows.append({'encoder': enc_name, 'layer': layer,
                                 'task': task, 'seed': seed, 'acc': acc})
            del Xl_tr, Xl_te
            torch.cuda.empty_cache()

        del X_tr, X_te
        # checkpoint parcial por robustez
        pd.DataFrame(rows).to_csv('layerwise_probe_results.csv', index=False)

    df = pd.DataFrame(rows)
    df.to_csv('layerwise_probe_results.csv', index=False)
    agg = (df.groupby(['encoder', 'task', 'layer'])['acc']
             .agg(['mean', 'std']).reset_index())
    print('\nResumo (última camada de cada encoder):')
    last = agg.loc[agg.groupby(['encoder', 'task'])['layer'].idxmax()]
    print(last.round(4).to_string(index=False))
    print('\nDONE')


if __name__ == '__main__':
    main()
