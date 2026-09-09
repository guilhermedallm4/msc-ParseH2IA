"""
Early exit dedicado para UPOS: encoder TRUNCADO na camada k + cabeça linear simples.

Motivação (artigo AAAI / layerwise_probe_results.csv): a informação de UPOS satura
cedo — probe linear atinge 98.1% na camada 3 do lin_bertimbau (final ~99.3%). O logit
lens falha nessas camadas apenas por descalibração da cabeça final; uma cabeça
treinada NA camada de saída recupera o teto informacional (cf. layerwise_frozen_heads).

Diferença para layerwise_probes.py: aqui o produto é um MODELO IMPLANTÁVEL — o
encoder é fisicamente truncado (encoder.encoder.layer[:k]), então a economia de
FLOPs é real: custo ≈ k/12 do encoder completo. O probe só mede informação; este
script mede informação + latência de inferência do modelo truncado.

Protocolo:
  encoder  : linear_BERTimbau_base (fine-tunado no MTL), congelado
  cabeça   : nn.Linear(768, 16) sobre o 1º subtoken de cada palavra
  treino   : split train completo (5.893 sents), AdamW lr=1e-3 wd=1e-4,
             30 épocas, batch 8192 tokens, 3 seeds — mesmo protocolo dos probes
  avaliação: test completo (1.683 sents), acurácia word-level no 1º subtoken
  benchmark: latência wall-clock do encoder truncado + cabeça no test set
             (batch 32, fp32, tokenização pré-computada), vs. 12 camadas

Saídas:
  upos_early_exit_results.csv   (layer, seed, upos_acc)
  upos_early_exit_benchmark.csv (layer, params_M, sec_test_set, speedup, acc_mean)
  upos_early_exit_heads/L{k}_seed{s}.pt  (state_dict das cabeças treinadas)

Uso:  .venv/bin/python upos_early_exit.py
"""
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

MSC = os.environ.get('PARSEH2IA_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = f'{MSC}/linear_BERTimbau_base'
TOKENIZER_NAME = 'neuralmind/bert-base-portuguese-cased'
LAYERS = (2, 3, 4, 5, 6, 12)       # candidatas a exit + camada final (referência)
SEEDS = (0, 1, 2)
EPOCHS = 30
BATCH = 8192
LR = 1e-3
WD = 1e-4
BENCH_BATCH = 32

UPOS_LABELS = ['DET', 'NOUN', 'VERB', 'PUNCT', 'SCONJ', 'ADP', 'ADJ', 'CCONJ', 'ADV',
               'PROPN', 'AUX', 'NUM', 'PRON', 'SYM', 'X', 'INTJ']
UPOS_TO_IDX = {l: i for i, l in enumerate(UPOS_LABELS)}


class UposEarlyExitTagger(nn.Module):
    """Modelo implantável: BERT truncado na camada k + classificador linear de UPOS."""

    def __init__(self, encoder, k, n_tags=len(UPOS_LABELS)):
        super().__init__()
        encoder.encoder.layer = encoder.encoder.layer[:k]   # truncamento físico
        self.bert = encoder
        self.head = nn.Linear(encoder.config.hidden_size, n_tags)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        hs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                       token_type_ids=token_type_ids).last_hidden_state
        return self.head(hs)


def extract_features(encoder, tokenizer, sentences, upos):
    """Features fp16 no 1º subtoken de cada palavra, para as camadas em LAYERS."""
    feats, golds = [], []
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
            hs = torch.stack([out.hidden_states[l] for l in LAYERS], dim=0)[:, 0]

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
        golds.append(np.array([UPOS_TO_IDX[upos[i][t]] for t in tok_idxs]))

    X = torch.cat(feats, dim=1)                              # [n_layers_sel, N, H]
    y = torch.tensor(np.concatenate(golds), dtype=torch.long)
    return X, y


def train_head(X_tr, y_tr, seed):
    torch.manual_seed(seed)
    head = nn.Linear(X_tr.shape[1], len(UPOS_LABELS)).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    loss_fct = nn.CrossEntropyLoss()
    n = X_tr.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            opt.zero_grad()
            loss = loss_fct(head(X_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
    head.eval()
    return head


@torch.no_grad()
def evaluate_head(head, X_te, y_te):
    preds = torch.cat([head(X_te[s:s + BATCH]).argmax(-1)
                       for s in range(0, X_te.shape[0], BATCH)])
    return (preds == y_te).float().mean().item()


@torch.no_grad()
def benchmark_latency(k, encoded_batches, warmup=5):
    """Tempo de inferência do tagger truncado em k camadas sobre o test set."""
    from transformers import AutoModel
    encoder = AutoModel.from_pretrained(MODEL_PATH, add_pooling_layer=False)
    model = UposEarlyExitTagger(encoder, k).to(DEVICE)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    for _ in range(warmup):
        model(**encoded_batches[0])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for enc in encoded_batches:
        model(**enc)
    torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    del model
    torch.cuda.empty_cache()
    return sec, params


if __name__ == '__main__':
    os.chdir(MSC)
    from datasets import load_from_disk
    from transformers import AutoModel, AutoTokenizer

    ds = load_from_disk(f'{MSC}/data_dois/complaints_dataset_obj_outxpos')
    tr, te = ds['train'], ds['test']
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print(f'Extraindo features congeladas das camadas {LAYERS}...')
    encoder = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE)
    X_tr, y_tr = extract_features(encoder, tokenizer, tr['tokens'], tr['upos'])
    X_te, y_te = extract_features(encoder, tokenizer, te['tokens'], te['upos'])
    del encoder
    torch.cuda.empty_cache()
    print(f'train tokens: {X_tr.shape[1]} | test tokens: {X_te.shape[1]}')

    os.makedirs('upos_early_exit_heads', exist_ok=True)
    rows = []
    for layer_pos, layer in enumerate(LAYERS):
        Xl_tr = X_tr[layer_pos].float().to(DEVICE)
        Xl_te = X_te[layer_pos].float().to(DEVICE)
        yl_tr, yl_te = y_tr.to(DEVICE), y_te.to(DEVICE)
        for seed in SEEDS:
            head = train_head(Xl_tr, yl_tr, seed)
            acc = evaluate_head(head, Xl_te, yl_te)
            rows.append({'layer': layer, 'seed': seed, 'upos_acc': acc})
            torch.save(head.state_dict(), f'upos_early_exit_heads/L{layer}_seed{seed}.pt')
            print(f'L={layer:2d} seed={seed} | upos_acc={acc:.4f}')
        del Xl_tr, Xl_te
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv('upos_early_exit_results.csv', index=False)

    print('\nBenchmark de latência (test set completo, batch '
          f'{BENCH_BATCH}, tokenização pré-computada)...')
    sents = te['tokens']
    encoded_batches = [
        tokenizer(sents[i:i + BENCH_BATCH], is_split_into_words=True,
                  return_tensors='pt', padding=True, truncation=True).to(DEVICE)
        for i in range(0, len(sents), BENCH_BATCH)]

    acc_mean = df.groupby('layer')['upos_acc'].mean()
    bench = []
    sec_full = None
    for k in sorted(LAYERS, reverse=True):                  # 12 primeiro (referência)
        sec, params = benchmark_latency(k, encoded_batches)
        if k == 12:
            sec_full = sec
        bench.append({'layer': k, 'params_M': round(params, 1),
                      'sec_test_set': round(sec, 3),
                      'speedup': round(sec_full / sec, 2),
                      'acc_mean': round(acc_mean[k], 4)})
        print(f'k={k:2d} | {params:6.1f}M params | {sec:6.3f}s | '
              f'{sec_full / sec:4.2f}x | acc={acc_mean[k]:.4f}')
    pd.DataFrame(bench).sort_values('layer').to_csv(
        'upos_early_exit_benchmark.csv', index=False)

    print('\n── Resumo (média ± dp sobre 3 seeds) ──')
    print(df.groupby('layer')['upos_acc'].agg(['mean', 'std']).round(4).to_string())
    print('\nReferência MTL implantado (12 camadas, cabeça original): UPOS 0.9922')
    print('DONE')
