"""
Comparação da análise por camada (logit lens) — material AAAI/NeurIPS.

4 modelos: {linear, biaffine} × {BERTimbau-base (monolíngue), mBERT (multilíngue)}.

Consome (gerados pelos notebooks e por layerwise_mbert_analysis.py):
  layerwise_preds_store_{linear,biaffine,linear_mbert,biaffine_mbert}.pkl
  layerwise_impact_summary_{task}[_{tag}].csv
  layerwise_block_skip_comparison[_{tag}].csv

Produz:
  imagens/layerwise_fig_per_layer.pdf        Fig. 1 — métricas por camada, 4 modelos (CI 95%)
  imagens/layerwise_fig_convergence.pdf      Fig. 2 — camada de convergência por tag (BERTimbau)
  imagens/layerwise_fig_task_depth.pdf       Fig. 3 — profundidade de convergência por tarefa, 4 modelos
  latex_tables/layerwise_aaai_tables.tex     Tabelas booktabs
  layerwise_significance.csv                 Bootstrap pareado (efeito de arquitetura e de encoder)

Uso:  .venv/bin/python layerwise_comparison_aaai.py
"""
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RNG_SEED = 42
N_BOOT   = 10_000
FINAL    = 12

# (arch, encoder) → sufixo dos arquivos ('' = legado BERTimbau)
MODELS = {
    ('linear',   'bertimbau'): 'linear',
    ('biaffine', 'bertimbau'): 'biaffine',
    ('linear',   'mbert'):     'linear_mbert',
    ('biaffine', 'mbert'):     'biaffine_mbert',
}
IMPACT_SUFFIX = {
    ('linear',   'bertimbau'): '',
    ('biaffine', 'bertimbau'): '_biaffine',
    ('linear',   'mbert'):     '_linear_mbert',
    ('biaffine', 'mbert'):     '_biaffine_mbert',
}
BLOCK_FILE = {
    ('linear',   'bertimbau'): 'layerwise_block_skip_comparison.csv',
    ('biaffine', 'bertimbau'): 'layerwise_block_skip_comparison_biaffine.csv',
    ('linear',   'mbert'):     'layerwise_block_skip_comparison_linear_mbert.csv',
    ('biaffine', 'mbert'):     'layerwise_block_skip_comparison_biaffine_mbert.csv',
}
MODEL_LBL  = {('linear', 'bertimbau'): 'Linear/BERTimbau',
              ('biaffine', 'bertimbau'): 'Biaffine/BERTimbau',
              ('linear', 'mbert'): 'Linear/mBERT',
              ('biaffine', 'mbert'): 'Biaffine/mBERT'}
COLORS     = {'linear': '#0072B2', 'biaffine': '#D55E00'}
LINESTYLES = {'bertimbau': '-', 'mbert': '--'}
METRICS    = ('upos_acc', 'deprel_acc', 'uas', 'las')
METRIC_LBL = {'upos_acc': 'UPOS accuracy', 'deprel_acc': 'DEPREL accuracy',
              'uas': 'UAS', 'las': 'LAS'}

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
    'legend.fontsize': 7.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.dpi': 150, 'pdf.fonttype': 42,
})


# ──────────────────────────────────────────────────────────────────────────────
# 1. Stores → contagens por sentença (para bootstrap pareado)
# ──────────────────────────────────────────────────────────────────────────────
def sentence_counts(store):
    preds, golds = store['preds'], store['golds']
    n_sent = len(preds)
    n_layers = preds[0]['upos'].shape[0]
    counts = {m: np.zeros((n_layers, n_sent), dtype=np.int32) for m in METRICS}
    totals = np.zeros(n_sent, dtype=np.int32)
    for s, (p, g) in enumerate(zip(preds, golds)):
        totals[s] = len(g['upos'])
        u_ok = p['upos'] == g['upos'][None, :]
        d_ok = p['deprel'] == g['deprel'][None, :]
        h_ok = p['head'] == g['head'][None, :]
        counts['upos_acc'][:, s]   = u_ok.sum(1)
        counts['deprel_acc'][:, s] = d_ok.sum(1)
        counts['uas'][:, s]        = h_ok.sum(1)
        counts['las'][:, s]        = (h_ok & d_ok).sum(1)
    return counts, totals


counts, totals = {}, None
for key, tag in MODELS.items():
    with open(f'layerwise_preds_store_{tag}.pkl', 'rb') as f:
        store = pickle.load(f)
    counts[key], t = sentence_counts(store)
    # pareamento exige o mesmo conjunto de sentenças/tokens em todos os modelos
    if totals is None:
        totals = t
    else:
        assert np.array_equal(totals, t), f'store desalinhado: {tag}'
    del store

n_layers = counts[('linear', 'bertimbau')]['upos_acc'].shape[0]
n_sent = len(totals)
grand_total = totals.sum()
print(f'{n_sent} sentenças | {grand_total} tokens | {n_layers} camadas | {len(MODELS)} modelos')


# ──────────────────────────────────────────────────────────────────────────────
# 2. Métricas por camada + IC 95% (mesmos índices de resample → tudo pareado)
# ──────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(RNG_SEED)
boot_idx = rng.integers(0, n_sent, size=(N_BOOT, n_sent))
boot_tot = totals[boot_idx].sum(1)

point, ci_lo, ci_hi, boot_final = {}, {}, {}, {}
for key in MODELS:
    for m in METRICS:
        c = counts[key][m]
        point[(key, m)] = c.sum(1) / grand_total
        boot = c[:, boot_idx].sum(2) / boot_tot[None, :]
        ci_lo[(key, m)] = np.percentile(boot, 2.5, axis=1)
        ci_hi[(key, m)] = np.percentile(boot, 97.5, axis=1)
        boot_final[(key, m)] = boot[FINAL]


def paired_test(key_a, key_b, m):
    """Δ = A − B na camada final, IC 95% e p-valor bilateral (bootstrap pareado)."""
    d_point = point[(key_a, m)][FINAL] - point[(key_b, m)][FINAL]
    d_boot = boot_final[(key_a, m)] - boot_final[(key_b, m)]
    lo, hi = np.percentile(d_boot, [2.5, 97.5])
    p = max(2 * min((d_boot <= 0).mean(), (d_boot >= 0).mean()), 1.0 / N_BOOT)
    return d_point, lo, hi, p


sig_rows = []
# (a) efeito da arquitetura, por encoder: biaffine − linear
for enc in ('bertimbau', 'mbert'):
    for m in METRICS:
        d, lo, hi, p = paired_test(('biaffine', enc), ('linear', enc), m)
        sig_rows.append({'contrast': f'biaffine-linear|{enc}', 'metric': m,
                         'a': point[(('biaffine', enc), m)][FINAL],
                         'b': point[(('linear', enc), m)][FINAL],
                         'delta': d, 'ci_low': lo, 'ci_high': hi, 'p_value': p})
# (b) efeito do encoder, por arquitetura: BERTimbau − mBERT (ganho monolíngue)
for arch in ('linear', 'biaffine'):
    for m in METRICS:
        d, lo, hi, p = paired_test((arch, 'bertimbau'), (arch, 'mbert'), m)
        sig_rows.append({'contrast': f'bertimbau-mbert|{arch}', 'metric': m,
                         'a': point[((arch, 'bertimbau'), m)][FINAL],
                         'b': point[((arch, 'mbert'), m)][FINAL],
                         'delta': d, 'ci_low': lo, 'ci_high': hi, 'p_value': p})
sig_df = pd.DataFrame(sig_rows)
sig_df.to_csv('layerwise_significance.csv', index=False)
print('\nBootstrap pareado (camada final):')
print(sig_df.round(4).to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Figura 1 — métricas por camada, 4 modelos
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.8), sharex=True)
for ax, m in zip(axes.flat, METRICS):
    for key in MODELS:
        arch, enc = key
        x = np.arange(n_layers)
        ax.plot(x, point[(key, m)], marker='o', ms=2.5, lw=1.2,
                color=COLORS[arch], ls=LINESTYLES[enc], label=MODEL_LBL[key])
        ax.fill_between(x, ci_lo[(key, m)], ci_hi[(key, m)],
                        color=COLORS[arch], alpha=0.10, lw=0)
    ax.set_title(METRIC_LBL[m])
    ax.set_xticks(range(0, n_layers, 2))
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
for ax in axes[1]:
    ax.set_xlabel('Layer (0 = embeddings)')
for ax in axes[:, 0]:
    ax.set_ylabel('Score')
axes[0, 0].legend(loc='upper left', frameon=True)
fig.tight_layout()
fig.savefig('imagens/layerwise_fig_per_layer.pdf', bbox_inches='tight')
plt.close(fig)
print('\nFig. 1 salva: imagens/layerwise_fig_per_layer.pdf')


# ──────────────────────────────────────────────────────────────────────────────
# 4. Figura 2 — convergência por tag (BERTimbau, linear vs biaffine)
# ──────────────────────────────────────────────────────────────────────────────
def load_impact(task, key):
    return pd.read_csv(f'layerwise_impact_summary_{task}{IMPACT_SUFFIX[key]}.csv')


fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9))
for ax, task, top_n in ((axes[0], 'upos', 12), (axes[1], 'deprel', 12)):
    lin = load_impact(task, ('linear', 'bertimbau')).set_index('label')
    bia = load_impact(task, ('biaffine', 'bertimbau')).set_index('label')
    labels = lin.sort_values('support', ascending=False).head(top_n).index
    y = np.arange(len(labels))
    ax.barh(y + 0.2, lin.loc[labels, 'conv_layer'], height=0.38,
            color=COLORS['linear'], label='Linear')
    ax.barh(y - 0.2, bia.loc[labels, 'conv_layer'], height=0.38,
            color=COLORS['biaffine'], label='Biaffine')
    ax.set_yticks(y, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel('Convergence layer')
    ax.set_title(f'{task.upper()} (BERTimbau)')
    ax.set_xlim(0, 12.5)
    ax.grid(alpha=0.3, axis='x')
axes[0].legend(loc='lower right')
fig.tight_layout()
fig.savefig('imagens/layerwise_fig_convergence.pdf', bbox_inches='tight')
plt.close(fig)
print('Fig. 2 salva: imagens/layerwise_fig_convergence.pdf')


# ──────────────────────────────────────────────────────────────────────────────
# 5. Figura 3 — profundidade de convergência por tarefa, 4 modelos
# ──────────────────────────────────────────────────────────────────────────────
conv_summary = {}
for task in ('upos', 'deprel', 'head'):
    for key in MODELS:
        s = load_impact(task, key)
        for col in ('best_layer_prob', 'conv_layer'):
            conv_summary[(task, key, col)] = np.average(s[col].astype(float),
                                                        weights=s['support'])

fig, ax = plt.subplots(figsize=(4.6, 2.9))
tasks = ('upos', 'deprel', 'head')
width = 0.2
for k, key in enumerate(MODELS):
    arch, enc = key
    vals = [conv_summary[(t, key, 'conv_layer')] for t in tasks]
    ax.bar(np.arange(len(tasks)) + (k - 1.5) * width, vals, width,
           color=COLORS[arch], alpha=1.0 if enc == 'bertimbau' else 0.55,
           label=MODEL_LBL[key])
ax.set_xticks(range(len(tasks)), labels=[t.upper() for t in tasks])
ax.set_ylabel('Convergence layer\n(support-weighted)')
ax.set_ylim(0, 12.5)
ax.grid(alpha=0.3, axis='y')
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig('imagens/layerwise_fig_task_depth.pdf', bbox_inches='tight')
plt.close(fig)
print('Fig. 3 salva: imagens/layerwise_fig_task_depth.pdf')


# ──────────────────────────────────────────────────────────────────────────────
# 6. Tabelas LaTeX
# ──────────────────────────────────────────────────────────────────────────────
def pct(x):
    return f'{100 * x:.2f}'


lines = []
lines.append('% Gerado por layerwise_comparison_aaai.py — não editar manualmente')
lines.append('% Requer: \\usepackage{booktabs, multirow}')
lines.append('')

# Tabela 1 — early exit, 4 modelos
lines.append('\\begin{table}[t]')
lines.append('\\centering')
lines.append('\\caption{Layer-wise read-out (logit lens) on the Porttinari test set '
             f'({n_sent:,} sentences, {grand_total:,} tokens). Heads are trained on layer~12; '
             'early exit reads all tasks from layer $L$.}')
lines.append('\\label{tab:early-exit}')
lines.append('\\begin{tabular}{llcccc}')
lines.append('\\toprule')
lines.append('Model & $L$ & UPOS & DEPREL & UAS & LAS \\\\')
sel_layers = [FINAL, 11, 10, 9, 8]
for key in MODELS:
    lines.append('\\midrule')
    for k, L in enumerate(sel_layers):
        vals = [pct(point[(key, m)][L]) for m in METRICS]
        first = f'\\multirow{{{len(sel_layers)}}}{{*}}{{{MODEL_LBL[key]}}}' if k == 0 else ''
        layer_lbl = f'{L} (full)' if L == FINAL else str(L)
        lines.append(f'{first} & {layer_lbl} & ' + ' & '.join(vals) + ' \\\\')
lines.append('\\bottomrule')
lines.append('\\end{tabular}')
lines.append('\\end{table}')
lines.append('')

# Tabela 2 — significância (dois contrastes)
lines.append('\\begin{table}[t]')
lines.append('\\centering')
lines.append('\\caption{Final-layer contrasts with 95\\% CI and two-sided $p$-values from a '
             f'paired bootstrap over sentences ({N_BOOT:,} resamples; shared resample indices).}}')
lines.append('\\label{tab:significance}')
lines.append('\\begin{tabular}{llcc}')
lines.append('\\toprule')
lines.append('Contrast & Metric & $\\Delta$ [95\\% CI] & $p$ \\\\')
lines.append('\\midrule')
contrast_lbl = {
    'biaffine-linear|bertimbau': 'Biaffine $-$ Linear (BERTimbau)',
    'biaffine-linear|mbert':     'Biaffine $-$ Linear (mBERT)',
    'bertimbau-mbert|linear':    'BERTimbau $-$ mBERT (Linear)',
    'bertimbau-mbert|biaffine':  'BERTimbau $-$ mBERT (Biaffine)',
}
prev = None
for _, r in sig_df.iterrows():
    if prev is not None and r.contrast != prev:
        lines.append('\\midrule')
    prev = r.contrast
    lbl = contrast_lbl[r.contrast] if r.metric == METRICS[0] else ''
    p_str = f'$<${1.0 / N_BOOT:.4f}' if r.p_value <= 1.0 / N_BOOT else f'{r.p_value:.4f}'
    lines.append(f'{lbl} & {METRIC_LBL[r.metric]} & '
                 f'{100 * r.delta:+.2f} [{100 * r.ci_low:+.2f}, {100 * r.ci_high:+.2f}] & {p_str} \\\\')
lines.append('\\bottomrule')
lines.append('\\end{tabular}')
lines.append('\\end{table}')
lines.append('')

# Tabela 3 — block skip (4 modelos, UPOS/LAS por compacidade)
block = {key: pd.read_csv(BLOCK_FILE[key]).set_index('config') for key in MODELS}
block_name_map = {
    'sanity: todos os blocos (== inferência normal)': 'All blocks (normal)',
    'APENAS bloco 12':                                'Block 12 only',
    'APENAS bloco 12, aplicado 12x':                  'Block 12 $\\times$12',
    'blocos 11 e 12':                                 'Blocks 11--12',
    'blocos 9 a 12':                                  'Blocks 9--12',
    'blocos 7 a 12 (pula metade inferior)':           'Blocks 7--12',
    'nenhum bloco (embeddings puras)':                'None (embeddings)',
}
lines.append('\\begin{table}[t]')
lines.append('\\centering')
lines.append('\\caption{Block-skip ablation (UPOS / LAS): embeddings fed directly into a subset of '
             'encoder blocks. Depth is compositional: the last block alone, or iterated, fails '
             'across architectures and pretraining regimes.}')
lines.append('\\label{tab:block-skip}')
lines.append('\\begin{tabular}{lcccc}')
lines.append('\\toprule')
lines.append(' & \\multicolumn{2}{c}{BERTimbau} & \\multicolumn{2}{c}{mBERT} \\\\')
lines.append('\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}')
lines.append('Active blocks & Linear & Biaffine & Linear & Biaffine \\\\')
lines.append('\\midrule')
for cfg, name in block_name_map.items():
    cells_row = []
    for enc in ('bertimbau', 'mbert'):
        for arch in ('linear', 'biaffine'):
            r = block[(arch, enc)].loc[cfg]
            cells_row.append(f'{pct(r.upos_acc)} / {pct(r.las)}')
    lines.append(f'{name} & ' + ' & '.join(cells_row) + ' \\\\')
lines.append('\\bottomrule')
lines.append('\\end{tabular}')
lines.append('\\end{table}')
lines.append('')

# Tabela 4 — convergência ponderada, 4 modelos
lines.append('\\begin{table}[t]')
lines.append('\\centering')
lines.append('\\caption{Support-weighted mean convergence layer per task '
             '(first layer with acc $\\geq 95\\%$ of final).}')
lines.append('\\label{tab:convergence}')
lines.append('\\begin{tabular}{lccc}')
lines.append('\\toprule')
lines.append('Model & UPOS & DEPREL & HEAD \\\\')
lines.append('\\midrule')
for key in MODELS:
    vals = [f"{conv_summary[(t, key, 'conv_layer')]:.1f}" for t in ('upos', 'deprel', 'head')]
    lines.append(f'{MODEL_LBL[key]} & ' + ' & '.join(vals) + ' \\\\')
lines.append('\\bottomrule')
lines.append('\\end{tabular}')
lines.append('\\end{table}')

os.makedirs('latex_tables', exist_ok=True)
with open('latex_tables/layerwise_aaai_tables.tex', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('Tabelas salvas: latex_tables/layerwise_aaai_tables.tex')

# Resumo numérico p/ redação da discussão
print('\n── Convergência ponderada (conv_layer) ──')
for key in MODELS:
    print(MODEL_LBL[key], {t: round(conv_summary[(t, key, 'conv_layer')], 1)
                           for t in ('upos', 'deprel', 'head')})
print('\nOK')


# ══════════════════════════════════════════════════════════════════════════════
# PARTE 2 — probes por camada, escala (BERTimbau-large) e runs independentes
# (arquivos gerados por layerwise_probes.py, layerwise_large_analysis.py e
#  layerwise_independent_runs.py; seções puladas se os arquivos não existirem)
# ══════════════════════════════════════════════════════════════════════════════
PROBE_LBL = {'lin_bertimbau': 'Linear/BERTimbau', 'bia_bertimbau': 'Biaffine/BERTimbau',
             'lin_mbert': 'Linear/mBERT', 'bia_mbert': 'Biaffine/mBERT',
             'bia_bertimbau_large': 'Biaffine/BERTimbau-large'}
PROBE_STYLE = {'lin_bertimbau': ('#0072B2', '-'), 'bia_bertimbau': ('#D55E00', '-'),
               'lin_mbert': ('#0072B2', '--'), 'bia_mbert': ('#D55E00', '--'),
               'bia_bertimbau_large': ('#009E73', '-')}

extra_lines = []

# ── Fig. 4: probes treinados por camada (5 seeds, ±dp) ────────────────────────
if os.path.exists('layerwise_probe_results.csv'):
    probe = pd.read_csv('layerwise_probe_results.csv')
    pagg = (probe.groupby(['encoder', 'task', 'layer'])['acc']
                 .agg(['mean', 'std']).reset_index())

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.6), sharey=True)
    for ax, task in zip(axes, ('upos', 'deprel', 'head')):
        for enc in ('lin_bertimbau', 'bia_bertimbau', 'lin_mbert', 'bia_mbert'):
            s = pagg[(pagg.encoder == enc) & (pagg.task == task)].sort_values('layer')
            if len(s) == 0:
                continue
            color, ls = PROBE_STYLE[enc]
            ax.plot(s['layer'], s['mean'], color=color, ls=ls, lw=1.2,
                    marker='o', ms=2.2, label=PROBE_LBL[enc])
            ax.fill_between(s['layer'], s['mean'] - s['std'], s['mean'] + s['std'],
                            color=color, alpha=0.25, lw=0)
        ax.set_title({'upos': 'UPOS', 'deprel': 'DEPREL', 'head': 'HEAD (positional)'}[task])
        ax.set_xlabel('Layer')
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('Probe accuracy')
    axes[0].legend(fontsize=6.5, loc='lower right')
    fig.tight_layout()
    fig.savefig('imagens/layerwise_fig_probes.pdf', bbox_inches='tight')
    plt.close(fig)
    print('Fig. 4 salva: imagens/layerwise_fig_probes.pdf')

    print('\nProbe HEAD (última camada):')
    last_head = pagg[pagg.task == 'head'].loc[
        lambda d: d.groupby('encoder')['layer'].idxmax()]
    print(last_head.round(4).to_string(index=False))
    print('desvio-padrão máximo entre seeds (todos os probes):',
          round(pagg['std'].max(), 4))

# ── Fig. 5: profundidade normalizada — generalização de escala ────────────────
# modelos base (12L): curvas já em memória (point); modelos large (24L): CSVs
SCALE_LARGE_FILES = {
    'Biaffine/BERTimbau-large (24L)': ('layerwise_early_exit_comparison_biaffine_bertimbau_large.csv', '#D55E00', ':'),
    'Linear/BERTimbau-large (24L)*':  ('layerwise_early_exit_comparison_ablacao_lin_bertimbau_large.csv', '#0072B2', ':'),
}

if all(os.path.exists(f) for f, _, _ in SCALE_LARGE_FILES.values()):
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for key, lbl, ls in ((('linear', 'bertimbau'), 'Linear/BERTimbau-base (12L)', '-'),
                         (('biaffine', 'bertimbau'), 'Biaffine/BERTimbau-base (12L)', '-')):
        las = point[(key, 'las')]
        x = np.arange(len(las)) / (len(las) - 1)
        ax.plot(x, las, color=COLORS[key[0]], ls=ls, lw=1.3, marker='o', ms=2.2, label=lbl)
    for lbl, (fname, color, ls) in SCALE_LARGE_FILES.items():
        df = pd.read_csv(fname).sort_values('layer')
        x = df['layer'] / df['layer'].max()
        ax.plot(x, df['las'], color=color, ls=ls, lw=1.3, marker='o', ms=2.2, label=lbl)
    ax.set_xlabel('Normalized depth $\\ell / L$')
    ax.set_ylabel('LAS')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig('imagens/layerwise_fig_scale.pdf', bbox_inches='tight')
    plt.close(fig)
    print('Fig. 5 salva: imagens/layerwise_fig_scale.pdf')

# ── Tabela 5: replicação em runs de treino independentes ─────────────────────
INDEP = {
    'Linear/BERTimbau':  ('layerwise_early_exit_comparison_ablacao_lin_bertimbau.csv',
                          ('linear', 'bertimbau')),
    'Biaffine/BERTimbau': ('layerwise_early_exit_comparison_ablacao_bia_bertimbau.csv',
                           ('biaffine', 'bertimbau')),
    'Linear/mBERT':      ('layerwise_early_exit_comparison_ablacao_lin_mbert.csv',
                          ('linear', 'mbert')),
}


def uas_half_layer(uas_curve):
    """Primeira camada com UAS >= 50% do valor final (profundidade de meia-resolução)."""
    final = uas_curve[-1]
    return int(np.argmax(uas_curve >= 0.5 * final))


if all(os.path.exists(f) for f, _ in INDEP.values()):
    rep_rows = []
    for lbl, (fname, key) in INDEP.items():
        indep = pd.read_csv(fname).sort_values('layer')
        main_uas = point[(key, 'uas')]
        rep_rows.append({
            'model': lbl,
            'main_las': point[(key, 'las')][FINAL],
            'indep_las': indep['las'].iloc[-1],
            'main_half_layer': uas_half_layer(main_uas),
            'indep_half_layer': uas_half_layer(indep['uas'].to_numpy()),
        })
    rep_df = pd.DataFrame(rep_rows)
    print('\nReplicação (run principal vs. run independente sem UPOS):')
    print(rep_df.round(4).to_string(index=False))

    extra_lines.append('\\begin{table}[t]')
    extra_lines.append('\\centering')
    extra_lines.append('\\caption{Replication across independent training runs (ablation '
                       'checkpoints trained without the UPOS head). Half-resolution depth = '
                       'first layer with UAS $\\geq 50\\%$ of its final value.}')
    extra_lines.append('\\label{tab:replication}')
    extra_lines.append('\\begin{tabular}{lcccc}')
    extra_lines.append('\\toprule')
    extra_lines.append(' & \\multicolumn{2}{c}{LAS} & \\multicolumn{2}{c}{Half-res.\\ layer} \\\\')
    extra_lines.append('\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}')
    extra_lines.append('Model & Main & Indep. & Main & Indep. \\\\')
    extra_lines.append('\\midrule')
    for _, r in rep_df.iterrows():
        extra_lines.append(f"{r['model']} & {pct(r['main_las'])} & {pct(r['indep_las'])} & "
                           f"{int(r['main_half_layer'])} & {int(r['indep_half_layer'])} \\\\")
    extra_lines.append('\\bottomrule')
    extra_lines.append('\\end{tabular}')
    extra_lines.append('\\end{table}')

# ── Tabela 6: probes vs logit lens (informação vs. comportamento) ─────────────
if os.path.exists('layerwise_probe_results.csv'):
    def layer95(vals):
        final = vals[-1]
        return int(np.argmax(np.asarray(vals) >= 0.95 * final))

    probe_map = {('linear', 'bertimbau'): 'lin_bertimbau',
                 ('biaffine', 'bertimbau'): 'bia_bertimbau',
                 ('linear', 'mbert'): 'lin_mbert',
                 ('biaffine', 'mbert'): 'bia_mbert'}
    metric_of = {'upos': 'upos_acc', 'deprel': 'deprel_acc', 'head': 'uas'}

    extra_lines.append('')
    extra_lines.append('\\begin{table}[t]')
    extra_lines.append('\\centering')
    extra_lines.append('\\caption{Information vs.\\ behavior: layer at which each read-out reaches '
                       '95\\% of its final value. Per-layer trained probes (5 seeds, '
                       '$\\sigma \\leq 0.003$) locate information much earlier than the deployed '
                       'logit-lens read-out, and the biaffine encoder stores head attachment in a '
                       'relational format invisible to single-token probes (final probe accuracy '
                       'in parentheses).}')
    extra_lines.append('\\label{tab:probes}')
    extra_lines.append('\\begin{tabular}{llcc}')
    extra_lines.append('\\toprule')
    extra_lines.append('Model & Task & Probe & Logit lens \\\\')
    extra_lines.append('\\midrule')
    for key, enc in probe_map.items():
        for task in ('upos', 'deprel', 'head'):
            s = pagg[(pagg.encoder == enc) & (pagg.task == task)].sort_values('layer')
            lens_curve = point[(key, metric_of[task])]
            probe_l = layer95(s['mean'].to_numpy())
            lens_l = layer95(lens_curve)
            probe_final = s['mean'].iloc[-1]
            row_lbl = MODEL_LBL[key] if task == 'upos' else ''
            extra_lines.append(f'{row_lbl} & {task.upper()} & '
                               f'{probe_l} ({pct(probe_final)}) & {lens_l} \\\\')
        extra_lines.append('\\midrule' if key != ('biaffine', 'mbert') else '\\bottomrule')
    extra_lines.append('\\end{tabular}')
    extra_lines.append('\\end{table}')

if extra_lines:
    with open('latex_tables/layerwise_aaai_tables.tex', 'a') as f:
        f.write('\n' + '\n'.join(extra_lines) + '\n')
    print('Tabelas extra anexadas a latex_tables/layerwise_aaai_tables.tex')

print('\nOK (parte 2)')


# ── Tabela 7: cabeças re-treinadas no encoder congelado (camadas de saturação) ─
if os.path.exists('layerwise_frozen_heads_results.csv'):
    fro = pd.read_csv('layerwise_frozen_heads_results.csv')
    fagg = (fro.groupby(['family', 'layer'])[['upos_acc', 'deprel_acc', 'uas', 'las']]
               .agg(['mean', 'std']))
    frozen_lines = []
    frozen_lines.append('')
    frozen_lines.append('\\begin{table}[t]')
    frozen_lines.append('\\centering')
    frozen_lines.append('\\caption{Head families retrained (3 seeds, mean; $\\sigma \\leq 0.003$) '
                        'on the FROZEN 24-layer biaffine-fine-tuned encoder, at each task\\textquotesingle s '
                        'saturation layer. A freshly trained pairwise (biaffine) head recovers '
                        'near-deployed attachment from intermediate layers; a freshly trained '
                        'positional (linear) head cannot extract head positions from the same '
                        'frozen representations at any depth. Deployed end-to-end MTL shown for '
                        'reference.}')
    frozen_lines.append('\\label{tab:frozen}')
    frozen_lines.append('\\begin{tabular}{llcccc}')
    frozen_lines.append('\\toprule')
    frozen_lines.append('Head & Layer & UPOS & DEPREL & UAS & LAS \\\\')
    frozen_lines.append('\\midrule')
    layer_note = {10: '10 (UPOS sat.)', 18: '18 (DEPREL sat.)', 23: '23 (HEAD sat.)', 24: '24 (final)'}
    for family in ('linear', 'biaffine'):
        sub = fagg.loc[family]
        for k, (layer, r) in enumerate(sub.iterrows()):
            first = (f'\\multirow{{{len(sub)}}}{{*}}{{{family.capitalize()}}}' if k == 0 else '')
            frozen_lines.append(
                f"{first} & {layer_note[int(layer)]} & "
                f"{100*r[('upos_acc','mean')]:.2f} & {100*r[('deprel_acc','mean')]:.2f} & "
                f"{100*r[('uas','mean')]:.2f} & {100*r[('las','mean')]:.2f} \\\\")
        frozen_lines.append('\\midrule')
    frozen_lines.append('Deployed MTL & 24 (end-to-end) & 99.34 & 97.79 & 91.52 & 90.56 \\\\')
    frozen_lines.append('\\bottomrule')
    frozen_lines.append('\\end{tabular}')
    frozen_lines.append('\\end{table}')

    with open('latex_tables/layerwise_aaai_tables.tex', 'a') as f:
        f.write('\n'.join(frozen_lines) + '\n')
    print('Tabela frozen anexada a latex_tables/layerwise_aaai_tables.tex')
