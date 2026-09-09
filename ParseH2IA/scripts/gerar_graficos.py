"""
Gera os gráficos para a seção de Resultados e Discussão da dissertação.

Saída: imagens/
  resultados_geral.pdf        — barras horizontais UAS/LAS todos os modelos
  linear_vs_biaffine.pdf      — impacto da arquitetura (agrupado por encoder)
  curvas_treinamento_btl.pdf  — curvas UAS durante o treino (BERTimbau Large)
  curvas_treinamento_enc.pdf  — curvas UAS por encoder (variante Linear MTL)
  instabilidade_deprel.pdf    — top-15 labels mais instáveis (forgetting events)
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ── Diretórios ────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
INST_DIR = os.path.join(BASE, "instability_results")
IMG_DIR  = os.path.join(BASE, "imagens")
os.makedirs(IMG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})

# ── Paleta e estilos ──────────────────────────────────────────────────────────
ENC_COLOR = {
    "BtL": "#1f4e79",   # azul escuro
    "BtB": "#2e75b6",   # azul médio
    "mB":  "#c55a11",   # laranja escuro
    "JaB": "#375623",   # verde escuro
    "PP":  "#7b2d8b",   # roxo (PortParser)
}
VARIANT_HATCH = {
    "Lin-cU":   "",
    "Biaff-cU": "//",
    "Lin-sU":   "..",
    "Biaff-sU": "xx",
}

# ── Dados: tabela principal ───────────────────────────────────────────────────
RESULTS = [
    # (label_display, encoder, variante, upos, uas, las)
    ("PortParser",          "PP",  "Baseline", 99.09, 96.08, 94.61),
    ("BtL — Lin MTL",       "BtL", "Lin-cU",   99.28, 94.99, 93.84),
    ("BtL — Lin sU",        "BtL", "Lin-sU",   None,  94.73, 93.52),
    ("BtL — Biaff MTL",     "BtL", "Biaff-cU", 99.34, 91.52, 90.56),
    ("BtL — Biaff sU",      "BtL", "Biaff-sU", None,  91.96, 90.85),
    ("BtB — Lin MTL",       "BtB", "Lin-cU",   99.22, 93.74, 92.49),
    ("BtB — Lin sU",        "BtB", "Lin-sU",   None,  93.52, 92.23),
    ("BtB — Biaff MTL",     "BtB", "Biaff-cU", 99.27, 89.84, 88.71),
    ("BtB — Biaff sU",      "BtB", "Biaff-sU", None,  90.81, 89.71),
    ("mB — Lin MTL",        "mB",  "Lin-cU",   98.88, 92.70, 91.04),
    ("mB — Lin sU",         "mB",  "Lin-sU",   None,  92.44, 90.81),
    ("mB — Biaff MTL",      "mB",  "Biaff-cU", 98.93, 86.92, 85.42),
    ("mB — Biaff sU",       "mB",  "Biaff-sU", None,  89.11, 87.61),
    ("JaB — Biaff sU",      "JaB", "Biaff-sU", None,  90.04, 88.45),
    ("JaB — Biaff MTL",     "JaB", "Biaff-cU", 98.92, 88.94, 87.44),
    ("JaB — Lin MTL",       "JaB", "Lin-cU",   98.81, 87.78, 86.25),
    ("JaB — Lin sU",        "JaB", "Lin-sU",   None,  87.00, 85.29),
]
# Ordena por UAS descendente
RESULTS_SORTED = sorted(RESULTS, key=lambda r: r[4], reverse=True)

# ── Figura 1: resultados_geral ────────────────────────────────────────────────
def fig_resultados_geral():
    labels  = [r[0]  for r in RESULTS_SORTED]
    encs    = [r[1]  for r in RESULTS_SORTED]
    uas     = [r[4]  for r in RESULTS_SORTED]
    las     = [r[5]  for r in RESULTS_SORTED]
    n       = len(labels)
    y       = np.arange(n)
    h       = 0.35

    fig, ax = plt.subplots(figsize=(10, 7))

    for i, (lbl, enc, u, l) in enumerate(zip(labels, encs, uas, las)):
        color  = ENC_COLOR[enc]
        alpha  = 0.90 if enc != "PP" else 1.0
        ax.barh(y[i] + h/2, u, h, color=color, alpha=alpha,
                edgecolor="white", linewidth=0.5)
        ax.barh(y[i] - h/2, l, h, color=color, alpha=0.50,
                edgecolor="white", linewidth=0.5, hatch="////")
        ax.text(max(u, l) + 0.15, y[i] + h/2,  f"{u:.2f}%", va="center", fontsize=7.5)
        ax.text(max(u, l) + 0.15, y[i] - h/2,  f"{l:.2f}%", va="center", fontsize=7.5)

    # Linhas de referência: PortParser UAS e LAS
    pp_uas = RESULTS[0][4]
    pp_las = RESULTS[0][5]
    ax.axvline(pp_uas, color=ENC_COLOR["PP"], linestyle="--", linewidth=1.2, alpha=0.8,
               label=f"PortParser UAS ({pp_uas:.2f}%)")
    ax.axvline(pp_las, color=ENC_COLOR["PP"], linestyle=":",  linewidth=1.2, alpha=0.8,
               label=f"PortParser LAS ({pp_las:.2f}%)")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Desempenho (%)")
    ax.set_xlim(83, 99)
    ax.set_title("Comparação de Desempenho: UAS e LAS por Modelo")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.invert_yaxis()

    # Legenda encoders
    enc_patches = [mpatches.Patch(color=ENC_COLOR[e], label=n_)
                   for e, n_ in [("BtL","BERTimbau Large"),("BtB","BERTimbau Base"),
                                  ("mB","mBERT"),("JaB","JabuticaBERT"),("PP","PortParser")]]
    uas_patch = mpatches.Patch(facecolor="gray", alpha=0.9, label="UAS (sólido)")
    las_patch = mpatches.Patch(facecolor="gray", alpha=0.5, hatch="////", label="LAS (hachura)")
    ax.legend(handles=enc_patches + [uas_patch, las_patch],
              loc="lower right", fontsize=8, framealpha=0.85)

    plt.tight_layout()
    out = os.path.join(IMG_DIR, "resultados_geral.pdf")
    plt.savefig(out)
    plt.close()
    print(f"  Salvo: {os.path.basename(out)}")


# ── Figura 2: linear_vs_biaffine ─────────────────────────────────────────────
def fig_linear_vs_biaffine():
    """Barras agrupadas: diferença UAS entre Linear e Biaffine, por encoder e tarefa."""
    encoders  = ["BtL", "BtB", "mB", "JaB"]
    enc_names = ["BERTimbau Large", "BERTimbau Base", "mBERT", "JabuticaBERT"]

    # (Lin-cU, Biaff-cU, Lin-sU, Biaff-sU) UAS por encoder
    data = {
        "BtL": {"Lin-cU": 94.99, "Biaff-cU": 91.52, "Lin-sU": 94.73, "Biaff-sU": 91.96},
        "BtB": {"Lin-cU": 93.74, "Biaff-cU": 89.84, "Lin-sU": 93.52, "Biaff-sU": 90.81},
        "mB":  {"Lin-cU": 92.70, "Biaff-cU": 86.92, "Lin-sU": 92.44, "Biaff-sU": 89.11},
        "JaB": {"Lin-cU": 87.78, "Biaff-cU": 88.94, "Lin-sU": 87.00, "Biaff-sU": 90.04},
    }

    variants   = ["Lin-cU", "Biaff-cU", "Lin-sU", "Biaff-sU"]
    var_labels = ["Lin cU", "Biaff cU", "Lin sU", "Biaff sU"]
    colors     = ["#1f4e79", "#5b9bd5", "#843c0c", "#f4b183"]
    hatches    = ["", "//", "..", "xx"]

    x   = np.arange(len(encoders))
    w   = 0.18
    fig, ax = plt.subplots(figsize=(9, 5))

    for j, (var, vlbl, col, hatch) in enumerate(zip(variants, var_labels, colors, hatches)):
        vals = [data[e][var] for e in encoders]
        offset = (j - 1.5) * w
        bars = ax.bar(x + offset, vals, w, label=vlbl,
                      color=col, hatch=hatch, edgecolor="white",
                      alpha=0.88, linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(enc_names, fontsize=9)
    ax.set_ylabel("UAS (%)")
    ax.set_ylim(84, 97.5)
    ax.set_title("UAS por Variante de Arquitetura e Encoder")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.5))

    plt.tight_layout()
    out = os.path.join(IMG_DIR, "linear_vs_biaffine.pdf")
    plt.savefig(out)
    plt.close()
    print(f"  Salvo: {os.path.basename(out)}")


# ── Figura 3: curvas de treinamento BtL ──────────────────────────────────────
def _load_head_uas(stem):
    path = os.path.join(INST_DIR, f"{stem}_head.csv")
    epochs, uas = [], []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                epochs.append(int(r["epoch"]))
                uas.append(float(r["uas"]) * 100)
    return epochs, uas


def fig_curvas_btl():
    curves = [
        ("predictions_results_bertimbau_large_linear",                     "Lin MTL",   "#1f4e79",  "-",  2.0),
        ("predictions_results_bertimbau_large_biaffine",                   "Biaff MTL", "#2e75b6",  "--", 1.8),
        ("bert-large-portuguese-cased_predictions_ablacao_linear",         "Lin sU",    "#843c0c",  "-.", 1.8),
        ("bert-large-portuguese-cased_predictions_ablacao_biaffine",       "Biaff sU",  "#f4b183",  ":",  1.8),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for stem, lbl, col, ls, lw in curves:
        ep, uas = _load_head_uas(stem)
        if ep:
            ax.plot(ep, uas, linestyle=ls, color=col, linewidth=lw, label=lbl)

    ax.set_xlabel("Época")
    ax.set_ylabel("UAS (%)")
    ax.set_title("Curvas de Aprendizado — BERTimbau Large (UAS no conjunto de teste)")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_xlim(1, None)
    ax.set_ylim(0, 100)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(2))
    plt.tight_layout()
    out = os.path.join(IMG_DIR, "curvas_treinamento_btl.pdf")
    plt.savefig(out)
    plt.close()
    print(f"  Salvo: {os.path.basename(out)}")


def fig_curvas_encoders():
    """Curvas UAS: variante Linear MTL para cada encoder."""
    curves = [
        ("predictions_results_bertimbau_large_linear",  "BERTimbau Large", ENC_COLOR["BtL"], "-",  2.0),
        ("predictions_results_bertimbau_base_linear",   "BERTimbau Base",  ENC_COLOR["BtB"], "--", 1.8),
        ("predictions_results_mbert_linear",            "mBERT",           ENC_COLOR["mB"],  "-.", 1.8),
        ("predictions_results_jabuticabert_linear",     "JabuticaBERT",    ENC_COLOR["JaB"], ":",  1.8),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for stem, lbl, col, ls, lw in curves:
        ep, uas = _load_head_uas(stem)
        if ep:
            ax.plot(ep, uas, linestyle=ls, color=col, linewidth=lw, label=lbl)

    ax.set_xlabel("Época")
    ax.set_ylabel("UAS (%)")
    ax.set_title("Curvas de Aprendizado — Linear MTL (UAS no conjunto de teste)")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_xlim(1, None)
    ax.set_ylim(0, 100)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(2))
    plt.tight_layout()
    out = os.path.join(IMG_DIR, "curvas_treinamento_enc.pdf")
    plt.savefig(out)
    plt.close()
    print(f"  Salvo: {os.path.basename(out)}")


# ── Figura 4: instabilidade DEPREL ───────────────────────────────────────────
def _load_instab_deprel(stem):
    path = os.path.join(INST_DIR, f"{stem}_deprel.csv")
    rows = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows[r["label"]] = int(r["forgetting_events"])
    return rows


def fig_instabilidade():
    """Top-15 labels por forgetting médio — 4 variantes do BtL."""
    variants = [
        ("predictions_results_bertimbau_large_linear",                   "Lin MTL",  "#1f4e79", ""),
        ("predictions_results_bertimbau_large_biaffine",                 "Biaff MTL","#2e75b6", "//"),
        ("bert-large-portuguese-cased_predictions_ablacao_linear",       "Lin sU",   "#843c0c", ".."),
        ("bert-large-portuguese-cased_predictions_ablacao_biaffine",     "Biaff sU", "#f4b183", "xx"),
    ]
    data = [(lbl, col, hatch, _load_instab_deprel(stem))
            for stem, lbl, col, hatch in variants]

    # Labels com dados em todos os modelos
    all_labels = set(data[0][3].keys())
    for _, _, _, d in data[1:]:
        all_labels &= set(d.keys())
    # Ordena por forgetting médio
    avg_forg = {lbl: sum(d[lbl] for _, _, _, d in data) / len(data)
                for lbl in all_labels}
    top15 = sorted(all_labels, key=lambda l: -avg_forg[l])[:15]

    y   = np.arange(len(top15))
    h   = 0.18
    fig, ax = plt.subplots(figsize=(9, 6))

    for j, (lbl_v, col, hatch, d) in enumerate(data):
        vals   = [d.get(l, 0) for l in top15]
        offset = (j - 1.5) * h
        ax.barh(y + offset, vals, h, label=lbl_v,
                color=col, hatch=hatch, edgecolor="white",
                alpha=0.88, linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels([f"\\texttt{{{l}}}" if False else l for l in top15], fontsize=9)
    ax.set_xlabel("Eventos de Forgetting (acerto → erro)")
    ax.set_title("Top-15 Rótulos DEPREL Mais Instáveis — BERTimbau Large")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.invert_yaxis()
    plt.tight_layout()
    out = os.path.join(IMG_DIR, "instabilidade_deprel.pdf")
    plt.savefig(out)
    plt.close()
    print(f"  Salvo: {os.path.basename(out)}")


# ── Figura 5: acurácia nos rótulos difíceis ───────────────────────────────────
def fig_labels_dificeis():
    """
    Rótulos com acurácia < 0.90 (labels difíceis).
    Dados do predict_test_df_* — best 4 models (BtL e BtB variantes Linear e Biaffine).
    Valores extraídos do CSV gerado por latex_per_label_accuracy.py.
    """
    # Acurácia por label (4 melhores modelos) — valores do predict_test_df_*
    # Fonte: resultados das tabelas individuais geradas (latex_tables/*.tex)
    models = ["BtL-Lin", "BtL-Biaff", "BtB-Lin", "BtB-Biaff"]
    colors = ["#1f4e79", "#2e75b6", "#843c0c", "#f4b183"]
    # Rótulos difíceis (acc < 0.92 no melhor modelo) e seus valores nos 4 modelos
    hard_labels = {
        "parataxis":    [0.7500, 0.7812, 0.7688, 0.6875],
        "discourse":    [0.6275, 0.7451, 0.6863, 0.6471],
        "vocative":     [0.6667, 0.5000, 0.5000, 0.3333],
        "acl":          [0.8893, 0.8925, 0.8697, 0.8893],
        "orphan":       [0.2143, 0.4286, 0.1429, 0.1429],
        "dislocated":   [0.3529, 0.5882, 0.3529, 0.5294],
        "expl:impers":  [0.8148, 0.8519, 0.9259, 0.9259],
        "advcl":        [0.9153, 0.9176, 0.9039, 0.9268],
        "ccomp:speech": [0.9538, 0.9385, 0.9385, 0.9231],
        "obl":          [0.9364, 0.9376, 0.9341, 0.9376],
    }
    # Ordena por valor médio
    ordered = sorted(hard_labels.items(), key=lambda x: sum(x[1])/len(x[1]))
    hlabels = [k for k, _ in ordered]
    values  = [v for _, v in ordered]

    y = np.arange(len(hlabels))
    h = 0.18
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for j, (m, col) in enumerate(zip(models, colors)):
        vals   = [values[i][j] * 100 for i in range(len(hlabels))]
        offset = (j - 1.5) * h
        bars = ax.barh(y + offset, vals, h, label=m,
                       color=col, edgecolor="white", alpha=0.88, linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(hlabels, fontsize=9)
    ax.set_xlabel("Acurácia (%)")
    ax.set_title("Acurácia nos Rótulos DEPREL Mais Difíceis (4 Melhores Modelos)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    plt.tight_layout()
    out = os.path.join(IMG_DIR, "labels_dificeis.pdf")
    plt.savefig(out)
    plt.close()
    print(f"  Salvo: {os.path.basename(out)}")


# ── Figura 6: delta MTL (impacto do UPOS no UAS) ─────────────────────────────
def fig_impacto_mtl():
    """Diferença UAS e LAS: MTL (com UPOS) − ablação (sem UPOS) por variante."""
    encoders  = ["BtL", "BtB", "mB", "JaB"]
    enc_names = ["BERTimbau Large", "BERTimbau Base", "mBERT", "JabuticaBERT"]
    delta_uas_lin   = [94.99-94.73, 93.74-93.52, 92.70-92.44, 87.78-87.00]
    delta_las_lin   = [93.84-93.52, 92.49-92.23, 91.04-90.81, 86.25-85.29]
    delta_uas_biaff = [91.52-91.96, 89.84-90.81, 86.92-89.11, 88.94-90.04]
    delta_las_biaff = [90.56-90.85, 88.71-89.71, 85.42-87.61, 87.44-88.45]

    x  = np.arange(len(encoders))
    w  = 0.18
    fig, ax = plt.subplots(figsize=(9, 4.5))

    datasets = [
        (delta_uas_lin,   "Lin — UAS",   "#1f4e79", ""),
        (delta_las_lin,   "Lin — LAS",   "#2e75b6", "//"),
        (delta_uas_biaff, "Biaff — UAS", "#843c0c", ".."),
        (delta_las_biaff, "Biaff — LAS", "#f4b183", "xx"),
    ]
    for j, (vals, lbl, col, hatch) in enumerate(datasets):
        offset = (j - 1.5) * w
        bars = ax.bar(x + offset, vals, w, label=lbl,
                      color=col, hatch=hatch, edgecolor="white",
                      alpha=0.88, linewidth=0.5)
        for bar, v in zip(bars, vals):
            ypos = bar.get_height() + 0.03 if v >= 0 else bar.get_height() - 0.18
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    f"{v:+.2f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(enc_names, fontsize=9)
    ax.set_ylabel("ΔUAS / ΔLAS (pp)")
    ax.set_title("Impacto do Multi-Task Learning (MTL): cU − sU por Encoder e Arquitetura")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = os.path.join(IMG_DIR, "impacto_mtl.pdf")
    plt.savefig(out)
    plt.close()
    print(f"  Salvo: {os.path.basename(out)}")


# ── Execução ──────────────────────────────────────────────────────────────────
print("Gerando gráficos para a dissertação...\n")
fig_resultados_geral()
fig_linear_vs_biaffine()
fig_curvas_btl()
fig_curvas_encoders()
fig_instabilidade()
fig_labels_dificeis()
fig_impacto_mtl()
print(f"\nTodos os gráficos salvos em: imagens/")
