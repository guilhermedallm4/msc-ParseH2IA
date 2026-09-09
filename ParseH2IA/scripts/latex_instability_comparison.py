"""
Gera tabelas LaTeX de comparação de instabilidade por rótulo DEPREL e UPOS.

Agrupamento: 4 tabelas, uma por modelo base.
Cada tabela compara as 4 variantes do mesmo encoder:
  Lin    — com UPOS, cabeçote linear
  Biaff  — com UPOS, cabeçote biaffine
  sLin   — sem UPOS (ablação), cabeçote linear
  sBiaff — sem UPOS (ablação), cabeçote biaffine

Layout por tabela (um único float table):
  Bloco DEPREL (minipage esq.): top-20 rótulos mais instáveis
    Cabeçalho duplo: Lin | Biaff | sLin | sBiaff
    Por coluna: Fg. (forgetting) + Ac.F (acurácia final)
    Fg. máximo por linha em negrito; Ac.F máxima por linha em negrito.

  Bloco UPOS (minipage dir., apenas Lin/Biaff):
    Mesma estrutura com 2 colunas variante

  Bloco HEAD UAS (abaixo, largura total):
    Épocas selecionadas × 4 UAS (uma coluna por variante)

Saída: latex_tables/00_tabelas_instabilidade_comparacao.tex
"""

import csv
import os

BASE     = os.path.dirname(os.path.abspath(__file__))
INST_DIR = os.path.join(BASE, "instability_results")
OUT_DIR  = os.path.join(BASE, "latex_tables")
os.makedirs(OUT_DIR, exist_ok=True)

DEPREL_ORDER = [
    "det", "nsubj", "root", "obj", "xcomp", "punct", "mark", "advcl", "case",
    "obl", "amod", "conj", "cc", "nmod", "advmod", "flat:name", "ccomp", "cop",
    "acl", "nummod", "acl:relcl", "ccomp:speech", "parataxis", "csubj",
    "aux:pass", "appos", "fixed", "nsubj:pass", "aux", "nsubj:outer",
    "obl:agent", "expl:impers", "expl", "discourse", "orphan", "dislocated",
    "flat", "flat:foreign", "iobj", "vocative", "csubj:outer", "list",
    "reparandum", "csubj:pass",
]
UPOS_ORDER = [
    "DET", "NOUN", "VERB", "PUNCT", "SCONJ", "ADP",
    "ADJ", "CCONJ", "ADV", "PROPN", "AUX", "NUM",
    "PRON", "SYM", "X", "INTJ",
]

# ── Grupos: um por modelo base ────────────────────────────────────────────────
GROUPS = [
    {
        "key":    "mbert",
        "label":  "mBERT",
        "abbrev": "mB",
        "variants": [
            {"col": "Lin",   "stem": "predictions_results_mbert_linear",                               "has_upos": True},
            {"col": "Biaff", "stem": "predictions_results_mbert_biaffine",                             "has_upos": True},
            {"col": "sLin",  "stem": "bert-base-multilingual-cased_predictions_ablacao_linear",        "has_upos": False},
            {"col": "sBiaff","stem": "bert-base-multilingual-cased_predictions_ablacao_biaffine",       "has_upos": False},
        ],
    },
    {
        "key":    "btb",
        "label":  "BERTimbau Base",
        "abbrev": "BtB",
        "variants": [
            {"col": "Lin",   "stem": "predictions_results_bertimbau_base_linear",                      "has_upos": True},
            {"col": "Biaff", "stem": "predictions_results_bertimbau_base_biaffine",                    "has_upos": True},
            {"col": "sLin",  "stem": "bert-base-portuguese-cased_predictions_ablacao_linear",          "has_upos": False},
            {"col": "sBiaff","stem": "bert-base-portuguese-cased_predictions_ablacao_biaffine",        "has_upos": False},
        ],
    },
    {
        "key":    "btl",
        "label":  "BERTimbau Large",
        "abbrev": "BtL",
        "note":   "Biaff treinado por 45 épocas; demais por 41.",
        "variants": [
            {"col": "Lin",   "stem": "predictions_results_bertimbau_large_linear",                     "has_upos": True},
            {"col": "Biaff", "stem": "predictions_results_bertimbau_large_biaffine",                   "has_upos": True},
            {"col": "sLin",  "stem": "bert-large-portuguese-cased_predictions_ablacao_linear",         "has_upos": False},
            {"col": "sBiaff","stem": "bert-large-portuguese-cased_predictions_ablacao_biaffine",       "has_upos": False},
        ],
    },
    {
        "key":    "jab",
        "label":  "JabuticaBERT",
        "abbrev": "JaB",
        "variants": [
            {"col": "Lin",   "stem": "predictions_results_jabuticabert_linear",                        "has_upos": True},
            {"col": "Biaff", "stem": "predictions_results_jabuticabert_biaffine",                      "has_upos": True},
            {"col": "sLin",  "stem": "modernJabuticaBERT-Base-1k_predictions_ablacao_linear",          "has_upos": False},
            {"col": "sBiaff","stem": "modernJabuticaBERT-Base-1k_predictions_ablacao_biaffine",        "has_upos": False},
        ],
    },
]


# ── I/O ───────────────────────────────────────────────────────────────────────
def _load_instab(stem, task):
    """Lê <stem>_<task>.csv → dict[label -> {n, forg, acc, drop}]"""
    path = os.path.join(INST_DIR, f"{stem}_{task}.csv")
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["label"]] = {
                "n":    int(r["n_tokens"]),
                "forg": int(r["forgetting_events"]),
                "acc":  float(r["final_acc"]),
                "drop": float(r["drop"]),
            }
    return rows


def _load_head(stem):
    """Lê <stem>_head.csv → dict[epoch -> uas]"""
    path = os.path.join(INST_DIR, f"{stem}_head.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {int(r["epoch"]): float(r["uas"]) for r in csv.DictReader(f)}


# ── LaTeX ─────────────────────────────────────────────────────────────────────
def _esc(s):
    return s.replace("_", r"\_").replace(":", r"\!:\!").replace("&", r"\&")


def _instab_cmp_body(label_data, col_names, label_type, top_n=20, min_n=10):
    """
    Tabular com cabeçalho duplo: por variante (col_names) × por métrica (Fg + Ac.F).
    Fg. máximo por linha em negrito; Ac.F máxima por linha em negrito.
    """
    nc       = len(col_names)
    # col_spec: l r | r r | r r | ...  (N separado dos pares de métricas)
    col_spec = "l r" + "| r r" * nc

    # Cabeçalho linha 1: multicolumn por variante
    mc = []
    for i, c in enumerate(col_names):
        sep = "|" if i < nc - 1 else ""
        mc.append(rf"\multicolumn{{2}}{{c{sep}}}{{\textbf{{{c}}}}}")
    h1 = r"\multicolumn{2}{c}{} & " + " & ".join(mc) + r" \\"

    # Cabeçalho linha 2: Fg / Ac.F por variante
    h2 = (
        rf"\textbf{{{label_type}}} & \textbf{{N}}"
        + (r" & \textit{Fg.} & \textit{Ac.F}" * nc)
        + r" \\"
    )

    # cmidrules (começam na col 3, pois col 1=label, col 2=N)
    cmid = "".join(
        rf"\cmidrule(lr){{{3 + i*2}-{4 + i*2}}}" for i in range(nc)
    )

    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        h1,
        h2,
        cmid,
    ]

    shown = 0
    for row in label_data:
        if row["n"] < min_n:
            continue
        if shown >= top_n:
            break
        shown += 1

        forg_vals = [row.get(f"forg_{i}") for i in range(nc)]
        acc_vals  = [row.get(f"acc_{i}")  for i in range(nc)]

        valid_forg = [v for v in forg_vals if v is not None]
        valid_acc  = [v for v in acc_vals  if v is not None]
        max_forg   = max(valid_forg) if valid_forg else None
        max_acc    = max(valid_acc)  if valid_acc  else None

        cells = []
        for i in range(nc):
            fg = forg_vals[i]
            ac = acc_vals[i]
            if fg is None:
                cells += ["---", "---"]
            else:
                fg_s = (rf"\textbf{{{fg:,}}}"
                        if max_forg is not None and fg == max_forg
                        else f"{fg:,}")
                ac_s = (rf"\textbf{{{ac:.4f}}}"
                        if max_acc is not None and abs(ac - max_acc) < 1e-8
                        else f"{ac:.4f}")
                cells += [fg_s, ac_s]

        lines.append(
            rf"\texttt{{{_esc(row['label'])}}} & {row['n']:,} & "
            + " & ".join(cells)
            + r" \\"
        )

    # Linha global (Ac.F global de cada variante)
    global_cells = []
    total_n      = 0
    for i in range(nc):
        tot_ok  = sum(round(r.get(f"acc_{i}", 0) * r["n"])
                      for r in label_data if r.get(f"acc_{i}") is not None and r["n"] >= min_n)
        tot_tok = sum(r["n"] for r in label_data
                      if r.get(f"acc_{i}") is not None and r["n"] >= min_n)
        tot_fg  = sum(r.get(f"forg_{i}", 0) for r in label_data
                      if r.get(f"forg_{i}") is not None and r["n"] >= min_n)
        if i == 0:
            total_n = tot_tok
        if tot_tok:
            global_cells += [f"{tot_fg:,}", f"{tot_ok/tot_tok:.4f}"]
        else:
            global_cells += ["---", "---"]

    lines += [
        r"\midrule",
        rf"\textbf{{Global}} & \textbf{{{total_n:,}}} & "
        + " & ".join(global_cells) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def _head_cmp_body(head_dicts, col_names):
    """
    Tabela HEAD UAS compacta: épocas selecionadas × variantes.
    head_dicts: list[dict[epoch->uas]] (uma por variante, na ordem de col_names)
    """
    all_epochs = sorted(set(ep for hd in head_dicts for ep in hd))
    max_ep     = max(all_epochs) if all_epochs else 0

    # Épocas de referência: a cada 5 + última de cada variante
    ref_epochs = sorted(set(
        list(range(1, max_ep + 1, 5))
        + [max(hd) for hd in head_dicts if hd]
    ))

    nc       = len(col_names)
    col_spec = "r " + " r" * nc
    header   = (
        r"\textbf{Ép.}"
        + "".join(rf" & \textbf{{{c}}}" for c in col_names)
        + r" \\"
    )

    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for ep in ref_epochs:
        cells = []
        uas_vals = [hd.get(ep) for hd in head_dicts]
        valid    = [v for v in uas_vals if v is not None]
        best     = max(valid) if valid else None
        for v in uas_vals:
            if v is None:
                cells.append("---")
            elif best is not None and abs(v - best) < 1e-8:
                cells.append(rf"\textbf{{{v:.4f}}}")
            else:
                cells.append(f"{v:.4f}")
        lines.append(rf"{ep} & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _build_label_rows(per_variant_data, label_order, sort_by="forg"):
    """
    Constrói lista de rows {label, n, forg_i, acc_i, ...}
    ordenada por média de forgetting decrescente (mais instáveis primeiro).
    """
    nc = len(per_variant_data)
    all_labels = set()
    for d in per_variant_data:
        all_labels.update(d.keys())
    # Mantém a ordem canônica para labels presentes
    ordered = [l for l in label_order if l in all_labels]

    rows = []
    for lbl in ordered:
        forgs = [per_variant_data[i].get(lbl, {}).get("forg") for i in range(nc)]
        accs  = [per_variant_data[i].get(lbl, {}).get("acc")  for i in range(nc)]
        n_lbl = next((per_variant_data[i][lbl]["n"] for i in range(nc)
                      if lbl in per_variant_data[i]), 0)
        row = {"label": lbl, "n": n_lbl}
        for i in range(nc):
            row[f"forg_{i}"] = forgs[i]
            row[f"acc_{i}"]  = accs[i]
        rows.append(row)

    valid_forg = lambda r: [r[f"forg_{i}"] for i in range(nc)
                             if r[f"forg_{i}"] is not None]
    rows.sort(key=lambda r: -(sum(valid_forg(r)) / len(valid_forg(r))
                               if valid_forg(r) else 0))
    return rows


def _wrap_table(grp, dep_body, upos_body, head_body):
    """Monta o float table completo com os três blocos."""
    note = grp.get("note", "")
    note_tex = (rf" \textit{{Nota: {note}}}" if note else "")

    cap = (
        f"Instabilidade por rótulo --- {grp['label']} ({grp['abbrev']}). "
        r"\textbf{Lin}\,=\,Linear cU, "
        r"\textbf{Biaff}\,=\,Biaffine cU, "
        r"\textbf{sLin}\,=\,sem UPOS Linear, "
        r"\textbf{sBiaff}\,=\,sem UPOS Biaffine. "
        r"Fg.\,=\,forgetting events (acerto\,$\to$\,erro). "
        r"Ac.F\,=\,acurácia final. "
        r"Fg. máximo e Ac.F máxima por linha em \textbf{negrito}."
        + note_tex
    )
    ref = f"tab:instab-{grp['key']}"

    lines = [
        r"\begin{table}[ht]",
        r"\centering\tiny",
        r"\setlength{\tabcolsep}{2pt}",
        rf"\caption{{{cap}}}",
        rf"\label{{{ref}}}",
        # DEPREL + UPOS lado a lado
        r"\begin{minipage}[t]{0.60\linewidth}",
        r"\centering",
        dep_body,
        r"\end{minipage}",
        r"\hfill",
        r"\begin{minipage}[t]{0.37\linewidth}",
        r"\centering",
        upos_body if upos_body else r"\textit{(sem UPOS)}",
        r"\end{minipage}",
        # HEAD UAS abaixo — \par\vspace inválido dentro de table; usar \\[6pt]
        r"\\[6pt]",
        r"\begin{minipage}[t]{\linewidth}",
        r"\centering\scriptsize",
        r"\textbf{UAS por época (HEAD)} \\[3pt]",
        head_body,
        r"\end{minipage}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


# ── Loop principal ────────────────────────────────────────────────────────────
all_tables = [
    "% Tabelas de instabilidade comparativas por modelo base\n"
    "% Gerado por latex_instability_comparison.py\n"
    "% Lin=Linear cU  Biaff=Biaffine cU  sLin=sem UPOS Linear  sBiaff=sem UPOS Biaffine\n"
    "% mB=mBERT  BtB=BERTimbau Base  BtL=BERTimbau Large  JaB=JabuticaBERT\n"
]

for grp in GROUPS:
    print(f"{'─'*60}")
    print(f"Grupo : {grp['label']} ({grp['abbrev']})")

    variants  = grp["variants"]
    col_names = [v["col"] for v in variants]

    # ── Carrega dados ──────────────────────────────────────────────────────────
    dep_data  = [_load_instab(v["stem"], "deprel") for v in variants]
    upos_data = [_load_instab(v["stem"], "upos")   if v["has_upos"] else {}
                 for v in variants]
    head_data = [_load_head(v["stem"]) for v in variants]

    for i, v in enumerate(variants):
        n_dep  = len(dep_data[i])
        n_upos = len(upos_data[i])
        n_ep   = max(head_data[i]) if head_data[i] else 0
        print(f"  {v['col']:<8}: dep={n_dep} lbl  upos={n_upos} lbl  head={n_ep} ép.")

    # ── Monta rows DEPREL ──────────────────────────────────────────────────────
    dep_rows  = _build_label_rows(dep_data, DEPREL_ORDER)
    dep_body  = _instab_cmp_body(dep_rows, col_names, "DEPREL", top_n=20)

    # ── Monta rows UPOS (só variantes com UPOS: Lin + Biaff) ──────────────────
    upos_body = None
    upos_variants_idx = [i for i, v in enumerate(variants) if v["has_upos"]]
    if upos_variants_idx:
        upos_data_filtered = [upos_data[i] for i in upos_variants_idx]
        upos_col_names     = [col_names[i] for i in upos_variants_idx]
        upos_rows = _build_label_rows(upos_data_filtered, UPOS_ORDER)
        # Reindexar as colunas 0..len-1 para os dados filtrados
        for row in upos_rows:
            for new_i, old_i in enumerate(upos_variants_idx):
                row[f"forg_{new_i}"] = row.pop(f"forg_{old_i}", None)
                row[f"acc_{new_i}"]  = row.pop(f"acc_{old_i}",  None)
            # remove cols além do necessário
            for j in range(len(upos_variants_idx), len(variants)):
                row.pop(f"forg_{j}", None)
                row.pop(f"acc_{j}",  None)
        upos_body = _instab_cmp_body(upos_rows, upos_col_names, "UPOS", top_n=20)

    # ── Monta HEAD ─────────────────────────────────────────────────────────────
    head_body = _head_cmp_body(head_data, col_names)

    # ── Monta tabela ───────────────────────────────────────────────────────────
    tex = _wrap_table(grp, dep_body, upos_body, head_body)
    all_tables.append(tex)

    # Resumo terminal
    print("  Global DEPREL Ac.F:", end="")
    for i, v in enumerate(variants):
        rows_i = [r for r in dep_rows if r.get(f"acc_{i}") is not None and r["n"] >= 10]
        if rows_i:
            ok  = sum(round(r[f"acc_{i}"] * r["n"]) for r in rows_i)
            tok = sum(r["n"] for r in rows_i)
            print(f"  {v['col']}={ok/tok:.4f}", end="")
    print()
    if head_data[0]:
        final_uas = {v["col"]: head_data[i].get(max(head_data[i]), 0)
                     for i, v in enumerate(variants) if head_data[i]}
        print("  UAS final:", "  ".join(f"{c}={u:.4f}" for c, u in final_uas.items()))

# ── Salva ─────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "00_tabelas_instabilidade_comparacao.tex")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(all_tables))

print(f"\n{'='*60}")
print(f"Salvo: latex_tables/00_tabelas_instabilidade_comparacao.tex")
print(r"\input{latex_tables/00_tabelas_instabilidade_comparacao}")
