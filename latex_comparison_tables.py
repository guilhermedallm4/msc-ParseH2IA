"""
Gera tabelas LaTeX de comparação de acurácia por rótulo DEPREL e UPOS.

Agrupamento: 4 tabelas, uma por modelo base.
Cada tabela compara as 4 variantes do mesmo encoder:
  Lin    — com UPOS, cabeçote linear
  Biaff  — com UPOS, cabeçote biaffine
  sLin   — sem UPOS (ablação), cabeçote linear
  sBiaff — sem UPOS (ablação), cabeçote biaffine

Layout por tabela:
  • DEPREL: 4 colunas (Lin | Biaff | sLin | sBiaff)  — minipage esquerda
  • UPOS:   2 colunas (Lin | Biaff)                   — minipage direita
  • Melhor por linha em \\textbf{negrito}

Saída: latex_tables/00_tabelas_comparacao.tex
"""

import ast
import collections
import csv
import os

csv.field_size_limit(10_000_000)

BASE     = os.path.dirname(os.path.abspath(__file__))
TEST_CSV = os.path.join(BASE, "data_dois", "test_outxpos.csv")
OUT_DIR  = os.path.join(BASE, "latex_tables")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Rótulos ───────────────────────────────────────────────────────────────────
DEPREL_LABELS = [
    "det", "nsubj", "root", "obj", "xcomp", "punct", "mark", "advcl", "case",
    "obl", "amod", "conj", "cc", "nmod", "advmod", "flat:name", "ccomp", "cop",
    "acl", "nummod", "acl:relcl", "ccomp:speech", "parataxis", "csubj",
    "aux:pass", "appos", "fixed", "nsubj:pass", "aux", "nsubj:outer",
    "obl:agent", "expl:impers", "expl", "discourse", "orphan", "dislocated",
    "flat", "flat:foreign", "iobj", "vocative", "csubj:outer", "list",
    "reparandum", "csubj:pass",
]
UPOS_LABELS = [
    "DET", "NOUN", "VERB", "PUNCT", "SCONJ", "ADP",
    "ADJ", "CCONJ", "ADV", "PROPN", "AUX", "NUM",
    "PRON", "SYM", "X", "INTJ",
]
deprel2idx = {l: i for i, l in enumerate(DEPREL_LABELS)}
upos2idx   = {l: i for i, l in enumerate(UPOS_LABELS)}

# ── Grupos: um por modelo base, 4 variantes cada ─────────────────────────────
#   Ordem das variantes: Lin, Biaff, sLin, sBiaff
GROUPS = [
    {
        "key":   "mbert",
        "label": "mBERT",
        "abbrev_full": "mB",
        "variants": [
            {"stem": "predict_test_df_mbert_linear",
             "col": "Lin",   "has_upos": True},
            {"stem": "predict_test_df_mbert_biaffine",
             "col": "Biaff", "has_upos": True},
            {"stem": "predict_test_ablacao_linear_google-bert_bert-base-multilingual-cased",
             "col": "sLin",  "has_upos": False},
            {"stem": "predict_test_ablacao_biaffine_google-bert_bert-base-multilingual-cased_fold0",
             "col": "sBiaff","has_upos": False},
        ],
    },
    {
        "key":   "btb",
        "label": "BERTimbau Base",
        "abbrev_full": "BtB",
        "variants": [
            {"stem": "predict_test_df_bertimbau_base_linear",
             "col": "Lin",   "has_upos": True},
            {"stem": "predict_test_df_bertimbau_base_biaffine",
             "col": "Biaff", "has_upos": True},
            {"stem": "predict_test_ablacao_linear_neuralmind_bert-base-portuguese-cased",
             "col": "sLin",  "has_upos": False},
            {"stem": "predict_test_ablacao_biaffine_neuralmind_bert-base-portuguese-cased_fold0",
             "col": "sBiaff","has_upos": False},
        ],
    },
    {
        "key":   "btl",
        "label": "BERTimbau Large",
        "abbrev_full": "BtL",
        "variants": [
            {"stem": "predict_test_df_bertimbau_large_linear",
             "col": "Lin",   "has_upos": True},
            {"stem": "predict_test_df_bertimbau_large_biaffine",
             "col": "Biaff", "has_upos": True},
            {"stem": "predict_test_ablacao_linear_neuralmind_bert-large-portuguese-cased",
             "col": "sLin",  "has_upos": False},
            {"stem": "predict_test_ablacao_biaffine_neuralmind_bert-large-portuguese-cased_fold0",
             "col": "sBiaff","has_upos": False},
        ],
    },
    {
        "key":   "jab",
        "label": "JabuticaBERT",
        "abbrev_full": "JaB",
        "variants": [
            {"stem": "predict_test_df_jabuticabert_linear",
             "col": "Lin",   "has_upos": True},
            {"stem": "predict_test_df_jabuticabert_biaffine",
             "col": "Biaff", "has_upos": True},
            {"stem": "predict_test_ablacao_linear_amadeusai_modernJabuticaBERT-Base-1k",
             "col": "sLin",  "has_upos": False},
            {"stem": "predict_test_ablacao_biaffine_amadeusai_modernJabuticaBERT-Base-1k",
             "col": "sBiaff","has_upos": False},
        ],
    },
]


# ── I/O ───────────────────────────────────────────────────────────────────────
def _parse(raw):
    try:
        return ast.literal_eval(raw)
    except Exception:
        return []


def _load_gold():
    g_dep, g_upo = [], []
    with open(TEST_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g_dep.extend(ast.literal_eval(row["deprel_tags"]))
            g_upo.extend(ast.literal_eval(row["upos_tags"]))
    return g_dep, g_upo


def _load_preds(stem):
    path = os.path.join(BASE, f"{stem}.csv")
    p_dep, p_upo = [], []
    has_upos = False
    with open(path, newline="", encoding="utf-8") as f:
        reader   = csv.DictReader(f)
        fnames   = reader.fieldnames or []
        has_upos = "upos_predictions" in fnames
        for row in reader:
            dep = _parse(row.get("deprel_predictions", "[]"))
            p_dep.extend(deprel2idx.get(d, -1) for d in dep)
            if has_upos:
                upos = _parse(row.get("upos_predictions", "[]"))
                if upos:
                    p_upo.extend(upos2idx.get(u, -1) for u in upos)
                else:
                    p_upo.extend([-1] * len(dep))
    if has_upos and all(x == -1 for x in p_upo):
        has_upos = False
    return p_dep, (p_upo if has_upos else []), has_upos


def _per_label_acc(gold, pred, labels, lbl2idx, min_count=5):
    n       = min(len(gold), len(pred))
    totals  = collections.Counter(gold[:n])
    correct = collections.Counter(g for g, p in zip(gold[:n], pred[:n]) if g == p)
    out = {}
    for lbl in labels:
        idx = lbl2idx[lbl]
        cnt = totals.get(idx, 0)
        if cnt >= min_count:
            out[lbl] = {"n": cnt, "acc": correct.get(idx, 0) / cnt}
    return out


# ── LaTeX ─────────────────────────────────────────────────────────────────────
def _esc(s):
    return s.replace("_", r"\_").replace(":", r"\!:\!").replace("&", r"\&")


def _cmp_body(label_data, col_names, label_type="DEPREL"):
    """
    label_data: list of dicts {label, n, acc_0, acc_1, ...}
    col_names:  list of column header strings
    """
    nc       = len(col_names)
    col_spec = "l r " + " r" * nc
    header   = (
        rf"\textbf{{{label_type}}} & \textbf{{N}}"
        + "".join(rf" & \textbf{{{c}}}" for c in col_names)
        + r" \\"
    )
    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    tot_ok  = [0] * nc
    tot_tok = [0] * nc

    for row in label_data:
        accs  = [row.get(f"acc_{i}") for i in range(nc)]
        valid = [a for a in accs if a is not None]
        best  = max(valid) if valid else None
        cells = []
        for a in accs:
            if a is None:
                cells.append("---")
            elif best is not None and abs(a - best) < 1e-8:
                cells.append(rf"\textbf{{{a:.4f}}}")
            else:
                cells.append(f"{a:.4f}")
        lines.append(
            rf"\texttt{{{_esc(row['label'])}}} & {row['n']:,} & "
            + " & ".join(cells) + r" \\"
        )
        for i in range(nc):
            a = row.get(f"acc_{i}")
            if a is not None:
                tot_ok[i]  += round(a * row["n"])
                tot_tok[i] += row["n"]

    g_accs = [
        (tot_ok[i] / tot_tok[i] if tot_tok[i] else 0.0) for i in range(nc)
    ]
    valid_g = [a for a in g_accs if a > 0]
    best_g  = max(valid_g) if valid_g else None
    g_cells = []
    for a in g_accs:
        if best_g is not None and abs(a - best_g) < 1e-8:
            g_cells.append(rf"\textbf{{{a:.4f}}}")
        else:
            g_cells.append(f"{a:.4f}" if a > 0 else "---")

    n_total = tot_tok[0] if tot_tok[0] else sum(r["n"] for r in label_data)
    lines += [
        r"\midrule",
        rf"\textbf{{Global}} & \textbf{{{n_total:,}}} & "
        + " & ".join(g_cells) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def _wrap_table(dep_body, upos_body, caption, ref_label):
    """DEPREL (esq.) + UPOS (dir.) em uma tabela com minipages."""
    if upos_body:
        return "\n".join([
            r"\begin{table}[ht]",
            r"\centering\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            rf"\caption{{{caption}}}",
            rf"\label{{{ref_label}}}",
            r"\begin{minipage}[t]{0.60\linewidth}",
            r"\centering",
            dep_body,
            r"\end{minipage}",
            r"\hfill",
            r"\begin{minipage}[t]{0.37\linewidth}",
            r"\centering",
            upos_body,
            r"\end{minipage}",
            r"\end{table}",
            "",
        ])
    return "\n".join([
        r"\begin{table}[ht]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{{ref_label}}}",
        dep_body,
        r"\end{table}",
        "",
    ])


# ── Pipeline principal ────────────────────────────────────────────────────────
print("Carregando gabarito...")
gold_dep, gold_upo = _load_gold()
print(f"  {len(gold_dep):,} tokens DEPREL | {len(gold_upo):,} UPOS\n")

all_tables = [
    "% Tabelas comparativas por modelo base — geradas por latex_comparison_tables.py\n"
    "% Lin=Linear cU  Biaff=Biaffine cU  sLin=sem UPOS Linear  sBiaff=sem UPOS Biaffine\n"
    "% mB=mBERT  BtB=BERTimbau Base  BtL=BERTimbau Large  JaB=JabuticaBERT\n"
    "% \\textbf{Negrito} = melhor resultado por rótulo.\n"
]

for grp in GROUPS:
    print(f"{'─'*60}")
    print(f"Grupo : {grp['label']} ({grp['abbrev_full']})")

    variants   = grp["variants"]
    col_names  = [v["col"] for v in variants]        # Lin, Biaff, sLin, sBiaff
    upos_cols  = [v["col"] for v in variants if v["has_upos"]]  # Lin, Biaff

    # ── Carrega predições ──────────────────────────────────────────────────────
    preds_dep = []
    preds_upo = []
    for v in variants:
        pd, pu, hu = _load_preds(v["stem"])
        preds_dep.append(pd)
        preds_upo.append(pu if (hu and v["has_upos"]) else None)
        print(f"  {v['col']:<8}: {v['stem']}")

    # ── Per-label acc — DEPREL (todas as 4 variantes) ────────────────────────
    per_v_dep = [
        _per_label_acc(gold_dep, p, DEPREL_LABELS, deprel2idx)
        for p in preds_dep
    ]
    common_dep = [
        lbl for lbl in DEPREL_LABELS
        if any(lbl in pm for pm in per_v_dep)
    ]
    dep_rows = []
    for lbl in common_dep:
        n_lbl = next((pm[lbl]["n"] for pm in per_v_dep if lbl in pm), 0)
        if n_lbl < 5:
            continue
        row = {"label": lbl, "n": n_lbl}
        for i, pm in enumerate(per_v_dep):
            row[f"acc_{i}"] = pm[lbl]["acc"] if lbl in pm else None
        dep_rows.append(row)

    dep_body = _cmp_body(dep_rows, col_names, "DEPREL")

    # ── Per-label acc — UPOS (apenas variantes com UPOS: Lin e Biaff) ─────────
    upos_body = None
    upo_preds_valid = [(i, p) for i, (v, p) in enumerate(zip(variants, preds_upo))
                       if v["has_upos"] and p is not None]
    if upo_preds_valid:
        per_v_upo = [
            _per_label_acc(gold_upo, p, UPOS_LABELS, upos2idx)
            for _, p in upo_preds_valid
        ]
        upo_col_names = [col_names[i] for i, _ in upo_preds_valid]
        common_upo = [
            lbl for lbl in UPOS_LABELS
            if any(lbl in pm for pm in per_v_upo)
        ]
        upo_rows = []
        for lbl in common_upo:
            n_lbl = next((pm[lbl]["n"] for pm in per_v_upo if lbl in pm), 0)
            if n_lbl < 5:
                continue
            row = {"label": lbl, "n": n_lbl}
            for i, pm in enumerate(per_v_upo):
                row[f"acc_{i}"] = pm[lbl]["acc"] if lbl in pm else None
            upo_rows.append(row)
        upos_body = _cmp_body(upo_rows, upo_col_names, "UPOS")

    # ── Monta tabela LaTeX ─────────────────────────────────────────────────────
    af    = grp["abbrev_full"]
    cap   = (
        f"Acurácia por rótulo --- {grp['label']} ({af}). "
        r"\textbf{Lin}\,=\,Linear com UPOS, "
        r"\textbf{Biaff}\,=\,Biaffine com UPOS, "
        r"\textbf{sLin}\,=\,sem UPOS Linear, "
        r"\textbf{sBiaff}\,=\,sem UPOS Biaffine. "
        r"\textbf{Negrito}\,=\,melhor por rótulo."
    )
    tex = _wrap_table(dep_body, upos_body, cap, f"tab:acc-{grp['key']}")
    all_tables.append(tex)

    # Imprime resumo global
    print("  Global DEPREL Acc:", end="")
    for i, v in enumerate(variants):
        ok  = sum(round(r[f"acc_{i}"] * r["n"]) for r in dep_rows if r.get(f"acc_{i}") is not None)
        tot = sum(r["n"] for r in dep_rows if r.get(f"acc_{i}") is not None)
        print(f"  {v['col']}={ok/tot:.4f}" if tot else f"  {v['col']}=---", end="")
    print()

# ── Salva ─────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "00_tabelas_comparacao.tex")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(all_tables))

print(f"\n{'='*60}")
print(f"Salvo: latex_tables/00_tabelas_comparacao.tex")
print(f"Tabelas: {len(GROUPS)} (uma por modelo base)")
print(r"\input{latex_tables/00_tabelas_comparacao}")
