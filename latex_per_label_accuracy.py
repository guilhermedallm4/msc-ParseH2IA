"""
Gera tabelas LaTeX de acurácia por rótulo DEPREL e UPOS para cada modelo.

Layout por arquivo .tex:
  - Uma tabela com DEPREL (esquerda) e UPOS (direita) lado a lado via minipage.
  - Modelos sem UPOS: apenas DEPREL (largura total).

Saída adicional:
  - latex_tables/00_resumo_acuracia.tex — tabela-resumo com todos os modelos.

Abreviações dos modelos:
  mB=mBERT  BtB=BERTimbau Base  BtL=BERTimbau Large  JaB=JabuticaBERT
  Lin=Linear  Biaff=Biaffine  cU=com UPOS  sU=sem UPOS  F0=Fold 0
"""

import ast
import collections
import csv
import glob
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

# ── Metadados por stem ────────────────────────────────────────────────────────
MODEL_META: dict[str, dict] = {
    "predict_test_df_bertimbau_base_biaffine":
        {"display": "BERTimbau Base — Biaffine (com UPOS)",   "abbrev": "BtB-Biaff-cU"},
    "predict_test_df_bertimbau_base_linear":
        {"display": "BERTimbau Base — Linear (com UPOS)",     "abbrev": "BtB-Lin-cU"},
    "predict_test_df_bertimbau_large_biaffine":
        {"display": "BERTimbau Large — Biaffine (com UPOS)",  "abbrev": "BtL-Biaff-cU"},
    "predict_test_df_bertimbau_large_linear":
        {"display": "BERTimbau Large — Linear (com UPOS)",    "abbrev": "BtL-Lin-cU"},
    "predict_test_df_jabuticabert_biaffine":
        {"display": "JabuticaBERT — Biaffine (com UPOS)",     "abbrev": "JaB-Biaff-cU"},
    "predict_test_df_jabuticabert_linear":
        {"display": "JabuticaBERT — Linear (com UPOS)",       "abbrev": "JaB-Lin-cU"},
    "predict_test_df_mbert_biaffine":
        {"display": "mBERT — Biaffine (com UPOS)",            "abbrev": "mB-Biaff-cU"},
    "predict_test_df_mbert_linear":
        {"display": "mBERT — Linear (com UPOS)",              "abbrev": "mB-Lin-cU"},
    "predict_test_ablacao_biaffine_neuralmind_bert-large-portuguese-cased_fold0":
        {"display": "BERTimbau Large — Biaffine (sem UPOS)",  "abbrev": "BtL-Biaff-sU"},
    "predict_test_ablacao_biaffine_neuralmind_bert-base-portuguese-cased_fold0":
        {"display": "BERTimbau Base — Biaffine (sem UPOS)",   "abbrev": "BtB-Biaff-sU"},
    "predict_test_ablacao_biaffine_google-bert_bert-base-multilingual-cased_fold0":
        {"display": "mBERT — Biaffine (sem UPOS)",            "abbrev": "mB-Biaff-sU"},
    "predict_test_ablacao_biaffine_amadeusai_modernJabuticaBERT-Base-1k":
        {"display": "JabuticaBERT — Biaffine (sem UPOS)",     "abbrev": "JaB-Biaff-sU"},
    "predict_test_ablacao_biaffine_amadeusai_modernJabuticaBERT-Base-1k_fold0":
        {"display": "JabuticaBERT — Biaffine (sem UPOS, F0)", "abbrev": "JaB-Biaff-sU-F0"},
    "predict_test_ablacao_linear_neuralmind_bert-large-portuguese-cased":
        {"display": "BERTimbau Large — Linear (sem UPOS)",    "abbrev": "BtL-Lin-sU"},
    "predict_test_ablacao_linear_neuralmind_bert-large-portuguese-cased_fold0":
        {"display": "BERTimbau Large — Linear (sem UPOS, F0)","abbrev": "BtL-Lin-sU-F0"},
    "predict_test_ablacao_linear_neuralmind_bert-base-portuguese-cased":
        {"display": "BERTimbau Base — Linear (sem UPOS)",     "abbrev": "BtB-Lin-sU"},
    "predict_test_ablacao_linear_neuralmind_bert-base-portuguese-cased_fold0":
        {"display": "BERTimbau Base — Linear (sem UPOS, F0)", "abbrev": "BtB-Lin-sU-F0"},
    "predict_test_ablacao_linear_google-bert_bert-base-multilingual-cased":
        {"display": "mBERT — Linear (sem UPOS)",              "abbrev": "mB-Lin-sU"},
    "predict_test_ablacao_linear_google-bert_bert-base-multilingual-cased_fold0":
        {"display": "mBERT — Linear (sem UPOS, F0)",          "abbrev": "mB-Lin-sU-F0"},
    "predict_test_ablacao_linear_amadeusai_modernJabuticaBERT-Base-1k":
        {"display": "JabuticaBERT — Linear (sem UPOS)",       "abbrev": "JaB-Lin-sU"},
    "predict_test_ablacao_linear_amadeusai_modernJabuticaBERT-Base-1k_fold0":
        {"display": "JabuticaBERT — Linear (sem UPOS, F0)",   "abbrev": "JaB-Lin-sU-F0"},
}

# ── Gabarito ──────────────────────────────────────────────────────────────────
print("Carregando gabarito...")
gold_deprel: list[int] = []
gold_upos:   list[int] = []
with open(TEST_CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        gold_deprel.extend(ast.literal_eval(row["deprel_tags"]))
        gold_upos.extend(  ast.literal_eval(row["upos_tags"]))
print(f"  Tokens: {len(gold_deprel):,}")


# ── Funções auxiliares ────────────────────────────────────────────────────────
def _parse(raw: str) -> list[str]:
    try:
        return ast.literal_eval(raw)
    except Exception:
        return []


def _per_label_acc(gold, pred, labels, lbl2idx) -> list[dict]:
    totals  = collections.Counter(gold)
    correct = collections.Counter(g for g, p in zip(gold, pred) if g == p)
    result  = []
    for lbl in labels:
        idx = lbl2idx.get(lbl, -1)
        if idx not in totals:
            continue
        n  = totals[idx]
        ok = correct.get(idx, 0)
        result.append({"label": lbl, "total": n, "correct": ok,
                        "acc": ok / n if n else 0.0})
    return sorted(result, key=lambda r: -r["total"])


def _esc(s: str) -> str:
    return s.replace("_", r"\_").replace(":", r"\!:\!").replace("&", r"\&")


def _tabular_body(rows: list[dict], col_name: str, min_count: int = 5) -> str:
    """Retorna o conteúdo interno da tabela (sem begin/end table)."""
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        rf"\textbf{{{col_name}}} & \textbf{{Tot.}} & \textbf{{Ok}} & \textbf{{Acc.}} \\",
        r"\midrule",
    ]
    for r in rows:
        if r["total"] < min_count:
            continue
        lines.append(
            rf"\texttt{{{_esc(r['label'])}}} & {r['total']:,} & "
            rf"{r['correct']:,} & {r['acc']:.4f} \\"
        )
    tot_n  = sum(r["total"]   for r in rows if r["total"] >= min_count)
    tot_ok = sum(r["correct"] for r in rows if r["total"] >= min_count)
    tot_ac = tot_ok / tot_n if tot_n else 0.0
    lines += [
        r"\midrule",
        rf"\textbf{{Total}} & \textbf{{{tot_n:,}}} & \textbf{{{tot_ok:,}}} & \textbf{{{tot_ac:.4f}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def _wrap_side_by_side(
    left_body:  str,
    right_body: str | None,
    caption:    str,
    ref_label:  str,
    left_w:     float = 0.55,
    right_w:    float = 0.42,
) -> str:
    """Envolve um ou dois tabulares em uma tabela LaTeX com minipages."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{ref_label}}}",
    ]
    if right_body:
        lines += [
            rf"\begin{{minipage}}[t]{{{left_w}\linewidth}}",
            r"\centering\scriptsize",
            left_body,
            r"\end{minipage}",
            r"\hfill",
            rf"\begin{{minipage}}[t]{{{right_w}\linewidth}}",
            r"\centering\scriptsize",
            right_body,
            r"\end{minipage}",
        ]
    else:
        lines += [
            r"\scriptsize",
            left_body,
        ]
    lines += [r"\end{table}", ""]
    return "\n".join(lines)


def _summary_table(summary_rows: list[dict]) -> str:
    """Tabela-resumo com todos os modelos e suas acurácias globais."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering\footnotesize",
        (r"\caption{Resumo de acurácia global por modelo. "
         r"cU\,=\,com UPOS, sU\,=\,sem UPOS, F0\,=\,Fold~0. "
         r"mB\,=\,mBERT, BtB\,=\,BERTimbau Base, BtL\,=\,BERTimbau Large, "
         r"JaB\,=\,JabuticaBERT.}"),
        r"\label{tab:resumo-acuracia-geral}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\textbf{Modelo} & \textbf{DEPREL Acc.} & \textbf{UPOS Acc.} \\",
        r"\midrule",
    ]
    for r in summary_rows:
        upos_str = f"{r['upos_acc']:.4f}" if r["upos_acc"] is not None else r"---"
        lines.append(
            rf"\texttt{{{r['abbrev']}}} & {r['deprel_acc']:.4f} & {upos_str} \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


# ── Loop principal ────────────────────────────────────────────────────────────
pred_files   = sorted(glob.glob(os.path.join(BASE, "predict_test*.csv")))
summary_rows = []
print(f"\nArquivos encontrados: {len(pred_files)}\n")

for pred_path in pred_files:
    stem    = os.path.basename(pred_path).replace(".csv", "")
    meta    = MODEL_META.get(stem, {"display": stem, "abbrev": stem[:20]})
    display = meta["display"]
    abbrev  = meta["abbrev"]
    safe    = stem.replace("_", "-").replace(".", "-")
    out_tex = os.path.join(OUT_DIR, f"{stem}.tex")

    print(f"{'─'*55}")
    print(f"Arquivo : {os.path.basename(pred_path)}")
    print(f"Modelo  : {abbrev} — {display}")

    # Lê predições
    pred_deprel: list[int] = []
    pred_upos:   list[int] = []
    has_upos = False

    with open(pred_path, newline="", encoding="utf-8") as f:
        reader     = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_upos   = "upos_predictions" in fieldnames
        for row in reader:
            dep = _parse(row.get("deprel_predictions", "[]"))
            pred_deprel.extend(deprel2idx.get(d, -1) for d in dep)
            if has_upos:
                upos = _parse(row.get("upos_predictions", "[]"))
                if upos:
                    pred_upos.extend(upos2idx.get(u, -1) for u in upos)
                else:
                    pred_upos.extend([-1] * len(dep))

    if has_upos and all(x == -1 for x in pred_upos):
        has_upos = False

    n     = min(len(gold_deprel), len(pred_deprel))
    g_dep = gold_deprel[:n]
    p_dep = pred_deprel[:n]

    acc_dep = sum(g == p for g, p in zip(g_dep, p_dep)) / n
    acc_upo = None
    print(f"  DEPREL: {acc_dep:.4f}", end="")

    # Corpo da tabela DEPREL
    deprel_rows  = _per_label_acc(g_dep, p_dep, DEPREL_LABELS, deprel2idx)
    deprel_body  = _tabular_body(deprel_rows, "DEPREL")

    # Corpo da tabela UPOS (quando disponível)
    upos_body = None
    if has_upos:
        g_upo = gold_upos[:n]
        p_upo = pred_upos[:n]
        acc_upo = sum(g == p for g, p in zip(g_upo, p_upo)) / n
        upos_rows = _per_label_acc(g_upo, p_upo, UPOS_LABELS, upos2idx)
        upos_body = _tabular_body(upos_rows, "UPOS")
        print(f" | UPOS: {acc_upo:.4f}")
    else:
        print()

    # Monta e salva .tex
    caption = (
        f"Acurácia por rótulo --- {abbrev} ({display}). "
        f"DEPREL Acc.\\,=\\,{acc_dep:.4f}"
        + (f", UPOS Acc.\\,=\\,{acc_upo:.4f}." if acc_upo is not None else ".")
    )
    tex = (
        f"% Modelo: {display}\n"
        + _wrap_side_by_side(deprel_body, upos_body, caption, f"tab:acc-{safe}")
    )
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"  Salvo : latex_tables/{os.path.basename(out_tex)}")

    summary_rows.append({
        "abbrev":     abbrev,
        "display":    display,
        "deprel_acc": acc_dep,
        "upos_acc":   acc_upo,
    })

# ── Tabela-resumo ─────────────────────────────────────────────────────────────
summary_tex_path = os.path.join(OUT_DIR, "00_resumo_acuracia.tex")
with open(summary_tex_path, "w", encoding="utf-8") as f:
    f.write(
        "% Tabela-resumo gerada por latex_per_label_accuracy.py\n"
        + _summary_table(summary_rows)
    )
print(f"\n{'='*55}")
print(f"Resumo salvo: latex_tables/00_resumo_acuracia.tex")
print(f"Total de arquivos .tex: {len(summary_rows) + 1}")
print()
print("\\input{latex_tables/00_resumo_acuracia}")
print("% Individuais:")
for p in sorted(glob.glob(os.path.join(OUT_DIR, "predict_test*.tex"))):
    print(f"\\input{{latex_tables/{os.path.basename(p).replace('.tex','')}}}")
