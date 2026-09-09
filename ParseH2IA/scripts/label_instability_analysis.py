"""
Análise de instabilidade de rótulos durante o treinamento.

Layout das tabelas LaTeX por arquivo .tex:
  - DEPREL instabilidade (esquerda) + UPOS instabilidade ou HEAD UAS (direita).
  - Tabela HEAD com epoch-a-epoch separada (compacta, full-width).
  - Modelos sem UPOS: DEPREL (esquerda) + HEAD resumo (direita).

Saída adicional:
  - instability_results/00_resumo_instabilidade.tex — tabela-resumo multirrelação.

Abreviações:
  mB=mBERT  BtB=BERTimbau Base  BtL=BERTimbau Large  JaB=JabuticaBERT
  Lin=Linear  Biaff=Biaffine  cU=com UPOS  sU=sem UPOS
"""

import collections
import csv
import glob
import json
import os

BASE    = os.path.dirname(os.path.abspath(__file__))
EP_DIR  = os.path.join(BASE, "epoch_predictions")
OUT_DIR = os.path.join(BASE, "instability_results")
os.makedirs(OUT_DIR, exist_ok=True)

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
LABEL_MAP = {"deprel": DEPREL_LABELS, "upos": UPOS_LABELS}

MODEL_META: dict[str, dict] = {
    "bert-base-multilingual-cased_predictions_ablacao_biaffine":
        {"display": "mBERT — Biaffine (sem UPOS)",            "abbrev": "mB-Biaff-sU"},
    "bert-base-multilingual-cased_predictions_ablacao_linear":
        {"display": "mBERT — Linear (sem UPOS)",              "abbrev": "mB-Lin-sU"},
    "bert-base-portuguese-cased_predictions_ablacao_biaffine":
        {"display": "BERTimbau Base — Biaffine (sem UPOS)",   "abbrev": "BtB-Biaff-sU"},
    "bert-base-portuguese-cased_predictions_ablacao_linear":
        {"display": "BERTimbau Base — Linear (sem UPOS)",     "abbrev": "BtB-Lin-sU"},
    "bert-large-portuguese-cased_predictions_ablacao_biaffine":
        {"display": "BERTimbau Large — Biaffine (sem UPOS)",  "abbrev": "BtL-Biaff-sU"},
    "bert-large-portuguese-cased_predictions_ablacao_linear":
        {"display": "BERTimbau Large — Linear (sem UPOS)",    "abbrev": "BtL-Lin-sU"},
    "modernJabuticaBERT-Base-1k_predictions_ablacao_biaffine":
        {"display": "JabuticaBERT — Biaffine (sem UPOS)",     "abbrev": "JaB-Biaff-sU"},
    "modernJabuticaBERT-Base-1k_predictions_ablacao_linear":
        {"display": "JabuticaBERT — Linear (sem UPOS)",       "abbrev": "JaB-Lin-sU"},
    "predictions_results_bertimbau_base_biaffine":
        {"display": "BERTimbau Base — Biaffine (com UPOS)",   "abbrev": "BtB-Biaff-cU"},
    "predictions_results_bertimbau_base_linear":
        {"display": "BERTimbau Base — Linear (com UPOS)",     "abbrev": "BtB-Lin-cU"},
    "predictions_results_bertimbau_large_biaffine":
        {"display": "BERTimbau Large — Biaffine (com UPOS)",  "abbrev": "BtL-Biaff-cU"},
    "predictions_results_bertimbau_large_linear":
        {"display": "BERTimbau Large — Linear (com UPOS)",    "abbrev": "BtL-Lin-cU"},
    "predictions_results_jabuticabert_biaffine":
        {"display": "JabuticaBERT — Biaffine (com UPOS)",     "abbrev": "JaB-Biaff-cU"},
    "predictions_results_jabuticabert_linear":
        {"display": "JabuticaBERT — Linear (com UPOS)",       "abbrev": "JaB-Lin-cU"},
    "predictions_results_mbert_biaffine":
        {"display": "mBERT — Biaffine (com UPOS)",            "abbrev": "mB-Biaff-cU"},
    "predictions_results_mbert_linear":
        {"display": "mBERT — Linear (com UPOS)",              "abbrev": "mB-Lin-cU"},
}


# ── Funções de análise ────────────────────────────────────────────────────────
def build_histories(entries):
    token_epochs = collections.defaultdict(list)
    for e in entries:
        token_epochs[e["index"]].append(e)
    n_epochs = max(len(v) for v in token_epochs.values())
    token_history, token_label = {}, {}
    for idx, ep_list in token_epochs.items():
        token_label[idx]   = ep_list[0]["correct_label"]
        token_history[idx] = [e["pred_label"] == e["correct_label"] for e in ep_list]
    return token_history, token_label, n_epochs


def per_label_instability(token_history, token_label, n_epochs, labels):
    label_histories = collections.defaultdict(list)
    for idx, hist in token_history.items():
        label_histories[token_label[idx]].append(hist)
    rows = []
    for label_idx in sorted(label_histories.keys()):
        hists = label_histories[label_idx]
        n     = len(hists)
        label = labels[label_idx] if label_idx < len(labels) else str(label_idx)
        forgetting = recovery = oscil = learned_forgotten = 0
        for hist in hists:
            for t in range(1, len(hist)):
                if hist[t - 1] and not hist[t]:
                    forgetting += 1
                elif not hist[t - 1] and hist[t]:
                    recovery   += 1
            oscil += sum(hist[t] != hist[t - 1] for t in range(1, len(hist)))
            if any(hist) and not hist[-1]:
                learned_forgotten += 1
        epoch_accs = [sum(h[ep] for h in hists) / n for ep in range(n_epochs)]
        peak_acc   = max(epoch_accs)
        final_acc  = epoch_accs[-1]
        rows.append({
            "label": label, "label_idx": label_idx, "n_tokens": n,
            "forgetting_events": forgetting, "recovery_events": recovery,
            "oscillation": oscil, "learned_forgotten": learned_forgotten,
            "peak_acc":  round(peak_acc,  6),
            "final_acc": round(final_acc, 6),
            "drop":      round(peak_acc - final_acc, 6),
            "epoch_accs": epoch_accs,
        })
    return sorted(rows, key=lambda r: -r["forgetting_events"])


def global_head_instability(token_history, n_epochs):
    all_hists = list(token_history.values())
    n = len(all_hists)
    epoch_rows = []
    for ep in range(n_epochs):
        correct    = sum(h[ep] for h in all_hists)
        forgetting = sum(1 for h in all_hists if ep > 0 and     h[ep - 1] and not h[ep])
        recovery   = sum(1 for h in all_hists if ep > 0 and not h[ep - 1] and     h[ep])
        epoch_rows.append({
            "epoch": ep + 1, "uas": round(correct / n, 6), "correct": correct,
            "total": n, "forgetting": forgetting, "recovery": recovery,
            "net_change": recovery - forgetting,
        })
    return epoch_rows


# ── Funções LaTeX ─────────────────────────────────────────────────────────────
def _esc(s):
    return s.replace("_", r"\_").replace(":", r"\!:\!").replace("&", r"\&")


def _instab_body(rows, top_n=15, min_count=5):
    """Corpo interno do tabular de instabilidade (sem begin/end table)."""
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        (r"\textbf{Rótulo} & \textbf{N} & \textbf{Forg.} & \textbf{Rec.}"
         r" & \textbf{Osc.} & \textbf{A+E} & \textbf{Pico}"
         r" & \textbf{Final} & \textbf{Drop} \\"),
        r"\midrule",
    ]
    shown = 0
    for r in rows:
        if r["n_tokens"] < min_count:
            continue
        if shown >= top_n:
            break
        lines.append(
            rf"\texttt{{{_esc(r['label'])}}} & {r['n_tokens']:,} & "
            rf"{r['forgetting_events']:,} & {r['recovery_events']:,} & "
            rf"{r['oscillation']:,} & {r['learned_forgotten']:,} & "
            rf"{r['peak_acc']:.4f} & {r['final_acc']:.4f} & {r['drop']:.4f} \\"
        )
        shown += 1
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _head_body(epoch_rows):
    """Corpo interno do tabular HEAD UAS (resumo a cada 5 épocas)."""
    n_ep     = len(epoch_rows)
    selected = sorted(set(list(range(0, n_ep, 5)) + [n_ep - 1]))
    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        (r"\textbf{Ép.} & \textbf{UAS} & \textbf{Corr.}"
         r" & \textbf{Forg.} & \textbf{Rec.} & \textbf{$\Delta$} \\"),
        r"\midrule",
    ]
    for i in selected:
        r = epoch_rows[i]
        lines.append(
            rf"{r['epoch']} & {r['uas']:.4f} & {r['correct']:,} & "
            rf"{r['forgetting']:,} & {r['recovery']:,} & "
            rf"{r['net_change']:+,} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _wrap_side_by_side(left_body, right_body, caption, ref_label,
                       left_w=0.54, right_w=0.42):
    """Embrulha dois corpos de tabular em uma tabela com dois minipages."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{ref_label}}}",
        rf"\begin{{minipage}}[t]{{{left_w}\linewidth}}",
        r"\centering\tiny",
        left_body,
        r"\end{minipage}",
        r"\hfill",
        rf"\begin{{minipage}}[t]{{{right_w}\linewidth}}",
        r"\centering\tiny",
        right_body,
        r"\end{minipage}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def _head_epoch_table(epoch_rows, caption, ref_label):
    """Tabela compacta de UAS por época em 4 colunas paralelas."""
    n_ep = len(epoch_rows)
    cols = [epoch_rows[i::4] for i in range(4) if epoch_rows[i::4]]
    max_r  = max(len(c) for c in cols)
    ncols  = len(cols)
    col_spec = " ".join(["rrrr"] * ncols)
    header_once = r"\textbf{Ép.} & \textbf{UAS} & \textbf{Fg} & \textbf{Rc}"
    header = " & ".join([header_once] * ncols) + r" \\"

    lines = [
        r"\begin{table}[ht]",
        r"\centering\scriptsize",
        rf"\caption{{{caption}}}",
        rf"\label{{{ref_label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for row_i in range(max_r):
        cells = []
        for col in cols:
            if row_i < len(col):
                r = col[row_i]
                cells.append(
                    rf"{r['epoch']} & {r['uas']:.4f} & "
                    rf"{r['forgetting']:,} & {r['recovery']:,}"
                )
            else:
                cells.append(" & & & ")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _summary_table(summary_rows):
    """Tabela-resumo geral: blocos separados para sem UPOS e com UPOS."""
    def _row_line(r, has_upos):
        d  = r["deprel"]
        h  = r["head"]
        up = r.get("upos")
        if has_upos and up:
            return (
                rf"\texttt{{{r['abbrev']}}} & "
                rf"{d['final_acc']:.4f} & {d['drop']:.4f} & {d['forgetting']:,} & "
                rf"{up['final_acc']:.4f} & {up['drop']:.4f} & {up['forgetting']:,} & "
                rf"{h['peak_uas']:.4f} & {h['final_uas']:.4f} \\"
            )
        return (
            rf"\texttt{{{r['abbrev']}}} & "
            rf"{d['final_acc']:.4f} & {d['drop']:.4f} & {d['forgetting']:,} & "
            r"--- & --- & --- & "
            rf"{h['peak_uas']:.4f} & {h['final_uas']:.4f} \\"
        )

    sem_u = [r for r in summary_rows if r.get("upos") is None]
    com_u = [r for r in summary_rows if r.get("upos") is not None]

    tex = [
        r"\begin{table}[ht]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        (r"\caption{Resumo de instabilidade por modelo. "
         r"Ac.F\,=\,acurácia final. Drop\,=\,pico\,$-$\,final. "
         r"Forg.\,=\,total de eventos de esquecimento. "
         r"cU\,=\,com UPOS, sU\,=\,sem UPOS. "
         r"mB\,=\,mBERT, BtB\,=\,BERTimbau Base, BtL\,=\,BERTimbau Large, "
         r"JaB\,=\,JabuticaBERT.}"),
        r"\label{tab:resumo-instabilidade}",
        r"\begin{tabular}{l|rrr|rrr|rr}",
        r"\toprule",
        (r" & \multicolumn{3}{c|}{\textbf{DEPREL}}"
         r" & \multicolumn{3}{c|}{\textbf{UPOS}}"
         r" & \multicolumn{2}{c}{\textbf{UAS}} \\"),
        (r"\textbf{Modelo} & Ac.F & Drop & Forg."
         r" & Ac.F & Drop & Forg."
         r" & Pico & Final \\"),
        r"\midrule",
    ]
    if sem_u:
        tex.append(r"\multicolumn{9}{l}{\textit{Ablação (sem UPOS)}} \\")
        tex.extend(_row_line(r, False) for r in sem_u)
    if com_u:
        tex.append(r"\midrule")
        tex.append(r"\multicolumn{9}{l}{\textit{Multi-tarefa (com UPOS)}} \\")
        tex.extend(_row_line(r, True) for r in com_u)
    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(tex)


# ── Loop principal ────────────────────────────────────────────────────────────
json_files   = sorted(glob.glob(os.path.join(EP_DIR, "*.json")))
summary_rows = []
print(f"Arquivos encontrados em epoch_predictions/: {len(json_files)}\n")

for json_path in json_files:
    stem    = os.path.basename(json_path).replace(".json", "")
    meta    = MODEL_META.get(stem, {"display": stem, "abbrev": stem[:18]})
    display = meta["display"]
    abbrev  = meta["abbrev"]
    safe    = stem.replace("_", "-").replace(".", "-")

    print(f"{'─'*65}")
    print(f"Arquivo : {os.path.basename(json_path)}")
    print(f"Modelo  : {abbrev}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    rp    = data["RANK_PREDICTIONS"]
    tasks = list(rp.keys())

    tex_parts    = [f"% {display} ({abbrev})\n"]
    task_results: dict[str, dict] = {}

    for task in tasks:
        entries          = rp[task]
        th, tl, n_epochs = build_histories(entries)
        n_tokens         = len(th)
        print(f"  {task.upper():<6}: {n_tokens:,} tokens, {n_epochs} épocas", end="")

        if task == "head":
            epoch_rows = global_head_instability(th, n_epochs)
            peak_uas   = max(r["uas"]       for r in epoch_rows)
            final_uas  = epoch_rows[-1]["uas"]
            total_forg = sum(r["forgetting"] for r in epoch_rows)
            total_rec  = sum(r["recovery"]   for r in epoch_rows)
            task_results["head"] = {
                "epoch_rows": epoch_rows, "peak_uas": peak_uas,
                "final_uas": final_uas, "forgetting": total_forg, "recovery": total_rec,
            }
            print(f"  UAS pico={peak_uas:.4f} final={final_uas:.4f} | forg={total_forg:,}")

            head_csv = os.path.join(OUT_DIR, f"{stem}_head.csv")
            with open(head_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["epoch","uas","correct","total",
                                                   "forgetting","recovery","net_change"])
                w.writeheader(); w.writerows(epoch_rows)

        else:
            labels = LABEL_MAP.get(task, [])
            rows   = per_label_instability(th, tl, n_epochs, labels)

            all_hists  = list(th.values())
            n_all      = len(all_hists)
            global_acc = [sum(h[ep] for h in all_hists) / n_all for ep in range(n_epochs)]
            peak_acc   = max(global_acc)
            final_acc  = global_acc[-1]
            total_forg = sum(r["forgetting_events"] for r in rows)
            print(f"  final={final_acc:.4f} pico={peak_acc:.4f} forg={total_forg:,}")

            task_results[task] = {
                "rows": rows, "peak_acc": peak_acc, "final_acc": final_acc,
                "drop": round(peak_acc - final_acc, 6), "forgetting": total_forg,
            }

            inst_csv = os.path.join(OUT_DIR, f"{stem}_{task}.csv")
            fields   = ["label","label_idx","n_tokens","forgetting_events",
                        "recovery_events","oscillation","learned_forgotten",
                        "peak_acc","final_acc","drop"]
            with open(inst_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r[k] for k in fields})

            epoch_csv    = os.path.join(OUT_DIR, f"{stem}_{task}_epoch_acc.csv")
            sorted_labels = [r["label"] for r in sorted(rows, key=lambda x: x["label_idx"])]
            rows_by_lbl   = {r["label"]: r for r in rows}
            with open(epoch_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["Epoch","Global"] + sorted_labels)
                w.writeheader()
                for ep in range(n_epochs):
                    row_d = {"Epoch": ep + 1, "Global": round(global_acc[ep], 6)}
                    for lbl in sorted_labels:
                        rr = rows_by_lbl.get(lbl)
                        row_d[lbl] = round(rr["epoch_accs"][ep], 6) if rr else ""
                    w.writerow(row_d)

    # ── Monta .tex com layout lado a lado ─────────────────────────────────────
    dep = task_results.get("deprel")
    upo = task_results.get("upos")
    hd  = task_results.get("head")

    if dep and upo:
        cap = (
            f"Top-15 instabilidade DEPREL (esq.) e UPOS (dir.) --- {abbrev}. "
            r"N\,=\,n.º tokens, Forg.\,=\,esquecimentos, Rec.\,=\,recuperações, "
            r"Osc.\,=\,oscilações, A+E\,=\,aprendeu+esqueceu."
        )
        tex_parts.append(_wrap_side_by_side(
            _instab_body(dep["rows"]),
            _instab_body(upo["rows"]),
            cap, f"tab:inst-{safe}",
        ))
        cap_h = (
            f"UAS por época --- {abbrev}. "
            r"Fg\,=\,forgetting (acerto$\to$erro). Rc\,=\,recovery (erro$\to$acerto)."
        )
        tex_parts.append(_head_epoch_table(hd["epoch_rows"], cap_h, f"tab:uas-{safe}"))

    elif dep and hd:
        cap = (
            f"Instabilidade DEPREL (esq.) e UAS por época (dir.) --- {abbrev}. "
            r"N\,=\,n.º tokens, Forg.\,=\,esquecimentos, Rec.\,=\,recuperações."
        )
        tex_parts.append(_wrap_side_by_side(
            _instab_body(dep["rows"]),
            _head_body(hd["epoch_rows"]),
            cap, f"tab:inst-{safe}",
        ))

    out_tex = os.path.join(OUT_DIR, f"{stem}.tex")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n\n".join(tex_parts))
    print(f"  Salvo : instability_results/{stem}.tex")

    entry: dict = {"abbrev": abbrev, "display": display}
    if dep:
        entry["deprel"] = {"final_acc": dep["final_acc"],
                           "drop": dep["drop"], "forgetting": dep["forgetting"]}
    if upo:
        entry["upos"]   = {"final_acc": upo["final_acc"],
                           "drop": upo["drop"], "forgetting": upo["forgetting"]}
    if hd:
        entry["head"]   = {"peak_uas": hd["peak_uas"], "final_uas": hd["final_uas"],
                           "forgetting": hd["forgetting"]}
    summary_rows.append(entry)

# ── Tabela-resumo ─────────────────────────────────────────────────────────────
summary_tex = os.path.join(OUT_DIR, "00_resumo_instabilidade.tex")
with open(summary_tex, "w", encoding="utf-8") as f:
    f.write("% Tabela-resumo gerada por label_instability_analysis.py\n"
            + _summary_table(summary_rows))

generated = glob.glob(os.path.join(OUT_DIR, "*.tex"))
csvs      = glob.glob(os.path.join(OUT_DIR, "*.csv"))
print(f"\n{'='*65}")
print(f"Arquivos .tex : {len(generated)}  |  .csv : {len(csvs)}")
print(f"\n\\input{{instability_results/00_resumo_instabilidade}}")
print("% Tabelas individuais:")
for p in sorted(glob.glob(os.path.join(OUT_DIR, "*.tex"))):
    if "resumo" not in os.path.basename(p):
        print(f"\\input{{instability_results/{os.path.basename(p).replace('.tex','')}}}")
