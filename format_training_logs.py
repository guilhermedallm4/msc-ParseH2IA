import glob
import csv
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# Carrega resultados finais do trainer.evaluate() agrupados por nome do modelo
jsonl_path = os.path.join(BASE, "ablacao_linear_results.jsonl")
eval_results = {}
with open(jsonl_path) as f:
    for line in f:
        entry = json.loads(line)
        key = entry["name"].replace("/", "_")
        eval_results[key] = {"uas": entry["uas"], "las": entry["las"]}

pattern = os.path.join(BASE, "training_log_ablacao_linear_*.csv")

for filepath in sorted(glob.glob(pattern)):
    model_name = os.path.basename(filepath).replace("training_log_ablacao_linear_", "").replace(".csv", "")
    print(f"% {model_name}")

    rows = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "epoch":      int(row["Epoch"]),
                "train_loss": float(row["Training Loss"]),
                "val_loss":   float(row["Validation Loss"]),
                "uas":        float(row["Uas"]),
                "las":        float(row["Las"]),
            })

    for r in rows:
        print(f"{r['epoch']} & {r['train_loss']:.6f} & {r['val_loss']:.6f} & {r['uas']:.6f} & {r['las']:.6f} \\\\")

    # Linha do trainer.evaluate() (melhor modelo carregado ao final)
    if model_name in eval_results:
        ev = eval_results[model_name]
        print(f"% trainer.evaluate()  UAS={ev['uas']:.6f}  LAS={ev['las']:.6f}")
        print(f"\\hline")
        print(f"best & -- & -- & {ev['uas']:.6f} & {ev['las']:.6f} \\\\")

    print()
