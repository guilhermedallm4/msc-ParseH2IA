#!/bin/bash

# Nome da sessão screen
SESSION_NAME="optuna_run"

# Caminho do notebook (fixo)
NOTEBOOK_PATH="/home/guilhermelima/msc/ablacao_sem_upos_linear.ipynb"

# Diretório do notebook
NOTEBOOK_DIR="$(dirname "$NOTEBOOK_PATH")"

# Nome do arquivo de saída
OUTPUT_NAME="executed_$(basename "$NOTEBOOK_PATH")"

# Ativar venv (ajuste se necessário)
VENV_PATH="/home/guilhermelima/msc/.venv"

# Comando de execução
EXEC_CMD="cd \"$NOTEBOOK_DIR\" && source \"$VENV_PATH/bin/activate\" && jupyter nbconvert --to notebook --execute \"$NOTEBOOK_PATH\" --output \"$OUTPUT_NAME\""

# Inicia screen
screen -dmS "$SESSION_NAME" bash -c "$EXEC_CMD; echo 'Execução finalizada'; exec bash"

echo "🚀 Notebook rodando na screen: $SESSION_NAME"
echo "📎 Para acompanhar: screen -r $SESSION_NAME"