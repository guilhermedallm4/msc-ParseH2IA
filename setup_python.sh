#!/usr/bin/env bash

set -e

PYTHON_VERSION=3.10.20
INSTALL_DIR=$HOME/python-$PYTHON_VERSION
VENV_DIR=$PWD/.venv

echo ">> Instalando dependências do sistema (Ubuntu/Debian)..."
sudo apt update
sudo apt install -y \
  build-essential \
  wget \
  libssl-dev \
  zlib1g-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  libffi-dev \
  libncursesw5-dev \
  xz-utils \
  tk-dev \
  libxml2-dev \
  libxmlsec1-dev \
  liblzma-dev

echo ">> Baixando Python $PYTHON_VERSION..."
cd /tmp
wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz

echo ">> Extraindo..."
tar -xf Python-$PYTHON_VERSION.tgz
cd Python-$PYTHON_VERSION

echo ">> Compilando (isso pode demorar)..."
./configure --prefix=$INSTALL_DIR --enable-optimizations
make -j$(nproc)
make install

echo ">> Criando virtualenv..."
$INSTALL_DIR/bin/python3.10 -m venv $VENV_DIR

echo ">> Ativando virtualenv..."
source $VENV_DIR/bin/activate

echo ">> Atualizando pip..."
pip install --upgrade pip setuptools wheel

echo ">> Pronto!"
echo "Python instalado em: $INSTALL_DIR"
echo "Virtualenv em: $VENV_DIR"
echo "Para ativar depois: source $VENV_DIR/bin/activate"