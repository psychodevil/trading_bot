#!/usr/bin/env bash
# ==============================================================================
# QuantumAlpha - Isolated Virtual Environment Setup Script
# Creates and configures a clean virtual environment without modifying system packages.
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"

echo "======================================================================"
echo "  QuantumAlpha: Setting up Isolated Virtual Environment"
echo "======================================================================"

# 1. Check Python Version
PYTHON_BIN="python3"
if ! command -v ${PYTHON_BIN} &> /dev/null; then
    echo "[-] Error: python3 is not installed."
    exit 1
fi

PY_VERSION=$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[+] Detected Python version: ${PY_VERSION}"

# 2. Create isolated venv if not already initialized
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "[*] Initializing isolated virtual environment at: ${VENV_DIR}"
    ${PYTHON_BIN} -m venv --without-pip "${VENV_DIR}"
fi

# 3. Bootstrap pip inside .venv if missing
if [ ! -f "${VENV_DIR}/bin/pip" ]; then
    echo "[*] Bootstrapping pip inside isolated .venv..."
    curl -sS https://bootstrap.pypa.io/get-pip.py -o "${PROJECT_DIR}/get-pip.py"
    "${VENV_DIR}/bin/python3" "${PROJECT_DIR}/get-pip.py"
    rm -f "${PROJECT_DIR}/get-pip.py"
fi

# 4. Activate venv
source "${VENV_DIR}/bin/activate"
echo "[+] Virtual environment active: $(which python)"

# 5. Install dependencies inside .venv
echo "[*] Installing dependencies inside .venv..."
pip install --upgrade pip
pip install -r "${PROJECT_DIR}/requirements.txt" || pip install Flask

echo "======================================================================"
echo "  Setup Complete! To activate your isolated environment:"
echo "    source .venv/bin/activate"
echo "  To start the Flask Web Application:"
echo "    python web/app.py"
echo "======================================================================"
