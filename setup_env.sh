#!/usr/bin/env bash
# ==============================================================================
# QuantumAlpha - Isolated Environment Setup Script
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

# 2. Check if .venv exists, otherwise create it
if [ ! -d "${VENV_DIR}" ]; then
    echo "[*] Creating virtual environment at: ${VENV_DIR}"
    ${PYTHON_BIN} -m venv "${VENV_DIR}" || {
        echo "[-] Standard venv module failed. Trying virtualenv..."
        virtualenv "${VENV_DIR}" || {
            echo "[!] Note: To create a local venv on Debian/Ubuntu, run: sudo apt install python3-venv"
            echo "[!] Alternatively, run inside Docker with zero system modifications:"
            echo "    docker compose up --build"
            exit 1
        }
    }
fi

# 3. Activate venv
source "${VENV_DIR}/bin/activate"
echo "[+] Virtual environment activated: $(which python)"

# 4. Install / Upgrade pip and dependencies inside venv
echo "[*] Installing dependencies from requirements.txt inside .venv..."
pip install --upgrade pip
pip install -r "${PROJECT_DIR}/requirements.txt"

echo "======================================================================"
echo "  Setup Complete! To activate your isolated environment:"
echo "    source .venv/bin/activate"
echo "  To start the Flask Web Application:"
echo "    python web/app.py"
echo "======================================================================"
