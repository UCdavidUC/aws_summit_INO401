#!/usr/bin/env bash
# ============================================================================
# 01_setup_envs.sh
# ----------------------------------------------------------------------------
# Crea los entornos virtuales de Python necesarios para cada componente del
# proyecto e instala sus dependencias:
#   - data_pipeline/.venv  -> scraping, procesamiento, generacion de dataset
#   - infra/.venv          -> AWS CDK app (Python)
#   - training/.venv       -> lanzador del training job (boto3 puro)
#
# No requiere argumentos. Es idempotente: si un venv ya existe, reutiliza el
# existente e instala/actualiza dependencias.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

setup_venv() {
  local dir="$1"
  local req_file="$2"
  echo "== Configurando entorno virtual en ${dir} =="
  if [ ! -d "${dir}/.venv" ]; then
    python3 -m venv "${dir}/.venv"
  fi
  "${dir}/.venv/bin/pip" install --upgrade pip -q
  if [ -f "${req_file}" ]; then
    "${dir}/.venv/bin/pip" install -r "${req_file}" -q
  fi
  echo "OK: ${dir}/.venv"
}

setup_venv "${ROOT_DIR}/data_pipeline" "${ROOT_DIR}/data_pipeline/requirements.txt"
setup_venv "${ROOT_DIR}/infra" "${ROOT_DIR}/infra/requirements.txt"
if [ -f "${ROOT_DIR}/infra/requirements-dev.txt" ]; then
  "${ROOT_DIR}/infra/.venv/bin/pip" install -r "${ROOT_DIR}/infra/requirements-dev.txt" -q
fi
# El launcher del training job solo necesita boto3; se reutiliza el venv de
# data_pipeline (ya lo incluye) para no duplicar entornos.

echo ""
echo "Entornos listos:"
echo "  - ${ROOT_DIR}/data_pipeline/.venv (scraping, procesamiento, dataset, y launcher de training)"
echo "  - ${ROOT_DIR}/infra/.venv (AWS CDK)"
