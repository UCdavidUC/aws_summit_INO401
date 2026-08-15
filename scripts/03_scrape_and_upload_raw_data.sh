#!/usr/bin/env bash
# ============================================================================
# 03_scrape_and_upload_raw_data.sh
# ----------------------------------------------------------------------------
# Descarga los documentos normativos publicos de CNBV (9 sectores
# supervisados) y Banxico (circulares/disposiciones historicas), y sube los
# PDFs + metadata.jsonl al bucket S3 creado en el paso anterior.
#
# Este scraping respeta los sitios objetivo (rate limiting con pausas entre
# requests) y usa un bundle de certificados combinado (certifi + los
# intermedios de GlobalSign/GoDaddy) porque ambos sitios presentan cadenas de
# certificados TLS incompletas. Ver data_pipeline/scraping/ca_bundle.py.
#
# Requiere: scripts/.env (generado por 02_deploy_data_and_training_stacks.sh)
# con DATA_BUCKET_NAME y AWS_REGION.
#
# Tiempo estimado: 15-25 minutos (depende de la latencia de red hacia los
# sitios de CNBV/Banxico). Descarga ~250MB-400MB de PDFs en total.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${SCRIPT_DIR}/.env" ]; then
  source "${SCRIPT_DIR}/.env"
fi
: "${DATA_BUCKET_NAME:?Debes correr 02_deploy_data_and_training_stacks.sh primero o exportar DATA_BUCKET_NAME manualmente}"
: "${AWS_REGION:=us-west-2}"

cd "${ROOT_DIR}/data_pipeline"
PYTHON="${ROOT_DIR}/data_pipeline/.venv/bin/python"

echo "== Descargando documentos de CNBV (9 sectores) =="
"${PYTHON}" scraping/scrape_cnbv.py --out-dir ./raw_cnbv

echo "== Descargando documentos de Banxico (circulares/disposiciones) =="
"${PYTHON}" scraping/scrape_banxico.py --out-dir ./raw_banxico

echo "== Subiendo documentos crudos a s3://${DATA_BUCKET_NAME}/raw/ =="
aws s3 sync ./raw_cnbv "s3://${DATA_BUCKET_NAME}/raw/cnbv/" --region "${AWS_REGION}"
aws s3 sync ./raw_banxico "s3://${DATA_BUCKET_NAME}/raw/banxico/" --region "${AWS_REGION}"

CNBV_COUNT=$(find ./raw_cnbv -name "*.pdf" | wc -l | tr -d ' ')
BANXICO_COUNT=$(find ./raw_banxico -name "*.pdf" | wc -l | tr -d ' ')

echo ""
echo "== Listo =="
echo "PDFs de CNBV descargados:    ${CNBV_COUNT}"
echo "PDFs de Banxico descargados: ${BANXICO_COUNT}"
echo "Subidos a: s3://${DATA_BUCKET_NAME}/raw/{cnbv,banxico}/"
