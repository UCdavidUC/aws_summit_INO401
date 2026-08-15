#!/usr/bin/env bash
# ============================================================================
# 04_process_and_build_dataset.sh
# ----------------------------------------------------------------------------
# Pipeline de procesamiento de datos:
#   1. Extrae texto de cada PDF descargado (pypdf) con limpieza basica.
#   2. Divide cada documento en chunks (~3000 caracteres, por parrafos).
#   3. Usa Amazon Bedrock (Claude Haiku por defecto) para generar pares de
#      instruccion-respuesta en espanol de Mexico a partir de cada chunk,
#      cubriendo resumen estructurado, checklist de obligaciones,
#      clasificacion regulatoria y explicacion en lenguaje sencillo.
#   4. Escribe train.jsonl / eval.jsonl (formato chat) y los sube a S3.
#
# Variables de entorno opcionales:
#   MAX_CHUNKS_PER_DOC    Tope de chunks por documento para no sobre-
#                         representar documentos largos (default: 10)
#   NUM_EXAMPLES_PER_CHUNK  Ejemplos generados por chunk (default: 2)
#   BEDROCK_MODEL_ID      Modelo de Bedrock usado para generar el dataset
#                         (default: us.anthropic.claude-haiku-4-5-20251001-v1:0)
#   MAX_WORKERS           Llamadas concurrentes a Bedrock (default: 12)
#
# IMPORTANTE: este paso invoca Amazon Bedrock miles de veces (una por chunk).
# Verifica el acceso a modelos habilitado en la consola de Bedrock antes de
# correr este script. Tiempo estimado: 30-45 minutos para un corpus similar
# al original (~3600 chunks seleccionados). Costo aproximado con Claude
# Haiku: unos pocos dolares.
#
# Requiere: scripts/.env con DATA_BUCKET_NAME y AWS_REGION.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${SCRIPT_DIR}/.env" ]; then
  source "${SCRIPT_DIR}/.env"
fi
: "${DATA_BUCKET_NAME:?Debes correr 02_deploy_data_and_training_stacks.sh primero o exportar DATA_BUCKET_NAME manualmente}"
: "${AWS_REGION:=us-west-2}"

MAX_CHUNKS_PER_DOC="${MAX_CHUNKS_PER_DOC:-10}"
NUM_EXAMPLES_PER_CHUNK="${NUM_EXAMPLES_PER_CHUNK:-2}"
BEDROCK_MODEL_ID="${BEDROCK_MODEL_ID:-us.anthropic.claude-haiku-4-5-20251001-v1:0}"
MAX_WORKERS="${MAX_WORKERS:-12}"

cd "${ROOT_DIR}/data_pipeline"
PYTHON="${ROOT_DIR}/data_pipeline/.venv/bin/python"

echo "== Extrayendo texto de PDFs de CNBV =="
"${PYTHON}" processing/extract_text.py --raw-dir ./raw_cnbv --source cnbv --out-dir ./processed/text

echo "== Extrayendo texto de PDFs de Banxico =="
"${PYTHON}" processing/extract_text.py --raw-dir ./raw_banxico --source banxico --out-dir ./processed/text

echo "== Generando chunks de texto =="
"${PYTHON}" processing/chunk_documents.py --text-dir ./processed/text --out ./processed/chunks.jsonl

echo "== Generando dataset de instruccion via Amazon Bedrock (modelo: ${BEDROCK_MODEL_ID}) =="
echo "   Esto puede tardar 30-45 minutos dependiendo del volumen de chunks."
"${PYTHON}" processing/build_dataset.py \
  --chunks ./processed/chunks.jsonl \
  --out-dir ./processed/dataset \
  --max-chunks-per-doc "${MAX_CHUNKS_PER_DOC}" \
  --num-examples-per-chunk "${NUM_EXAMPLES_PER_CHUNK}" \
  --region "${AWS_REGION}" \
  --max-workers "${MAX_WORKERS}" \
  --model-id "${BEDROCK_MODEL_ID}" \
  --eval-fraction 0.1 \
  --seed 42

echo "== Subiendo dataset a s3://${DATA_BUCKET_NAME}/datasets/full/ =="
aws s3 cp ./processed/dataset/train.jsonl "s3://${DATA_BUCKET_NAME}/datasets/full/train.jsonl" --region "${AWS_REGION}"
aws s3 cp ./processed/dataset/eval.jsonl "s3://${DATA_BUCKET_NAME}/datasets/full/eval.jsonl" --region "${AWS_REGION}"
aws s3 cp ./processed/dataset/dataset_with_metadata.jsonl "s3://${DATA_BUCKET_NAME}/datasets/full/dataset_with_metadata.jsonl" --region "${AWS_REGION}"

TRAIN_COUNT=$(wc -l < ./processed/dataset/train.jsonl | tr -d ' ')
EVAL_COUNT=$(wc -l < ./processed/dataset/eval.jsonl | tr -d ' ')

echo ""
echo "== Listo =="
echo "Ejemplos de entrenamiento: ${TRAIN_COUNT}"
echo "Ejemplos de evaluacion:    ${EVAL_COUNT}"
echo "Dataset local en:          data_pipeline/processed/dataset/{train,eval}.jsonl"
echo "Subido a:                  s3://${DATA_BUCKET_NAME}/datasets/full/"
