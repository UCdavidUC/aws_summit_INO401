#!/usr/bin/env bash
# ============================================================================
# 06b_evaluate_models.sh
# ----------------------------------------------------------------------------
# Evalua calidad (perplexity sobre eval.jsonl) y rendimiento (tokens/s,
# TTFT, latencia P50/P99) de TODOS los modelos entrenados en la corrida en
# paralelo mas reciente (05b_run_parallel_finetuning_jobs.sh), usando
# training/evaluate_models.py.
#
# Esto corresponde a la Seccion 16 de docs/technical_documentation.md
# (evaluacion de calidad y rendimiento post-entrenamiento), aplicada aqui a
# los adaptadores recien entrenados en lugar de a un endpoint desplegado.
#
# Requiere:
#   - training/eval_requirements.txt instalado (torch, transformers, peft,
#     accelerate) en un entorno con GPU disponible (idealmente la MISMA
#     instancia/entorno donde se entreno, para reusar el cache de modelos
#     de Hugging Face Hub).
#   - scripts/.env con DATA_BUCKET_NAME, AWS_REGION, y PARALLEL_RUN_MANIFEST
#     (generado por 05b_run_parallel_finetuning_jobs.sh).
#
# Salida: training/eval_metrics_<job_name>.json por modelo +
# training/eval_summary_<timestamp>.json consolidado.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${SCRIPT_DIR}/.env" ]; then
  source "${SCRIPT_DIR}/.env"
fi
: "${DATA_BUCKET_NAME:?Debes correr 02_deploy_data_and_training_stacks.sh primero o exportar DATA_BUCKET_NAME manualmente}"
: "${AWS_REGION:=us-west-2}"
: "${PARALLEL_RUN_MANIFEST:?Debes correr 05b_run_parallel_finetuning_jobs.sh primero o exportar PARALLEL_RUN_MANIFEST manualmente}"

EVAL_JSONL="${ROOT_DIR}/data_pipeline/processed/dataset/eval.jsonl"
if [ ! -f "${EVAL_JSONL}" ]; then
  echo "ERROR: No se encontro ${EVAL_JSONL}. Corre primero 04_process_and_build_dataset.sh"
  exit 1
fi

EVAL_VENV="${ROOT_DIR}/training/.venv-eval"
if [ ! -d "${EVAL_VENV}" ]; then
  echo "== Creando entorno virtual de evaluacion (torch/transformers/peft) =="
  python3 -m venv "${EVAL_VENV}"
  "${EVAL_VENV}/bin/pip" install -q --upgrade pip
  "${EVAL_VENV}/bin/pip" install -q -r "${ROOT_DIR}/training/eval_requirements.txt" boto3
fi

cd "${ROOT_DIR}/training"

echo "== Evaluando calidad y rendimiento de los modelos de: ${PARALLEL_RUN_MANIFEST} =="
"${EVAL_VENV}/bin/python" evaluate_models.py \
  --run-manifest "${PARALLEL_RUN_MANIFEST}" \
  --bucket "${DATA_BUCKET_NAME}" \
  --region "${AWS_REGION}" \
  --eval-jsonl "${EVAL_JSONL}"

LATEST_SUMMARY=$(ls -t eval_summary_*.json | head -n1)
echo ""
echo "== Listo =="
echo "Resumen de evaluacion: ${ROOT_DIR}/training/${LATEST_SUMMARY}"
echo "Metricas individuales: ${ROOT_DIR}/training/eval_metrics_<job_name>.json"
echo ""
echo "Analiza y visualiza los resultados (seaborn) en la Seccion de evaluacion del notebook:"
echo "  entrenamiento_slm_cnbv_banxico.ipynb"
