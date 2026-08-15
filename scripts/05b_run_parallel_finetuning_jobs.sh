#!/usr/bin/env bash
# ============================================================================
# 05b_run_parallel_finetuning_jobs.sh
# ----------------------------------------------------------------------------
# Variante de 05_run_finetuning_job.sh que lanza EN PARALELO un training job
# QLoRA independiente por cada modelo del catalogo
# (training/model_catalog.py: qwen2.5-1.5b, qwen3-0.6b por defecto), todos
# sobre el mismo dataset generado en el paso 04, para poder comparar tiempo
# de entrenamiento, memoria y loss entre modelos bajo las mismas condiciones
# (Seccion 3/7/17 de docs/technical_documentation.md).
#
# Usa training/launch_training_job.py en su modo multi-modelo
# (--model-keys), que:
#   - crea un training job por modelo con la instancia recomendada de su
#     ficha en el catalogo (o la que se fije en INSTANCE_TYPE para todos),
#   - los espera concurrentemente con un ThreadPoolExecutor,
#   - escribe metrics_<job_name>.json por modelo y un
#     parallel_run_<timestamp>.json consolidado.
#
# Variables de entorno opcionales:
#   MODEL_KEYS      Lista separada por comas de claves del catalogo
#                   (default: catalogo completo, ver model_catalog.py)
#   INSTANCE_TYPE   Si se define, sobreescribe la instancia recomendada del
#                   catalogo para TODOS los modelos (default: por modelo)
#   MAX_WORKERS     Jobs en paralelo (default: 2, uno por modelo)
#   EPOCHS          Numero de epochs (default: 3)
#
# IMPORTANTE (costo/tiempo/cuota): este script lanza varias instancias GPU
# SIMULTANEAMENTE. Verifica antes de correr que la cuota de servicio
# 'ml.g6.xlarge for training job usage' en Service Quotas sea
# suficiente para el numero de jobs en paralelo (por defecto 2), o reduce
# MAX_WORKERS / MODEL_KEYS.
#
# Requiere: scripts/.env con DATA_BUCKET_NAME, TRAINING_ROLE_ARN, AWS_REGION.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${SCRIPT_DIR}/.env" ]; then
  source "${SCRIPT_DIR}/.env"
fi
: "${DATA_BUCKET_NAME:?Debes correr 02_deploy_data_and_training_stacks.sh primero o exportar DATA_BUCKET_NAME manualmente}"
: "${TRAINING_ROLE_ARN:?Debes correr 02_deploy_data_and_training_stacks.sh primero o exportar TRAINING_ROLE_ARN manualmente}"
: "${AWS_REGION:=us-west-2}"

EPOCHS="${EPOCHS:-3}"
MAX_WORKERS="${MAX_WORKERS:-2}"

TRAIN_JSONL="${ROOT_DIR}/data_pipeline/processed/dataset/train.jsonl"
EVAL_JSONL="${ROOT_DIR}/data_pipeline/processed/dataset/eval.jsonl"

if [ ! -f "${TRAIN_JSONL}" ] || [ ! -f "${EVAL_JSONL}" ]; then
  echo "ERROR: No se encontraron ${TRAIN_JSONL} / ${EVAL_JSONL}."
  echo "Corre primero 04_process_and_build_dataset.sh"
  exit 1
fi

cd "${ROOT_DIR}/training"
PYTHON="${ROOT_DIR}/data_pipeline/.venv/bin/python"  # reutiliza el venv que ya tiene boto3

ARGS=(
  --bucket "${DATA_BUCKET_NAME}"
  --role-arn "${TRAINING_ROLE_ARN}"
  --region "${AWS_REGION}"
  --train-jsonl "${TRAIN_JSONL}"
  --eval-jsonl "${EVAL_JSONL}"
  --epochs "${EPOCHS}"
  --max-workers "${MAX_WORKERS}"
)
if [ -n "${MODEL_KEYS:-}" ]; then
  ARGS+=(--model-keys "${MODEL_KEYS}")
fi
if [ -n "${INSTANCE_TYPE:-}" ]; then
  ARGS+=(--instance-type "${INSTANCE_TYPE}")
fi

echo "== Lanzando training jobs QLoRA EN PARALELO (multi-modelo) =="
echo "   Bucket:       ${DATA_BUCKET_NAME}"
echo "   Rol:          ${TRAINING_ROLE_ARN}"
echo "   Modelos:      ${MODEL_KEYS:-catalogo completo (ver training/model_catalog.py)}"
echo "   Max workers:  ${MAX_WORKERS}"
echo "   Epochs:       ${EPOCHS}"
echo "   Esto puede tardar 60-120 minutos (todos los jobs corren simultaneamente)."
echo ""

"${PYTHON}" launch_training_job.py "${ARGS[@]}"

LATEST_MANIFEST=$(ls -t parallel_run_*.json | head -n1)
echo ""
echo "== Listo =="
echo "Resumen consolidado de la corrida en paralelo: ${ROOT_DIR}/training/${LATEST_MANIFEST}"
echo "Metricas individuales por modelo: ${ROOT_DIR}/training/metrics_<job_name>.json"

mkdir -p "${SCRIPT_DIR}"
{
  echo ""
  echo "PARALLEL_RUN_MANIFEST=${ROOT_DIR}/training/${LATEST_MANIFEST}"
} >> "${SCRIPT_DIR}/.env"

echo ""
echo "Siguiente paso: evalua calidad/rendimiento de cada modelo con:"
echo "  ./06b_evaluate_models.sh"
