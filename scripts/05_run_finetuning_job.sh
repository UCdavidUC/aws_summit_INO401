#!/usr/bin/env bash
# ============================================================================
# 05_run_finetuning_job.sh
# ----------------------------------------------------------------------------
# Lanza el SageMaker Training Job de fine-tuning QLoRA de Qwen2.5-1.5B-Instruct
# sobre el dataset generado en el paso anterior, y espera hasta que finalice.
#
# Detalles tecnicos (ver training/source/train_qlora.py):
#   - Cuantizacion NF4 de 4 bits del modelo base (bitsandbytes)
#   - Adaptadores LoRA (r=16, alpha=32) sobre q/k/v/o_proj y gate/up/down_proj
#   - 3 epochs, batch=4, grad-accum=4, lr=2e-4
#   - Instancia: ml.g6.2xlarge (1x GPU NVIDIA L4 24GB; mismo VRAM que
#     ml.g5.2xlarge/A10G pero ~15-20% mas barata por hora)
#
# Variables de entorno opcionales:
#   INSTANCE_TYPE   Tipo de instancia de SageMaker (default: ml.g6.2xlarge)
#   EPOCHS          Numero de epochs (default: 3)
#   JOB_NAME        Nombre del training job (default: se genera con timestamp)
#
# IMPORTANTE (costo/tiempo): este job usa una instancia GPU real y tarda
# aproximadamente 60-100 minutos para un dataset de tamano similar al
# original (~6500 ejemplos). Verifica antes de correr que la cuota de
# servicio 'ml.g6.2xlarge for training job usage' sea >= 1 en la region
# elegida (Service Quotas > Amazon SageMaker).
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

INSTANCE_TYPE="${INSTANCE_TYPE:-ml.g6.2xlarge}"
EPOCHS="${EPOCHS:-3}"

TRAIN_JSONL="${ROOT_DIR}/data_pipeline/processed/dataset/train.jsonl"
EVAL_JSONL="${ROOT_DIR}/data_pipeline/processed/dataset/eval.jsonl"

if [ ! -f "${TRAIN_JSONL}" ] || [ ! -f "${EVAL_JSONL}" ]; then
  echo "ERROR: No se encontraron ${TRAIN_JSONL} / ${EVAL_JSONL}."
  echo "Corre primero 04_process_and_build_dataset.sh"
  exit 1
fi

cd "${ROOT_DIR}/training"
PYTHON="${ROOT_DIR}/data_pipeline/.venv/bin/python"  # reutiliza el venv que ya tiene boto3

echo "== Lanzando SageMaker Training Job (QLoRA, Qwen2.5-1.5B-Instruct) =="
echo "   Bucket:       ${DATA_BUCKET_NAME}"
echo "   Rol:          ${TRAINING_ROLE_ARN}"
echo "   Instancia:    ${INSTANCE_TYPE}"
echo "   Epochs:       ${EPOCHS}"
echo "   Esto puede tardar 60-100 minutos. El script espera hasta que termine."
echo ""

ARGS=(
  --bucket "${DATA_BUCKET_NAME}"
  --role-arn "${TRAINING_ROLE_ARN}"
  --region "${AWS_REGION}"
  --instance-type "${INSTANCE_TYPE}"
  --train-jsonl "${TRAIN_JSONL}"
  --eval-jsonl "${EVAL_JSONL}"
  --epochs "${EPOCHS}"
)
if [ -n "${JOB_NAME:-}" ]; then
  ARGS+=(--job-name "${JOB_NAME}")
fi

"${PYTHON}" launch_training_job.py "${ARGS[@]}"

# El launcher imprime el nombre del job y guarda metrics_<job_name>.json en
# este directorio. Extraemos el ultimo archivo de metricas generado para
# encontrar el nombre del job y la ubicacion del modelo en S3.
LATEST_METRICS_FILE=$(ls -t metrics_*.json | head -n1)
JOB_NAME_DETECTED=$(python3 -c "import json; print(json.load(open('${LATEST_METRICS_FILE}'))['training_job_name'])")
JOB_STATUS=$(python3 -c "import json; print(json.load(open('${LATEST_METRICS_FILE}'))['training_job_status'])")

mkdir -p "${SCRIPT_DIR}"
{
  echo ""
  echo "TRAINING_JOB_NAME=${JOB_NAME_DETECTED}"
  echo "MODEL_S3_URI=s3://${DATA_BUCKET_NAME}/models/${JOB_NAME_DETECTED}/${JOB_NAME_DETECTED}/output/model.tar.gz"
} >> "${SCRIPT_DIR}/.env"

echo ""
echo "== Listo =="
echo "Estado del job:    ${JOB_STATUS}"
echo "Nombre del job:    ${JOB_NAME_DETECTED}"
echo "Metricas guardadas en: ${ROOT_DIR}/training/${LATEST_METRICS_FILE}"
echo "Modelo (adaptador LoRA) en: s3://${DATA_BUCKET_NAME}/models/${JOB_NAME_DETECTED}/${JOB_NAME_DETECTED}/output/model.tar.gz"

if [ "${JOB_STATUS}" != "Completed" ]; then
  echo "AVISO: el job no termino con estado Completed. Revisa la consola de SageMaker antes de continuar."
  exit 1
fi
