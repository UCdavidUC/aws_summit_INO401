#!/usr/bin/env bash
# ============================================================================
# 06_deploy_agent_runtime.sh
# ----------------------------------------------------------------------------
# 1. Descarga el adaptador LoRA entrenado (model.tar.gz) desde S3, extrae
#    solo los archivos del adaptador + tokenizer, y los vuelve a subir a
#    s3://<bucket>/models/latest/ (prefijo que agent_runtime/app.py lee via
#    la variable de entorno MODEL_S3_PREFIX).
# 2. Despliega SlmAgentRuntimeStack con 'cdk deploy', lo que construye la
#    imagen Docker ARM64 del agente (requiere Docker corriendo) y la publica
#    en un repositorio ECR administrado por CDK, y crea el recurso
#    AWS::BedrockAgentCore::Runtime.
#
# Requiere: scripts/.env con DATA_BUCKET_NAME, AWS_REGION, MODEL_S3_URI
# (generado por 05_run_finetuning_job.sh).
#
# Tiempo estimado: 3-5 minutos (build de imagen + push a ECR + creacion del
# runtime).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${SCRIPT_DIR}/.env" ]; then
  source "${SCRIPT_DIR}/.env"
fi
: "${DATA_BUCKET_NAME:?Debes correr 02_deploy_data_and_training_stacks.sh primero o exportar DATA_BUCKET_NAME manualmente}"
: "${MODEL_S3_URI:?Debes correr 05_run_finetuning_job.sh primero o exportar MODEL_S3_URI manualmente (s3://bucket/models/<job>/<job>/output/model.tar.gz)}"
: "${AWS_REGION:=us-west-2}"

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker no esta corriendo. Es necesario para construir la imagen ARM64 del agente."
  echo "En macOS con Colima: colima start --arch aarch64 --vm-type=vz --cpu 4 --memory 8 --disk 60"
  exit 1
fi

WORKDIR=$(mktemp -d)
trap 'rm -rf "${WORKDIR}"' EXIT

echo "== Descargando adaptador entrenado desde ${MODEL_S3_URI} =="
aws s3 cp "${MODEL_S3_URI}" "${WORKDIR}/model.tar.gz" --region "${AWS_REGION}"

echo "== Extrayendo artefactos del adaptador =="
mkdir -p "${WORKDIR}/extracted"
tar -xzf "${WORKDIR}/model.tar.gz" -C "${WORKDIR}/extracted"

# Solo se sube el adaptador LoRA + tokenizer, NO los pesos del modelo base
# (esos se descargan de Hugging Face Hub al iniciar el contenedor). Esto
# evita duplicar varios GB en S3.
REQUIRED_FILES=(adapter_model.safetensors adapter_config.json tokenizer.json tokenizer_config.json chat_template.jinja)

echo "== Subiendo adaptador a s3://${DATA_BUCKET_NAME}/models/latest/ =="
for f in "${REQUIRED_FILES[@]}"; do
  if [ -f "${WORKDIR}/extracted/${f}" ]; then
    aws s3 cp "${WORKDIR}/extracted/${f}" "s3://${DATA_BUCKET_NAME}/models/latest/${f}" --region "${AWS_REGION}"
  else
    echo "AVISO: no se encontro ${f} en el artefacto del modelo (puede no ser critico, p.ej. chat_template.jinja)."
  fi
done

echo "== Desplegando SlmAgentRuntimeStack (construira imagen Docker ARM64) =="
cd "${ROOT_DIR}/infra"
source .venv/bin/activate
export CDK_DEFAULT_ACCOUNT="${CDK_DEFAULT_ACCOUNT:-$(aws sts get-caller-identity --query Account --output text)}"
export CDK_DEFAULT_REGION="${AWS_REGION}"

cdk deploy SlmAgentRuntimeStack --require-approval never

AGENT_RUNTIME_ARN=$(aws cloudformation describe-stacks \
  --stack-name SlmAgentRuntimeStack --region "${AWS_REGION}" \
  --query "Stacks[0].Outputs[?OutputKey=='AgentRuntimeArn'].OutputValue" --output text)
AGENT_RUNTIME_ID=$(aws cloudformation describe-stacks \
  --stack-name SlmAgentRuntimeStack --region "${AWS_REGION}" \
  --query "Stacks[0].Outputs[?OutputKey=='AgentRuntimeId'].OutputValue" --output text)

cat >> "${SCRIPT_DIR}/.env" <<EOF

AGENT_RUNTIME_ARN=${AGENT_RUNTIME_ARN}
AGENT_RUNTIME_ID=${AGENT_RUNTIME_ID}
EOF

echo ""
echo "== Listo =="
echo "AgentCore Runtime ARN: ${AGENT_RUNTIME_ARN}"
echo "AgentCore Runtime ID:  ${AGENT_RUNTIME_ID}"
echo ""
echo "Verifica el estado con:"
echo "  aws bedrock-agentcore-control get-agent-runtime --agent-runtime-id ${AGENT_RUNTIME_ID} --region ${AWS_REGION}"
echo "Prueba el agente con:"
echo "  ./scripts/07_invoke_agent.sh \"Explica brevemente el proposito de la Circular Unica de Bancos.\""
