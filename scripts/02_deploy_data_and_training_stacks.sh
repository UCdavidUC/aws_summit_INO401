#!/usr/bin/env bash
# ============================================================================
# 02_deploy_data_and_training_stacks.sh
# ----------------------------------------------------------------------------
# Hace 'cdk bootstrap' (si es necesario) y despliega las 2 stacks que no
# requieren Docker: SlmDataPipelineStack (bucket S3 + politica IAM) y
# SlmTrainingStack (rol de ejecucion de SageMaker).
#
# El AgentRuntimeStack (que si requiere Docker para construir la imagen del
# agente) se despliega por separado en 06_deploy_agent_runtime.sh, una vez
# que el modelo ya fue entrenado.
#
# Variables de entorno opcionales:
#   AWS_REGION      Region de despliegue (default: us-west-2)
#   CDK_DEFAULT_ACCOUNT  Cuenta AWS destino (default: se autodetecta con STS)
#
# Salida: exporta a scripts/.env las variables DATA_BUCKET_NAME y
# TRAINING_ROLE_ARN, usadas por los scripts siguientes.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export AWS_REGION="${AWS_REGION:-us-west-2}"
export CDK_DEFAULT_ACCOUNT="${CDK_DEFAULT_ACCOUNT:-$(aws sts get-caller-identity --query Account --output text)}"
export CDK_DEFAULT_REGION="${AWS_REGION}"

echo "== Desplegando infraestructura base en cuenta ${CDK_DEFAULT_ACCOUNT}, region ${AWS_REGION} =="

cd "${ROOT_DIR}/infra"
source .venv/bin/activate

# Nota: la region de despliegue esta fijada dentro de app.py (REGION =
# "us-west-2"), porque Bedrock AgentCore Runtime solo esta disponible en un
# subconjunto de regiones. Si necesitas otra region, edita infra/app.py.
ACTUAL_REGION=$(python3 -c "import re; print(re.search(r'REGION = \"([^\"]+)\"', open('app.py').read()).group(1))")
if [ "${ACTUAL_REGION}" != "${AWS_REGION}" ]; then
  echo "AVISO: infra/app.py tiene fijada la region '${ACTUAL_REGION}', distinta de AWS_REGION='${AWS_REGION}'."
  echo "Se usara la region fijada en app.py (${ACTUAL_REGION}). Edita infra/app.py si necesitas cambiarla."
  AWS_REGION="${ACTUAL_REGION}"
fi

echo "-- cdk bootstrap --"
cdk bootstrap "aws://${CDK_DEFAULT_ACCOUNT}/${AWS_REGION}"

echo "-- cdk deploy SlmDataPipelineStack --"
cdk deploy SlmDataPipelineStack --require-approval never

echo "-- cdk deploy SlmTrainingStack --"
cdk deploy SlmTrainingStack --require-approval never

DATA_BUCKET_NAME=$(aws cloudformation describe-stacks \
  --stack-name SlmDataPipelineStack --region "${AWS_REGION}" \
  --query "Stacks[0].Outputs[?OutputKey=='DataBucketName'].OutputValue" --output text)

TRAINING_ROLE_ARN=$(aws cloudformation describe-stacks \
  --stack-name SlmTrainingStack --region "${AWS_REGION}" \
  --query "Stacks[0].Outputs[?OutputKey=='TrainingExecutionRoleArn'].OutputValue" --output text)

mkdir -p "${SCRIPT_DIR}"
cat > "${SCRIPT_DIR}/.env" <<EOF
AWS_REGION=${AWS_REGION}
CDK_DEFAULT_ACCOUNT=${CDK_DEFAULT_ACCOUNT}
DATA_BUCKET_NAME=${DATA_BUCKET_NAME}
TRAINING_ROLE_ARN=${TRAINING_ROLE_ARN}
EOF

echo ""
echo "== Listo =="
echo "Bucket de datos:        ${DATA_BUCKET_NAME}"
echo "Rol de entrenamiento:   ${TRAINING_ROLE_ARN}"
echo "Variables guardadas en: ${SCRIPT_DIR}/.env"
