#!/usr/bin/env bash
# ============================================================================
# 00_check_prerequisites.sh
# ----------------------------------------------------------------------------
# Verifica que las herramientas necesarias esten instaladas antes de
# reproducir el pipeline en una cuenta nueva:
#   - AWS CLI configurado con credenciales validas
#   - Python 3.10+
#   - Node.js + AWS CDK CLI
#   - Docker (o Colima en macOS) corriendo, necesario para construir la
#     imagen ARM64 del agente en el paso de AgentCore Runtime
#   - Acceso a modelos de Amazon Bedrock (Claude Haiku) en la region elegida
#
# No modifica nada; solo informa y aborta si falta algo critico.
# ============================================================================
set -euo pipefail

echo "== Verificando prerequisitos =="

command -v aws >/dev/null 2>&1 || { echo "ERROR: AWS CLI no encontrado. Instala https://aws.amazon.com/cli/"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 no encontrado."; exit 1; }
command -v node >/dev/null 2>&1 || { echo "ERROR: node no encontrado. Instala Node.js (requerido por AWS CDK)."; exit 1; }
command -v cdk >/dev/null 2>&1 || { echo "ERROR: AWS CDK CLI no encontrado. Instala con: npm install -g aws-cdk"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker no encontrado. En macOS instala Docker Desktop o Colima (brew install colima docker docker-buildx)."; exit 1; }

if ! aws sts get-caller-identity >/dev/null 2>&1; then
  echo "ERROR: No se pudo verificar la identidad de AWS. Configura credenciales con 'aws configure' o variables de entorno."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker no esta corriendo. Si usas Colima: colima start --arch aarch64 --vm-type=vz --cpu 4 --memory 8 --disk 60"
  exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
USER_ARN=$(aws sts get-caller-identity --query Arn --output text)
echo "Cuenta AWS detectada: ${ACCOUNT_ID}"
echo "Identidad: ${USER_ARN}"

PYTHON_VERSION=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
echo "Python: ${PYTHON_VERSION}"
echo "CDK: $(cdk --version)"
echo "Docker: $(docker --version)"

echo ""
echo "== Recordatorios importantes antes de continuar =="
echo "1. Este pipeline usa Amazon Bedrock (modelo Claude Haiku) para generar el"
echo "   dataset de instruccion. Verifica que el acceso a modelos este habilitado"
echo "   en la region elegida: consola de Bedrock > Model access."
echo "2. El fine-tuning usa una instancia ml.g6.2xlarge (GPU) en SageMaker."
echo "   Verifica la cuota de servicio 'ml.g6.2xlarge for training job usage' >= 1"
echo "   en Service Quotas para la region elegida. Si es 0, solicita un aumento"
echo "   (puede tardar minutos a horas en aprobarse)."
echo "3. AWS Bedrock AgentCore Runtime solo esta disponible en un subconjunto de"
echo "   regiones (us-east-1, us-west-2, eu-central-1, etc). Consulta:"
echo "   https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-regions.html"
echo ""
echo "Prerequisitos OK."
