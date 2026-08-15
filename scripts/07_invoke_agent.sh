#!/usr/bin/env bash
# ============================================================================
# 07_invoke_agent.sh
# ----------------------------------------------------------------------------
# Invoca el AgentCore Runtime de prueba con un prompt dado y muestra la
# respuesta. Util para validar el despliegue end-to-end.
#
# Uso:
#   ./07_invoke_agent.sh "Tu pregunta sobre cumplimiento regulatorio aqui"
#
# NOTA sobre latencia: la primera invocacion (o cualquiera que dispare un
# cold start de un nuevo contenedor) puede tardar 60-130+ segundos, porque
# el agente descarga el modelo base desde Hugging Face Hub y el adaptador
# LoRA desde S3, y corre inferencia sobre CPU (sin cuantizar). Esto es
# esperado en este entorno de prueba; ver docs/portability_mx_central_1.md
# para la ruta de optimizacion (cuantizacion GGUF/OpenVINO) de cara a un
# despliegue productivo.
#
# Requiere: scripts/.env con AGENT_RUNTIME_ARN y AWS_REGION.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${SCRIPT_DIR}/.env" ]; then
  source "${SCRIPT_DIR}/.env"
fi
: "${AGENT_RUNTIME_ARN:?Debes correr 06_deploy_agent_runtime.sh primero o exportar AGENT_RUNTIME_ARN manualmente}"
: "${AWS_REGION:=us-west-2}"

PROMPT="${1:-¿Qué obligaciones tiene una institución de crédito respecto a la conservación de registros de instrucciones de transferencia?}"
SESSION_ID="cli-session-$(date +%s)000000000000000000"  # AgentCore exige >=33 caracteres

PYTHON="${ROOT_DIR}/data_pipeline/.venv/bin/python"

echo "== Invocando AgentCore Runtime =="
echo "Prompt: ${PROMPT}"
echo "(puede tardar 60-130+ segundos en cold start; esperando...)"
echo ""

"${PYTHON}" - "${AGENT_RUNTIME_ARN}" "${AWS_REGION}" "${SESSION_ID}" "${PROMPT}" <<'PYEOF'
import sys, json, time
import boto3
from botocore.config import Config

agent_runtime_arn, region, session_id, prompt = sys.argv[1:5]

config = Config(read_timeout=900, connect_timeout=60, retries={"max_attempts": 0})
client = boto3.client("bedrock-agentcore", region_name=region, config=config)

payload = json.dumps({"prompt": prompt})

t0 = time.time()
response = client.invoke_agent_runtime(
    agentRuntimeArn=agent_runtime_arn,
    runtimeSessionId=session_id,
    payload=payload,
    qualifier="DEFAULT",
)
body = response["response"].read()
elapsed = time.time() - t0

print(f"Tiempo de respuesta: {elapsed:.1f}s")
print("Respuesta:")
try:
    parsed = json.loads(body)
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
except json.JSONDecodeError:
    print(body.decode("utf-8", errors="replace"))
PYEOF
