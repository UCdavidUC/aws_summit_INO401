#!/usr/bin/env bash
# ============================================================================
# run_all.sh
# ----------------------------------------------------------------------------
# Orquesta la reproduccion completa del pipeline end-to-end, en orden:
#   00 - Verifica prerequisitos (AWS CLI, CDK, Docker, credenciales)
#   01 - Crea entornos virtuales de Python
#   02 - Despliega infraestructura base (bucket S3 + rol de SageMaker)
#   03 - Descarga documentos de CNBV/Banxico y los sube a S3
#   04 - Procesa los documentos y genera el dataset de fine-tuning
#   05 - Lanza y espera el training job de QLoRA en SageMaker
#   06 - Despliega el AgentCore Runtime de prueba con el modelo entrenado
#   07 - Invoca el agente con un prompt de ejemplo para validar
#
# ADVERTENCIA DE COSTO: este script lanza recursos reales que generan costo
# en tu cuenta de AWS (Amazon Bedrock, SageMaker Training Job en instancia
# GPU, S3, Bedrock AgentCore Runtime). El costo total aproximado observado
# en la ejecucion original fue de unos pocos dolares (~$2-5 USD), pero puede
# variar segun la region y el volumen de datos. Revisa cada script antes de
# ejecutar si quieres ajustar el alcance (p.ej. limitar sectores del
# scraping, o el numero de chunks usados para generar el dataset).
#
# Uso:
#   ./run_all.sh                 # corre todo el pipeline
#   ./run_all.sh --from 04       # reanuda desde el paso 04 (util si algo
#                                  fallo a mitad de camino y ya tienes
#                                  scripts/.env con las variables previas)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FROM_STEP="00"
if [ "${1:-}" == "--from" ]; then
  FROM_STEP="${2:?Especifica el numero de paso, p.ej. --from 04}"
fi

STEPS=(
  "00_check_prerequisites.sh"
  "01_setup_envs.sh"
  "02_deploy_data_and_training_stacks.sh"
  "03_scrape_and_upload_raw_data.sh"
  "04_process_and_build_dataset.sh"
  "05_run_finetuning_job.sh"
  "06_deploy_agent_runtime.sh"
)

for step_script in "${STEPS[@]}"; do
  step_num="${step_script:0:2}"
  if [[ "${step_num}" < "${FROM_STEP}" ]]; then
    echo "Saltando ${step_script} (antes de --from ${FROM_STEP})"
    continue
  fi
  echo ""
  echo "############################################################"
  echo "## Ejecutando ${step_script}"
  echo "############################################################"
  bash "${SCRIPT_DIR}/${step_script}"
done

echo ""
echo "############################################################"
echo "## Pipeline completo. Probando el agente desplegado..."
echo "############################################################"
bash "${SCRIPT_DIR}/07_invoke_agent.sh"

echo ""
echo "== Reproduccion completa =="
echo "Revisa scripts/.env para ver todos los recursos creados (bucket, rol, job, runtime)."
