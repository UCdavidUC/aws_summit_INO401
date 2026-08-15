# Scripts de reproduccion — SLM de cumplimiento CNBV/Banxico

Estos scripts reproducen de punta a punta el pipeline implementado en este
proyecto (scraping → procesamiento → dataset → fine-tuning QLoRA → despliegue
en Bedrock AgentCore Runtime) en **cualquier cuenta de AWS**.

## Requisitos previos

- AWS CLI configurado con credenciales de la cuenta destino (`aws configure`
  o variables de entorno / SSO).
- Permisos IAM suficientes para crear: buckets S3, roles/politicas IAM,
  SageMaker Training Jobs, repositorios ECR, y recursos de Bedrock AgentCore.
- Acceso a modelos de **Amazon Bedrock** habilitado en la region elegida
  (consola de Bedrock → Model access) — se usa Claude Haiku para generar el
  dataset de instruccion.
- Cuota de servicio **`ml.g6.2xlarge for training job usage` ≥ 1** en Service
  Quotas (SageMaker) en la region elegida. Si es 0, solicita el aumento antes
  de correr el paso 05.
- **Bedrock AgentCore Runtime** disponible en la region elegida (por defecto
  `us-west-2`; ver [regiones soportadas](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-regions.html)).
- Python 3.10+, Node.js + AWS CDK CLI (`npm install -g aws-cdk`), y Docker
  corriendo (en macOS, Colima es una alternativa ligera a Docker Desktop:
  `brew install colima docker docker-buildx && colima start --arch aarch64 --vm-type=vz`).

## Uso rapido (todo el pipeline)

```bash
cd scripts
./run_all.sh
```

Esto corre los 7 pasos en orden y al final invoca el agente desplegado con un
prompt de ejemplo. Si algo falla a mitad de camino, puedes reanudar desde un
paso especifico una vez corregido el problema:

```bash
./run_all.sh --from 04
```

## Uso paso a paso

| Script | Que hace | Tiempo aprox. |
|---|---|---|
| `00_check_prerequisites.sh` | Verifica AWS CLI, CDK, Docker, credenciales | segundos |
| `01_setup_envs.sh` | Crea los venvs de Python e instala dependencias | 1-2 min |
| `02_deploy_data_and_training_stacks.sh` | `cdk deploy` del bucket S3 y el rol de SageMaker | 1-2 min |
| `03_scrape_and_upload_raw_data.sh` | Descarga PDFs de CNBV/Banxico y los sube a S3 | 15-25 min |
| `04_process_and_build_dataset.sh` | Extrae texto, chunkea, genera dataset via Bedrock, sube a S3 | 30-45 min |
| `05_run_finetuning_job.sh` | Lanza el training job QLoRA en SageMaker y espera | 60-100 min |
| `05b_run_parallel_finetuning_jobs.sh` | (Opcional) Lanza EN PARALELO un training job por cada modelo del catalogo (`training/model_catalog.py`) sobre el mismo dataset, para comparar tiempo/memoria/loss entre modelos | 60-120 min (jobs simultaneos) |
| `06_deploy_agent_runtime.sh` | Sube el adaptador entrenado y despliega AgentCore Runtime | 3-5 min |
| `06b_evaluate_models.sh` | (Opcional) Evalua calidad (perplexity) y rendimiento (tok/s, latencia) de los modelos de la corrida en paralelo | 10-30 min |
| `07_invoke_agent.sh "<prompt>"` | Invoca el agente desplegado para probarlo | 1-2 min |

### Fine-tuning de multiples modelos en paralelo (opcional)

Como alternativa a `05_run_finetuning_job.sh` (que entrena un unico modelo,
Qwen2.5-1.5B-Instruct), `05b_run_parallel_finetuning_jobs.sh` entrena **varios
modelos candidatos en paralelo** (por defecto: Qwen2.5-1.5B y Qwen3-0.6B;
ver `training/model_catalog.py`), cada uno en su propio SageMaker Training
Job, todos sobre el mismo dataset SFT. Cada job
captura sus propias metricas de tiempo, memoria (GPU y CPU) y loss en
`training/metrics_<job_name>.json`, mas un resumen consolidado en
`training/parallel_run_<timestamp>.json`.

Despues de entrenar, `06b_evaluate_models.sh` carga cada adaptador y mide su
**perplexity** sobre `eval.jsonl` y su **rendimiento de inferencia**
(tokens/segundo, TTFT, latencia P50/P99), escribiendo
`training/eval_metrics_<job_name>.json` por modelo. La seccion de analisis
del notebook (`entrenamiento_slm_cnbv_banxico.ipynb`) carga estos archivos
con `pandas` y los visualiza con `seaborn` para comparar los modelos entre
si.

```bash
cd scripts
MODEL_KEYS=qwen2.5-1.5b,qwen3-0.6b ./05b_run_parallel_finetuning_jobs.sh
./06b_evaluate_models.sh
```

Cada script lee/escribe variables compartidas en `scripts/.env` (bucket,
rol, nombre del job, ARN del runtime), por lo que puedes ejecutarlos de forma
independiente siempre que ese archivo exista con las variables necesarias.

## Variables de entorno configurables

- `AWS_REGION` (default `us-west-2`) — nota: la region tambien esta fijada en
  `infra/app.py`; si la cambias, edita ese archivo tambien.
- `MAX_CHUNKS_PER_DOC`, `NUM_EXAMPLES_PER_CHUNK`, `BEDROCK_MODEL_ID`,
  `MAX_WORKERS` — controlan el tamano/costo de la generacion del dataset
  (paso 04).
- `INSTANCE_TYPE`, `EPOCHS`, `JOB_NAME` — controlan el training job (paso 05).

## Advertencia de costo

Este pipeline crea recursos reales que generan costo: llamadas a Amazon
Bedrock (dataset), una instancia GPU de SageMaker (fine-tuning), y Bedrock
AgentCore Runtime (despliegue de prueba). En la ejecucion original el costo
total fue de unos pocos dolares. Revisa cada script si quieres acotar el
alcance (por ejemplo, reduciendo `--sectors` en el scraping de CNBV o
`MAX_CHUNKS_PER_DOC`/`NUM_EXAMPLES_PER_CHUNK` en el dataset).

## Limpieza de recursos

Estos scripts no incluyen un paso de destrucción automática. Para eliminar
los recursos creados:

```bash
cd ../infra
source .venv/bin/activate
cdk destroy SlmAgentRuntimeStack SlmTrainingStack SlmDataPipelineStack
```

Nota: el bucket S3 (`SlmDataBucket`) tiene `RemovalPolicy.RETAIN`, por lo que
no se borrara automaticamente con `cdk destroy` — debes vaciarlo y borrarlo
manualmente si ya no lo necesitas (`aws s3 rb s3://<bucket> --force`).
