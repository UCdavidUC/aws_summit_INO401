# Arquitectura: Fine-tuning de un SLM para cumplimiento regulatorio CNBV/Banxico

## 1. Objetivo

Entrenar (fine-tuning) un Small Language Model (SLM) especializado en el dominio de
cumplimiento regulatorio de la CNBV (Comisión Nacional Bancaria y de Valores) y Banxico
(Banco de México), para soportar agentes que evalúen carpetas de cumplimiento y estructuren
documentos regulatorios. El modelo resultante debe ser portable a la región `mx-central-1`
(que solo dispone de cómputo Intel x86 y ARM/Graviton, sin GPU).

## 2. Decisiones clave y justificación

### 2.1 Modelo base: Qwen2.5-1.5B-Instruct

| Criterio | Qwen2.5-1.5B-Instruct | Alternativas descartadas |
|---|---|---|
| Licencia | Apache-2.0 (uso comercial libre) | Qwen2.5-3B/7B-Instruct usan "Qwen Research License" (no comercial); Llama-3.2-1B/3B usan licencia comunitaria gated (requiere aceptación en HF, fricción de acceso, cláusulas de nombrado) |
| Idiomas | 29 idiomas incl. español | Salamandra-2B (BSC) está más enfocado a español/catalán pero su ecosistema de cuantización ARM/Intel es menos maduro |
| Tamaño | 1.54B params (~3GB en fp16, ~1GB en int4) | Modelos >3B complican la meta de baja latencia en CPU sin GPU |
| Contexto | 32K tokens | Suficiente para chunks de circulares/disposiciones |
| Ecosistema de cuantización | GGUF (llama.cpp, con optimizaciones ARM KleidiAI ya probadas en Graviton) e IR de OpenVINO (INT4/INT8) para CPU Intel; el servidor OVMS de Intel soporta nativamente GGUF de Qwen2.5 1.5B/3B/7B | — |
| Fine-tuning | Soporte maduro en `transformers` + `peft` + `trl` + `bitsandbytes` | — |

Fuente: [Qwen2.5-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct),
[Qwen2.5-3B-Instruct license](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) (qwen-research,
no comercial), [Llama 3.2 Community License](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
(gated + cláusulas de atribución), [Salamandra BSC](https://huggingface.co/BSC-LT/salamandra-2b-instruct)
(Apache-2.0, foco en español pero tooling de cuantización ARM menos probado).

### 2.2 Técnica de fine-tuning: QLoRA

- Cuantización NF4 de 4 bits del modelo base + adaptadores LoRA entrenables (rank 16-32) sobre
  capas de atención y MLP.
- Ventajas: reduce memoria de entrenamiento en ~70%, permite entrenar en una sola GPU
  `ml.g6.2xlarge`/`ml.g6.4xlarge` (NVIDIA L4, 24GB), y los adaptadores resultantes son ligeros
  (~10-50MB) para fusionar y re-cuantizar hacia GGUF/OpenVINO después. La familia G6 sustituye
  a la generación anterior G5 (A10G): mismo VRAM, ~15-20% más económica por hora (Sección 17.1
  de `docs/technical_documentation.md`).
- Librerías: `transformers`, `peft`, `trl` (SFTTrainer), `bitsandbytes`, `accelerate`.

### 2.3 Regiones

| Componente | Región | Motivo |
|---|---|---|
| Ingesta y procesamiento de datos | us-west-2 | Junto al resto del pipeline |
| SageMaker Training Job (fine-tuning) | us-west-2 | Cuotas de GPU (`ml.g6.4xlarge`) ya disponibles en la cuenta; AgentCore/Custom Model Import no están en mx-central-1 |
| Despliegue de prueba (Bedrock AgentCore Runtime) | us-west-2 | AgentCore Runtime **no está disponible en mx-central-1** ([tabla de regiones soportadas](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-regions.html): 19 regiones, México Central no incluida) |
| Despliegue futuro en producción (portabilidad) | mx-central-1 | Requisito del negocio (data residency). Solo cuenta con cómputo Intel x86 y ARM Graviton (sin GPU), confirmado por instancias EC2 disponibles en esa región (familias C6in Intel, Graviton) |

Ver `docs/portability_mx_central_1.md` para el detalle de cómo se logra la portabilidad.

## 3. Arquitectura del pipeline

```
┌──────────────────────┐     ┌──────────────────────┐
│   CNBV (9 sectores)   │     │   Banxico (323        │
│   paginas normatividad│     │   circulares, indice   │
│   → PDFs directos     │     │   cronologico)         │
└──────────┬────────────┘     └──────────┬────────────┘
           │  scrape_cnbv.py             │  scrape_banxico.py
           ▼                              ▼
   ┌───────────────────────────────────────────┐
   │   S3: s3://<bucket>/raw/{cnbv,banxico}/    │   (PDFs + metadata.jsonl)
   └───────────────────┬─────────────────────────┘
                        │  extract_text.py (pypdf)
                        ▼
   ┌───────────────────────────────────────────┐
   │   S3: s3://<bucket>/processed/text/        │   (texto limpio por documento)
   └───────────────────┬─────────────────────────┘
                        │  build_dataset.py
                        ▼
   ┌───────────────────────────────────────────┐
   │ S3: s3://<bucket>/datasets/train.jsonl     │   (formato instrucción/chat SFT)
   │                    /datasets/eval.jsonl    │
   └───────────────────┬─────────────────────────┘
                        │  SageMaker Training Job (QLoRA)
                        │  ml.g6.*xlarge, HF Training DLC
                        ▼
   ┌───────────────────────────────────────────┐
   │ S3: s3://<bucket>/models/<job-name>/       │   (adaptadores LoRA + métricas)
   └───────────────────┬─────────────────────────┘
                        │  merge + package
                        ▼
   ┌───────────────────────────────────────────┐
   │  ECR: imagen del agente (Qwen2.5-1.5B      │
   │  + adaptador fusionado, servido con        │
   │  transformers/vLLM en el contenedor)       │
   └───────────────────┬─────────────────────────┘
                        │  agentcore.Runtime (CDK)
                        ▼
   ┌───────────────────────────────────────────┐
   │  Bedrock AgentCore Runtime (us-west-2)     │   ← pruebas end-to-end
   └───────────────────────────────────────────┘
```

## 4. Componentes AWS (CDK)

1. **DataPipelineStack**: buckets S3 (`raw`, `processed`, `datasets`, `models`), roles IAM
   de mínimo privilegio para scripts locales/jobs que leen/escriben en esos prefijos.
2. **TrainingStack**: rol de ejecución de SageMaker (acceso a S3 del pipeline, CloudWatch Logs,
   ECR de solo lectura para el DLC de HuggingFace), definición reutilizable para lanzar el
   training job (el lanzamiento real se hace vía script boto3, no vía recurso CDK, dado que
   un training job es una ejecución puntual, no infraestructura de larga duración).
3. **AgentRuntimeStack**: repositorio ECR para la imagen del agente, `agentcore.Runtime` L2
   construct (IAM auth, red pública, logging a CloudWatch), rol de ejecución con permisos de
   `s3:GetObject` sobre el bucket de modelos.

## 5. Dataset de fine-tuning

Se construyen ejemplos de instrucción en español de México a partir de los documentos, con
tres tipos de tareas (reflejando los casos de uso del negocio):

1. **Resumen estructurado de disposiciones**: dado un fragmento de una circular/disposición,
   generar un resumen estructurado (objeto, sujetos obligados, plazos, sanciones).
2. **Extracción de requisitos de cumplimiento**: dado un fragmento, listar obligaciones
   accionables en formato estructurado (JSON) para checklist de cumplimiento.
3. **Clasificación y enrutamiento**: dado un fragmento, identificar a qué sector/entidad
   regulada aplica y qué tipo de norma es.

Cada ejemplo sigue el formato de chat de Qwen2.5 (`system`/`user`/`assistant`) serializado en
JSONL, dividido en `train.jsonl` (90%) y `eval.jsonl` (10%).

## 6. Métricas capturadas durante el fine-tuning

### 6.1 Entrenamiento single-job (modelo único)

- Duración total del job (`TrainingStartTime` → `TrainingEndTime`, vía `describe_training_job`).
- `train_loss` y `eval_loss` por step/epoch (vía CloudWatch Metrics del Training Job).
- Throughput (samples/sec, tokens/sec).
- Costo estimado (tipo de instancia × duración × precio on-demand).
- Tamaño del adaptador LoRA resultante.
- Memoria pico de GPU (`torch.cuda.max_memory_allocated`/`max_memory_reserved`) y memoria
  residente (RSS) del proceso de entrenamiento antes de cargar el modelo y después de
  entrenar (vía `psutil`), capturadas dentro de `training/source/train_qlora.py`.
- Porcentaje de parámetros entrenables (`trainable_params` / `total_params`), específico del
  adaptador LoRA de cada modelo.

### 6.2 Fine-tuning en paralelo (múltiples modelos candidatos)

Para comparar modelos candidatos bajo las mismas condiciones de datos e hiperparámetros, el
proyecto soporta lanzar **un SageMaker Training Job independiente por modelo, todos en
paralelo**, vía `training/launch_training_job.py --model-keys` (usa un `ThreadPoolExecutor`
para crear y esperar los jobs concurrentemente). El catálogo de modelos candidatos
(`training/model_catalog.py`) recoge los modelos evaluados en
`docs/technical_documentation.md` (Secciones 3.3, 6 y 7.1) que además tienen una ruta de
fine-tuning QLoRA/LoRA documentada: **Qwen2.5-1.5B-Instruct** (modelo base actual) y
**Qwen3-0.6B**.

Cada job, además de las métricas de la sección 6.1, reporta:

- Utilización de CPU/memoria/GPU durante el entrenamiento, leída del namespace de CloudWatch
  `/aws/sagemaker/TrainingJobs` (`CPUUtilization`, `MemoryUtilization`, `GPUUtilization`,
  `GPUMemoryUtilization`), promedio y máximo sobre la ventana de entrenamiento.
- Instancia recomendada por modelo (`ml.g6.xlarge` para ambos, modelos ≤1.5B), según la
  Sección 17.1 de `docs/technical_documentation.md`.

El resultado se guarda en `training/metrics_<job_name>.json` por modelo, más un archivo
consolidado `training/parallel_run_<timestamp>.json` con el resumen de la corrida completa
(útil para el análisis comparativo en el notebook, Sección 8).

### 6.3 Evaluación de rendimiento post-entrenamiento (multi-modelo)

Después del entrenamiento, `training/evaluate_models.py` carga cada adaptador LoRA (base +
`peft.PeftModel`) y mide:

- **Calidad**: perplexity sobre `eval.jsonl` (exponencial de la pérdida de entropía cruzada
  promedio), comparable directamente contra el `eval_loss` reportado durante el entrenamiento.
- **Rendimiento de inferencia**: tokens por segundo, tiempo al primer token (TTFT) y latencia
  total de generación (P50/P99), sobre un set fijo de prompts de benchmark de longitud corta,
  media y larga — siguiendo el enfoque de benchmarking de la Sección 16.3 de
  `docs/technical_documentation.md`, aplicado aquí a los adaptadores recién entrenados en
  lugar de a un endpoint ya desplegado en `mx-central-1`.

Los resultados se guardan en `training/eval_metrics_<job_name>.json` por modelo y un resumen
consolidado `training/eval_summary_<timestamp>.json`. El notebook carga estos archivos con
`pandas` y los visualiza con `seaborn` (gráficas comparativas de tiempo de entrenamiento,
memoria, loss, perplexity y tokens/segundo entre modelos) para apoyar la decisión de qué
modelo promover a la ruta de portabilidad (Sección 10 / Sección 10 de
`docs/technical_documentation.md`).

## 7. Riesgos y limitaciones conocidas

- El scraping de CNBV/Banxico depende de la estructura HTML actual de esos sitios; si cambia,
  los scripts de `data_pipeline/scraping/` requerirán ajuste.
- El corpus resultante es de tamaño modesto (cientos de documentos regulatorios), adecuado para
  fine-tuning de estilo/formato y vocabulario de dominio, no para inyectar conocimiento
  exhaustivo — para eso se recomienda complementar con RAG en producción.
- AgentCore Runtime no soporta `mx-central-1`; el despliegue de prueba solicitado se realiza en
  `us-west-2`. Ver `docs/portability_mx_central_1.md`.
