# Fine-tuning de un SLM para cumplimiento regulatorio CNBV / Banxico

Elaborado para el **AWS Summit Ciudad de México 2026**.

Pipeline de punta a punta para hacer fine-tuning de un **Small Language Model (SLM)**
especializado en cumplimiento regulatorio mexicano (CNBV y Banxico), pensado para dar soporte
a agentes que evalúan carpetas de cumplimiento y estructuran documentos regulatorios a **baja
latencia y bajo costo**, con portabilidad a una región sin GPU (`mx-central-1`).

> Los documentos usados son información pública que CNBV y Banxico publican en cumplimiento de
> sus obligaciones de transparencia. Ver el aviso completo en la introducción del notebook.

## Punto de entrada

El notebook **[`entrenamiento_slm_cnbv_banxico.ipynb`](./entrenamiento_slm_cnbv_banxico.ipynb)**
es la referencia principal del proyecto: documenta y reproduce, sección por sección, todo el
flujo (infraestructura, datos, entrenamiento, despliegue, portabilidad), con teoría, comandos
reales y análisis de artefactos ya generados. Empieza ahí.

Para reproducir todo el pipeline desde la terminal en lugar del notebook, usa los scripts en
[`scripts/`](./scripts/README.md) (`./scripts/run_all.sh`).

## Arquitectura del pipeline

```
┌──────────────────────┐     ┌──────────────────────┐
│   CNBV (9 sectores)   │     │   Banxico (~300       │
│   normatividad web    │     │   circulares)         │
└──────────┬────────────┘     └──────────┬────────────┘
           │ scrape_cnbv.py              │ scrape_banxico.py
           ▼                             ▼
   ┌─────────────────────────────────────────────┐
   │   raw/{cnbv,banxico}/  (PDFs + metadata)     │   ← datalake S3 (capa bronze)
   └───────────────────┬───────────────────────────┘
                        │ extract_text.py + chunk_documents.py
                        ▼
   ┌─────────────────────────────────────────────┐
   │   processed/text/, processed/chunks.jsonl    │   ← capa silver
   └───────────────────┬───────────────────────────┘
                        │ build_dataset.py (Bedrock: Claude Haiku 4.5)
                        ▼
   ┌─────────────────────────────────────────────┐
   │   datasets/<job>/{train,eval}.jsonl          │   ← capa gold
   └───────────────────┬───────────────────────────┘
                        │ SageMaker Training Job (QLoRA, ml.g6.2xlarge)
                        ▼
   ┌─────────────────────────────────────────────┐
   │   models/<job>/  (adaptador LoRA + métricas) │
   └───────────────────┬───────────────────────────┘
                        │ imagen ARM64 + agentcore.Runtime (CDK)
                        ▼
   ┌─────────────────────────────────────────────┐
   │   Bedrock AgentCore Runtime                  │   ← pruebas end-to-end
   └───────────────────────────────────────────────┘
```

Un Step Function semanal (`SlmDocumentSyncStack`) mantiene sincronizado el datalake contra las
fuentes públicas y dispara la preparación de datos cuando hay documentos nuevos o actualizados,
permitiendo re-entrenar el modelo de forma continua. Ver `docs/architecture.md` y la sección
1.1 del notebook para el detalle de las capas del datalake y ese ciclo continuo.

## Estructura del repositorio

| Carpeta | Rol |
|---|---|
| `entrenamiento_slm_cnbv_banxico.ipynb` | Notebook principal: teoría + reproducción + análisis de todo el pipeline |
| `infra/` | AWS CDK (Python): `DataPipelineStack`, `TrainingStack`, `AgentRuntimeStack`, `DocumentSyncStack` |
| `data_pipeline/scraping/` | Scrapers de CNBV y Banxico + utilidad de certificados TLS |
| `data_pipeline/processing/` | Extracción de texto, chunking y generación del dataset (SFT vía Bedrock) |
| `data_pipeline/sync/` | Sincronización semanal del datalake contra las fuentes públicas |
| `data_pipeline/catalog_lambda/` | Lambda que mantiene el catálogo de documentos en DynamoDB |
| `training/` | Lanzador del training job (`launch_training_job.py`) y entry point QLoRA (`source/`) |
| `agent_runtime/` | Agente servido en Bedrock AgentCore Runtime (imagen de contenedor) |
| `scripts/` | Scripts de reproducción end-to-end paso a paso, ver [`scripts/README.md`](./scripts/README.md) |
| `docs/` | Decisiones de arquitectura (`architecture.md`) y guía de portabilidad a `mx-central-1` |

## Requisitos

- Python 3.10+ y Jupyter (`pip install -r requirements.txt`) para el notebook.
- Node.js + AWS CDK CLI y Docker (para desplegar `infra/`).
- Credenciales de AWS con acceso a S3, DynamoDB, Bedrock, SageMaker, ECS/Fargate y AgentCore.
- Cuota de `ml.g6.2xlarge for training job usage` ≥ 1 en Service Quotas (SageMaker) para el
  fine-tuning real.

Detalle completo de prerrequisitos y variables de entorno en
[`scripts/README.md`](./scripts/README.md).

## Advertencia de costo

Reproducir el pipeline completo crea recursos reales que generan costo: llamadas a Amazon
Bedrock, una instancia GPU de SageMaker, y Bedrock AgentCore Runtime. Revisa
[`scripts/README.md`](./scripts/README.md) para el detalle de costo aproximado y cómo acotar
el alcance.

## Documentación adicional

- [`docs/architecture.md`](./docs/architecture.md) — decisiones de arquitectura y justificación
  del modelo base, la técnica de fine-tuning y las regiones usadas.
- [`docs/portability_mx_central_1.md`](./docs/portability_mx_central_1.md) — cómo portar el
  modelo entrenado a una región sin GPU (Intel/ARM) vía GGUF u OpenVINO.
- [`infra/README.md`](./infra/README.md) — comandos básicos de CDK para este proyecto.
