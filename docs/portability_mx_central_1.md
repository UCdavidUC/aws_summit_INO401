# Portabilidad del SLM a la región mx-central-1 (México Central)

## 1. Resumen del problema

El requisito de negocio es que el modelo pueda desplegarse en `mx-central-1`, pero
esta región tiene dos restricciones duras al momento de este diseño:

1. **No hay GPU disponible.** Las familias de instancias EC2 confirmadas en la región
   son Intel (p.ej. C6in) y AWS Graviton (ARM), sin instancias `g4dn`/`g5`/`p*` de GPU.
2. **Bedrock AgentCore Runtime y Bedrock Custom Model Import no están disponibles ahí.**
   Confirmado contra la [tabla oficial de regiones de AgentCore](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-regions.html)
   (19 regiones soportadas, México Central no incluida) y la documentación de
   Custom Model Import (solo us-east-1, us-west-2, eu-central-1).

Por eso el entrenamiento (SageMaker) y el despliegue de prueba (AgentCore Runtime) de
este proyecto se ejecutan en `us-west-2`. Este documento describe cómo llevar el modelo
resultante a `mx-central-1` el día que el negocio lo requiera, usando únicamente cómputo
Intel x86 o ARM (Graviton), sin GPU.

## 2. Por qué el modelo elegido es portable

Qwen2.5-1.5B-Instruct fue elegido, entre otras razones, por tener el ecosistema de
cuantización para CPU más maduro entre los candidatos evaluados (ver `architecture.md`,
sección 2.1):

- **GGUF + llama.cpp**: Qwen2.5 1.5B/3B/7B están en la lista de modelos con soporte
  nativo verificado por OpenVINO Model Server para carga directa de GGUF, y Arm ha
  contribuido optimizaciones especificas (KleidiAI) a llama.cpp que se benefician
  directamente en instancias Graviton.
- **OpenVINO IR (INT4/INT8)**: Intel publica y mantiene versiones pre-cuantizadas de
  Qwen2.5 en formato OpenVINO IR (`OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov`, etc.), y
  OpenVINO es la vía recomendada por Intel para inferencia eficiente en sus CPUs
  (instancias `C6in` de mx-central-1).

Al ser un adaptador LoRA (no un fine-tuning completo de todos los pesos), el artefacto
de salida del training job es pequeño (decenas de MB) y se **fusiona** con el modelo
base antes de cuantizar, generando un único modelo denso listo para GGUF/OpenVINO.

## 3. Pipeline de conversión (post-entrenamiento)

```
Adaptador LoRA (SM_MODEL_DIR)  +  Qwen2.5-1.5B-Instruct (pesos base fp16)
                    │
                    ▼  peft.PeftModel.merge_and_unload()
        Modelo fusionado (fp16, ~3GB)
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
  Ruta ARM (Graviton)      Ruta Intel (C6in)
  llama.cpp convert +      optimum-intel export
  quantize -> GGUF Q4_K_M   -> OpenVINO IR INT4/INT8
  (~900MB-1.1GB)            (~800MB-1GB)
        │                        │
        ▼                        ▼
  llama.cpp / llama-server   OpenVINO Model Server (OVMS)
  o Ollama en EC2 Graviton   o Python + optimum-intel en EC2 Intel
```

Pasos concretos:

1. **Fusionar el adaptador**: `PeftModel.from_pretrained(base_model, adapter_dir).merge_and_unload()`,
   luego `save_pretrained()` para obtener un modelo denso estándar de `transformers`.
2. **Ruta ARM/Graviton (llama.cpp/GGUF)**:
   - Convertir con `convert_hf_to_gguf.py` (del repo `llama.cpp`) a GGUF fp16.
   - Cuantizar con `llama-quantize` a `Q4_K_M` (mejor relación calidad/tamaño según
     benchmarks públicos) u otro nivel según la latencia objetivo.
   - Servir con `llama.cpp`/`llama-server` (con soporte KleidiAI compilado para ARM)
     o con Ollama, sobre una instancia EC2 Graviton (`c7g`/`c8g`) o contenedor en
     ECS/Fargate ARM64.
3. **Ruta Intel (OpenVINO)**:
   - Exportar con `optimum-cli export openvino --model <ruta_modelo_fusionado> --weight-format int4 <salida>`.
   - Servir con OpenVINO Model Server (OVMS) o directamente con `optimum-intel`
     (`OVModelForCausalLM`) en una instancia EC2 Intel (`c6in`) o contenedor.

Ambas rutas producen un artefacto que corre exclusivamente en CPU, sin dependencias de
CUDA/GPU, adecuado para `mx-central-1`.

## 4. Opciones de despliegue en mx-central-1

Dado que AgentCore Runtime no está disponible ahí, las alternativas de hosting dentro
de la región son:

| Opción | Descripción | Cuándo usarla |
|---|---|---|
| **Amazon SageMaker Endpoint (CPU)** | Endpoint de inferencia en tiempo real sobre instancia CPU (`ml.c6i`/`ml.c7g` si SageMaker soporta Graviton ahí) | Si se necesita autoscaling gestionado y integración nativa con el resto de SageMaker |
| **ECS/Fargate o EC2 (Graviton o Intel) con contenedor propio** | Contenedor con `llama.cpp`/OVMS sirviendo el modelo cuantizado detrás de un ALB | Más control sobre latencia/costo; requiere gestionar el contenedor propio |
| **AWS Lambda (contenedor, ARM64)** | Para cargas de baja frecuencia/baja latencia de arranque en frío tolerable | Casos de uso esporádicos, no para tráfico sostenido |

En cualquier caso, la interfaz de invocación (payload de entrada/salida) puede
mantenerse compatible con la usada en `agent_runtime/app.py` para minimizar el cambio
en los agentes que consumen el modelo.

## 5. Qué NO cambia al portar

- El **dataset y el proceso de fine-tuning** (SageMaker + QLoRA en us-west-2) no
  necesitan repetirse por región; el adaptador entrenado una vez se reutiliza para
  generar los artefactos GGUF/OpenVINO de cualquier región.
- El **contrato de la API del agente** (prompt del sistema, formato de entrada/salida)
  se mantiene igual; solo cambia el runtime de inferencia subyacente.

## 6. Qué SÍ cambia

- Se pierde la gestión totalmente administrada de AgentCore Runtime (sesiones,
  escalado automático, observabilidad integrada); estas capacidades deberán
  reimplementarse parcialmente si se requieren en `mx-central-1` (p.ej. logging propio
  a CloudWatch, un ALB con health checks, etc.).
- La latencia y el throughput serán distintos a los observados en GPU. En las pruebas
  de este proyecto, el runtime de AgentCore en `us-west-2` sirviendo el modelo sin
  cuantizar (bf16, vía `transformers` puro) sobre la CPU subyacente tomó entre 77 y 130
  segundos por invocación en frío (carga de modelo incluida). Cuantizado a 4 bits con
  GGUF/OpenVINO —no probado en este proyecto, ver sección 3— la literatura pública de
  Arm sobre llama.cpp reporta del orden de decenas de tokens/segundo en generación con
  batch=1 para modelos de tamaño similar en Graviton3/4, lo que reduciría
  significativamente esa latencia. Esto debe validarse empíricamente con carga real en
  `mx-central-1` antes de considerarse definitivo.

## 7. Fuentes

- [Supported AWS Regions - Amazon Bedrock AgentCore](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-regions.html)
- [Amazon Bedrock Custom Model Import — regiones soportadas](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-import-model.html)
- [Loading GGUF models in OVMS — modelos soportados incluyen Qwen2.5 1.5B/3B/7B](https://docs.openvino.ai/nightly/model-server/ovms_demos_gguf.html)
- [Arm Learning Paths — llama.cpp en AWS Graviton, optimizaciones KleidiAI](https://learn.arm.com/learning-paths/servers-and-cloud-computing/llama_cpp_streamline/)
- [OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov en Hugging Face](https://huggingface.co/OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov)
