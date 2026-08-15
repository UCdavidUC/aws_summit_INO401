"""
Catalogo de modelos candidatos para el fine-tuning paralelo del SLM de
cumplimiento CNBV/Banxico.

Este catalogo traduce a codigo la Seccion 3 ("Candidate Models") y la
Seccion 7.1 ("Supported Approaches") de `docs/technical_documentation.md`:
de los modelos ahi evaluados para inferencia CPU en `mx-central-1`, se
seleccionan los que ademas tienen una ruta de fine-tuning QLoRA/LoRA
documentada con GPU en `us-west-2` (Secciones 7.1 y 17.1), para poder
entrenarlos **en paralelo** sobre el mismo dataset SFT (Seccion 5 del
notebook) y comparar sus metricas de entrenamiento y de inferencia.

Cada entrada incluye:
  - `model_id`: identificador de Hugging Face Hub.
  - `license`: licencia (todas Apache-2.0 o MIT, aptas para uso comercial;
    ver Seccion 3.2 punto 5 de technical_documentation.md).
  - `lora_target_modules`: modulos objetivo de LoRA, especificos de cada
    arquitectura (los modelos de la familia Qwen usan proyecciones de
    atencion y MLP separadas).
  - `recommended_instance_type`: instancia de SageMaker recomendada para
    QLoRA de ese tamano de modelo (Seccion 17.1 de technical_documentation.md).
    Familia G6 (GPU NVIDIA L4, 24GB): mismo VRAM que G5/A10G pero ~15-20%
    mas barata por hora (verificado via AWS Price List API, us-west-2), y
    es la recomendada por AWS para cargas de trabajo de este tamano desde
    la introduccion de G6 (ver Seccion 17.1, "Cost Optimization Strategies").
  - `cpu_inference_path`: formato/runtime recomendado para el despliegue en
    `mx-central-1` (Seccion 4.3 y Tabla 10.1), documentado aqui solo como
    referencia para la evaluacion post-entrenamiento (Secciones 5-6 de este
    modulo se limitan al entrenamiento; la evaluacion vive en
    `evaluate_models.py`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    license: str
    params_b: float
    lora_target_modules: List[str]
    recommended_instance_type: str
    cpu_inference_path: str
    notes: str = ""


# Modulos objetivo de LoRA por familia de arquitectura.
_QWEN_LLAMA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


MODEL_CATALOG = {
    # Modelo base actual del notebook (Seccion 6.1): punto de comparacion.
    "qwen2.5-1.5b": ModelSpec(
        key="qwen2.5-1.5b",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        license="Apache-2.0",
        params_b=1.5,
        lora_target_modules=_QWEN_LLAMA_TARGETS,
        recommended_instance_type="ml.g6.xlarge",
        cpu_inference_path="GGUF Q4_K_M (llama.cpp)",
        notes="Modelo base actual (notebook secciones 6-9). Referencia de comparacion.",
    ),
    # Seccion 3.3 "Lightweight Option": corre sin cuantizar en CPU, permite
    # comparar contra un modelo aun mas pequeno y barato de entrenar/servir.
    "qwen3-0.6b": ModelSpec(
        key="qwen3-0.6b",
        model_id="Qwen/Qwen3-0.6B",
        license="Apache-2.0",
        params_b=0.6,
        lora_target_modules=_QWEN_LLAMA_TARGETS,
        recommended_instance_type="ml.g6.xlarge",
        cpu_inference_path="Sin cuantizar (transformers) o GGUF Q8_0",
        notes="Seccion 3.3: soporta modo de razonamiento y tool-use pese a su tamano.",
    ),
}


DEFAULT_MODEL_KEYS = ["qwen2.5-1.5b", "qwen3-0.6b"]


def get_model_spec(key: str) -> ModelSpec:
    try:
        return MODEL_CATALOG[key]
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_CATALOG))
        raise KeyError(f"Modelo '{key}' no esta en el catalogo. Validos: {valid}") from exc


def list_model_specs(keys: List[str] | None = None) -> List[ModelSpec]:
    keys = keys or DEFAULT_MODEL_KEYS
    return [get_model_spec(k) for k in keys]


if __name__ == "__main__":
    for spec in list_model_specs():
        print(f"[{spec.key}] {spec.model_id} ({spec.params_b}B, {spec.license})")
        print(f"    lora_target_modules: {spec.lora_target_modules}")
        print(f"    instancia recomendada: {spec.recommended_instance_type}")
        print(f"    ruta de inferencia CPU: {spec.cpu_inference_path}")
        print()
