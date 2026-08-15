"""
Agente de prueba para AgentCore Runtime: SLM de cumplimiento regulatorio
CNBV/Banxico (Qwen2.5-1.5B-Instruct + adaptador QLoRA).

Al iniciar, carga el modelo base Qwen2.5-1.5B-Instruct desde Hugging Face Hub
y le aplica el adaptador LoRA entrenado (descargado desde el bucket S3
indicado en las variables de entorno MODEL_BUCKET / MODEL_S3_PREFIX) usando
`peft.PeftModel`. Expone un unico entrypoint HTTP a traves de
`bedrock_agentcore.BedrockAgentCoreApp`, que es el contrato que espera
Bedrock AgentCore Runtime.

Este archivo se empaqueta como imagen de contenedor (ver Dockerfile) y se
publica via el construct `agentcore.Runtime` (AgentRuntimeArtifact.from_asset)
en infra/stacks/agent_runtime_stack.py.
"""
import os
import logging

import boto3
from bedrock_agentcore import BedrockAgentCoreApp
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cnbv_banxico_slm_agent")

MODEL_BUCKET = os.environ.get("MODEL_BUCKET")
MODEL_S3_PREFIX = os.environ.get("MODEL_S3_PREFIX", "models/latest/")
LOCAL_MODEL_DIR = "/opt/ml_model"
BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")

SYSTEM_PROMPT = (
    "Eres un asistente especializado en cumplimiento regulatorio financiero "
    "mexicano (CNBV y Banxico). Ayudas a evaluar carpetas de cumplimiento y a "
    "estructurar documentos regulatorios de forma precisa y concisa, citando "
    "el fundamento normativo cuando sea posible."
)

app = BedrockAgentCoreApp()

_model = None
_tokenizer = None


def _download_adapter_from_s3():
    """Descarga el adaptador LoRA (y tokenizer asociado) del fine-tuning
    QLoRA desde S3 si esta configurado. Si no hay bucket configurado
    (p.ej. smoke test local), se usa el modelo base sin fine-tuning.

    El artefacto en S3 contiene UNICAMENTE el adaptador LoRA (no los pesos
    completos del modelo base fusionados): adapter_model.safetensors,
    adapter_config.json, tokenizer.json, tokenizer_config.json,
    chat_template.jinja. Esto evita duplicar en S3 los ~3GB de pesos del
    modelo base, que de todas formas se descargan de Hugging Face Hub."""
    if not MODEL_BUCKET:
        logger.warning(
            "MODEL_BUCKET no configurado; se usara el modelo base %s sin fine-tuning",
            BASE_MODEL_ID,
        )
        return False

    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    found = False
    for page in paginator.paginate(Bucket=MODEL_BUCKET, Prefix=MODEL_S3_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel_path = key[len(MODEL_S3_PREFIX):]
            if not rel_path:
                continue
            found = True
            dest_path = os.path.join(LOCAL_MODEL_DIR, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            logger.info("Descargando s3://%s/%s -> %s", MODEL_BUCKET, key, dest_path)
            s3.download_file(MODEL_BUCKET, key, dest_path)
    return found


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return

    has_adapter = _download_adapter_from_s3()

    logger.info("Cargando modelo base: %s", BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype="auto")

    if has_adapter:
        logger.info("Cargando tokenizer y adaptador LoRA desde %s", LOCAL_MODEL_DIR)
        _tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
        _model = PeftModel.from_pretrained(base_model, LOCAL_MODEL_DIR)
    else:
        logger.info("Cargando tokenizer base (sin adaptador fine-tuneado)")
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        _model = base_model


@app.entrypoint
def invoke(payload: dict) -> dict:
    """Entrypoint invocado por Bedrock AgentCore Runtime.

    payload esperado: {"prompt": "<texto del usuario>"}
    """
    _load_model()

    user_prompt = payload.get("prompt", "")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer([text], return_tensors="pt")
    generated = _model.generate(**inputs, max_new_tokens=512)
    output_ids = generated[0][inputs["input_ids"].shape[-1]:]
    response_text = _tokenizer.decode(output_ids, skip_special_tokens=True)

    return {"result": response_text}


if __name__ == "__main__":
    app.run()
