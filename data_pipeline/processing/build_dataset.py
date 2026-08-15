"""
Generacion del dataset de instruccion (SFT) para el fine-tuning QLoRA de
Qwen2.5-1.5B-Instruct, a partir de los chunks de texto de documentos CNBV y
Banxico (ver chunk_documents.py).

Para cada chunk seleccionado, se invoca un modelo de Amazon Bedrock
(Claude Haiku 4.5, us-west-2) para generar 1-2 pares instruccion-respuesta
en espanol de Mexico, cubriendo las tres tareas del caso de uso:
  1. Resumen estructurado de la disposicion/circular (objeto, sujetos
     obligados, plazos, sanciones).
  2. Extraccion de obligaciones de cumplimiento en formato de checklist.
  3. Clasificacion del fragmento (sector/tipo de norma) con justificacion.

El resultado se guarda en formato JSONL de chat (system/user/assistant),
compatible con `trl.SFTTrainer` / `transformers` chat templates, dividido en
train.jsonl (90%) y eval.jsonl (10%).

Para controlar el volumen y evitar sobre-representar documentos muy largos
(p.ej. la Circular Unica de Bancos, con cientos de chunks), se aplica un cap
de chunks por documento original.

Uso:
    python build_dataset.py --chunks ../processed/chunks.jsonl \
        --out-dir ../processed/dataset --max-chunks-per-doc 10 --region us-west-2
"""
import argparse
import json
import logging
import os
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_dataset")

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

SYSTEM_PROMPT_SLM = (
    "Eres un asistente especializado en cumplimiento regulatorio financiero "
    "mexicano (CNBV y Banxico). Ayudas a evaluar carpetas de cumplimiento y a "
    "estructurar documentos regulatorios de forma precisa y concisa, citando "
    "el fundamento normativo cuando sea posible."
)

GENERATOR_PROMPT_TEMPLATE = """Eres un experto en regulacion financiera mexicana (CNBV y Banco de Mexico). \
A partir del siguiente fragmento de un documento normativo ({source}, archivo: {original_pdf}), \
genera exactamente {num_examples} ejemplos de entrenamiento para un asistente de IA especializado \
en cumplimiento regulatorio. Cada ejemplo debe tener una "instruction" (una tarea realista que un \
analista de cumplimiento le pediria al asistente sobre ESTE fragmento especifico) y una "response" \
(la respuesta completa y precisa del asistente, basada UNICAMENTE en la informacion del fragmento).

Varia el tipo de tarea entre estas categorias, eligiendo la(s) mas apropiada(s) para el contenido \
del fragmento:
- Resumen estructurado: objeto de la disposicion, sujetos obligados, plazos y sanciones aplicables.
- Extraccion de obligaciones de cumplimiento en formato de lista/checklist accionable.
- Clasificacion regulatoria: a que sector aplica y que tipo de norma es, con justificacion breve.
- Explicacion en lenguaje sencillo de un requisito especifico del fragmento.

Reglas:
- Todo en espanol de Mexico.
- La respuesta debe basarse solo en el fragmento proporcionado; no inventes datos que no esten ahi.
- Si el fragmento es muy tecnico o fragmentario (p.ej. solo una tabla o definiciones sueltas), genera \
una instruccion que tenga sentido con ese contenido (p.ej. "explica estos terminos" o "estructura esta \
tabla").
- Responde UNICAMENTE con un array JSON valido de {num_examples} objetos con las claves "instruction" \
y "response". No incluyas texto adicional antes o despues del JSON.

Fragmento:
---
{chunk_text}
---
"""


def load_chunks(chunks_path: str) -> list:
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def select_balanced_chunks(chunks: list, max_chunks_per_doc: int, max_total: int, seed: int = 42) -> list:
    by_doc = defaultdict(list)
    for c in chunks:
        by_doc[c["original_pdf"]].append(c)

    rng = random.Random(seed)
    selected = []
    for doc, doc_chunks in by_doc.items():
        rng.shuffle(doc_chunks)
        selected.extend(doc_chunks[:max_chunks_per_doc])

    rng.shuffle(selected)
    if max_total is not None and len(selected) > max_total:
        selected = selected[:max_total]
    return selected


def call_bedrock_generate(
    client, chunk: dict, num_examples: int, model_id: str, max_retries: int = 4
) -> list:
    prompt = GENERATOR_PROMPT_TEMPLATE.format(
        source=chunk["source"],
        original_pdf=chunk["original_pdf"],
        num_examples=num_examples,
        chunk_text=chunk["text"][:6000],
    )

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "temperature": 0.4,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }

    for attempt in range(max_retries):
        try:
            resp = client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(resp["body"].read())
            text = payload["content"][0]["text"]
            text = text.strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = text[4:]
            examples = json.loads(text)
            if isinstance(examples, dict):
                examples = [examples]
            return [
                e
                for e in examples
                if isinstance(e, dict) and "instruction" in e and "response" in e
            ]
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in ("ThrottlingException", "TooManyRequestsException"):
                wait = 2 ** attempt
                logger.warning("Throttled, reintentando en %ds (intento %d)", wait, attempt + 1)
                time.sleep(wait)
                continue
            logger.error("Error de Bedrock: %s", exc)
            return []
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.warning("Respuesta no parseable como JSON (intento %d): %s", attempt + 1, exc)
            time.sleep(1)
            continue
    return []


def to_chat_record(instruction: str, response: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_SLM},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    }


def run(
    chunks_path: str,
    out_dir: str,
    max_chunks_per_doc: int,
    max_total_chunks: int,
    num_examples_per_chunk: int,
    region: str,
    eval_fraction: float,
    seed: int,
    max_workers: int = 8,
    model_id: str = MODEL_ID,
):
    os.makedirs(out_dir, exist_ok=True)
    chunks = load_chunks(chunks_path)
    logger.info("Chunks totales disponibles: %d", len(chunks))

    selected = select_balanced_chunks(chunks, max_chunks_per_doc, max_total_chunks, seed=seed)
    logger.info("Chunks seleccionados para generacion: %d", len(selected))

    # Un cliente boto3 por hilo evita compartir el mismo objeto de conexion
    # HTTP entre threads.
    thread_local = threading.local()

    def get_client():
        if not hasattr(thread_local, "client"):
            thread_local.client = boto3.client("bedrock-runtime", region_name=region)
        return thread_local.client

    def process_chunk(chunk):
        client = get_client()
        examples = call_bedrock_generate(client, chunk, num_examples_per_chunk, model_id)
        return chunk, examples

    all_records = []
    generation_log_path = os.path.join(out_dir, "generation_log.jsonl")
    processed_count = 0
    lock = threading.Lock()

    with open(generation_log_path, "w", encoding="utf-8") as log_f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_chunk, chunk): chunk for chunk in selected}

            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    _, examples = future.result()
                except Exception as exc:
                    logger.error("Error procesando chunk %s: %s", chunk["original_pdf"], exc)
                    examples = []

                with lock:
                    log_f.write(
                        json.dumps(
                            {
                                "chunk_original_pdf": chunk["original_pdf"],
                                "chunk_index": chunk["chunk_index"],
                                "num_examples_generated": len(examples),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    log_f.flush()

                    for ex in examples:
                        record = to_chat_record(ex["instruction"], ex["response"])
                        record["_source"] = chunk["source"]
                        record["_original_pdf"] = chunk["original_pdf"]
                        all_records.append(record)

                    processed_count += 1
                    if processed_count % 50 == 0:
                        logger.info(
                            "Progreso: %d/%d chunks procesados, %d ejemplos generados hasta ahora",
                            processed_count,
                            len(selected),
                            len(all_records),
                        )

    logger.info("Generacion finalizada. Total de ejemplos: %d", len(all_records))

    rng = random.Random(seed)
    rng.shuffle(all_records)
    split_idx = int(len(all_records) * (1 - eval_fraction))
    train_records = all_records[:split_idx]
    eval_records = all_records[split_idx:]

    train_path = os.path.join(out_dir, "train.jsonl")
    eval_path = os.path.join(out_dir, "eval.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            clean = {"messages": r["messages"]}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for r in eval_records:
            clean = {"messages": r["messages"]}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    with open(os.path.join(out_dir, "dataset_with_metadata.jsonl"), "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(
        "Dataset escrito: train=%d ejemplos (%s), eval=%d ejemplos (%s)",
        len(train_records),
        train_path,
        len(eval_records),
        eval_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Generador de dataset SFT CNBV/Banxico")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-chunks-per-doc", type=int, default=10)
    parser.add_argument("--max-total-chunks", type=int, default=None)
    parser.add_argument("--num-examples-per-chunk", type=int, default=2)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--model-id", default=MODEL_ID, help="Modelo de Bedrock usado para generar el dataset")
    args = parser.parse_args()

    run(
        chunks_path=args.chunks,
        out_dir=args.out_dir,
        max_chunks_per_doc=args.max_chunks_per_doc,
        max_total_chunks=args.max_total_chunks,
        num_examples_per_chunk=args.num_examples_per_chunk,
        region=args.region,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        max_workers=args.max_workers,
        model_id=args.model_id,
    )


if __name__ == "__main__":
    main()
