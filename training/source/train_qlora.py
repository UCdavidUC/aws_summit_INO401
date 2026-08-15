"""
Entry point de SageMaker Training Job para el fine-tuning QLoRA de un SLM
sobre el corpus de cumplimiento regulatorio de CNBV/Banxico.

Este script se ejecuta DENTRO del contenedor del SageMaker HuggingFace
Training DLC (ver launch_training_job.py). Espera los datasets de
entrenamiento y evaluacion en formato JSONL de chat (claves "messages")
montados en los canales de SageMaker (SM_CHANNEL_TRAIN, SM_CHANNEL_EVAL).

Tecnica: QLoRA (cuantizacion NF4 de 4 bits del modelo base + adaptadores
LoRA entrenables sobre proyecciones de atencion y MLP), usando
transformers + peft + trl (SFTTrainer) + bitsandbytes.

El script es agnostico al modelo base: `--model-id` y `--lora-target-modules`
permiten reutilizarlo para cualquiera de los modelos del catalogo en
`training/model_catalog.py` (Qwen2.5, Qwen3), de forma que
`launch_training_job.py` pueda lanzar un training job independiente por
modelo y correrlos en paralelo sobre el mismo dataset.

Al finalizar, guarda:
  - El adaptador LoRA entrenado (ligero, ~10-50MB) en SM_MODEL_DIR.
  - Metricas de entrenamiento (duracion, loss final, throughput, memoria
    pico de GPU/CPU y tamano del adaptador en disco) en un archivo
    metrics.json dentro de SM_MODEL_DIR, para que launch_training_job.py
    pueda recopilarlas junto con las metricas del describe_training_job de
    SageMaker.
"""
import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

try:
    import psutil
except ImportError:  # pragma: no cover - degrada con gracia si falta la dependencia
    psutil = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model-key", type=str, default="qwen2.5-1.5b",
                         help="Clave del modelo en training/model_catalog.py (solo para trazabilidad en metrics.json)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Lista separada por comas de los modulos objetivo de LoRA (varia por arquitectura, ver model_catalog.py)",
    )
    parser.add_argument("--seed", type=int, default=42)

    # Rutas de SageMaker (inyectadas automaticamente por el SDK/entorno)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./output"))
    parser.add_argument("--train-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "./data/train"))
    parser.add_argument("--eval-dir", type=str, default=os.environ.get("SM_CHANNEL_EVAL", "./data/eval"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output_data"))

    return parser.parse_args()


def find_jsonl_file(directory: str) -> str:
    for fname in os.listdir(directory):
        if fname.endswith(".jsonl"):
            return os.path.join(directory, fname)
    raise FileNotFoundError(f"No se encontro archivo .jsonl en {directory}")


def get_process_rss_mb() -> float | None:
    """Memoria residente (RSS) del proceso actual, en MB. None si psutil no esta disponible."""
    if psutil is None:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def get_dir_size_mb(path: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if os.path.isfile(fpath):
                total += os.path.getsize(fpath)
    return total / (1024 ** 2)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    gpu_available = torch.cuda.is_available()
    if gpu_available:
        torch.cuda.reset_peak_memory_stats()
    rss_before_load_mb = get_process_rss_mb()

    lora_target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

    print(f"[train_qlora] Cargando modelo base: {args.model_id}")
    print(f"[train_qlora] Modulos objetivo de LoRA: {lora_target_modules}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    trainable_params, all_params = model.get_nb_trainable_parameters()

    train_path = find_jsonl_file(args.train_dir)
    eval_path = find_jsonl_file(args.eval_dir)
    print(f"[train_qlora] Dataset de entrenamiento: {train_path}")
    print(f"[train_qlora] Dataset de evaluacion: {eval_path}")

    train_dataset = load_dataset("json", data_files=train_path, split="train")
    eval_dataset = load_dataset("json", data_files=eval_path, split="train")

    def formatting_func(example):
        return tokenizer.apply_chat_template(example["messages"], tokenize=False)

    sft_config = SFTConfig(
        output_dir="/tmp/checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        max_length=args.max_seq_length,
        packing=False,
        report_to=[],
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,
    )

    print("[train_qlora] Iniciando entrenamiento...")
    start_time = time.time()
    train_result = trainer.train()
    training_duration_seconds = time.time() - start_time
    print(f"[train_qlora] Entrenamiento finalizado en {training_duration_seconds:.1f}s")

    # Memoria pico de GPU durante el entrenamiento (torch.cuda.max_memory_*
    # acumula desde el ultimo reset_peak_memory_stats(), llamado antes de
    # cargar el modelo, por lo que captura carga + entrenamiento completos).
    gpu_peak_allocated_mb = None
    gpu_peak_reserved_mb = None
    if gpu_available:
        gpu_peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        gpu_peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

    rss_after_train_mb = get_process_rss_mb()

    eval_metrics = trainer.evaluate()
    print(f"[train_qlora] Metricas de evaluacion: {eval_metrics}")

    print(f"[train_qlora] Guardando adaptador LoRA en {args.model_dir}")
    os.makedirs(args.model_dir, exist_ok=True)
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    adapter_size_mb = get_dir_size_mb(args.model_dir)

    metrics = {
        "model_key": args.model_key,
        "base_model_id": args.model_id,
        "num_train_examples": len(train_dataset),
        "num_eval_examples": len(eval_dataset),
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_target_modules": lora_target_modules,
        "trainable_params": trainable_params,
        "total_params": all_params,
        "trainable_params_pct": round(100 * trainable_params / all_params, 4) if all_params else None,
        "training_duration_seconds": training_duration_seconds,
        "train_runtime_seconds_hf": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        "train_steps_per_second": train_result.metrics.get("train_steps_per_second"),
        "final_train_loss": train_result.metrics.get("train_loss"),
        "eval_loss": eval_metrics.get("eval_loss"),
        # Metricas de memoria (Seccion 8 del notebook / requisito de negocio
        # de capturar memoria utilizada por modelo durante el entrenamiento).
        "gpu_peak_allocated_mb": gpu_peak_allocated_mb,
        "gpu_peak_reserved_mb": gpu_peak_reserved_mb,
        "cpu_rss_before_load_mb": rss_before_load_mb,
        "cpu_rss_after_train_mb": rss_after_train_mb,
        "adapter_size_mb": round(adapter_size_mb, 2),
    }

    os.makedirs(args.output_data_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_data_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Tambien se guarda una copia junto al modelo, por si SM_OUTPUT_DATA_DIR
    # no se recolecta en el job (p.ej. al invocar el script fuera de SageMaker).
    with open(os.path.join(args.model_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[train_qlora] Metricas guardadas: {json.dumps(metrics, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
