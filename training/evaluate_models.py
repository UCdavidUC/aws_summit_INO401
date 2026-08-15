"""
Evaluacion de rendimiento post-entrenamiento para los adaptadores LoRA
producidos por `launch_training_job.py` (single-job o multi-modelo en
paralelo).

Esto corresponde a la Seccion 16 ("Post Deployment Model Evaluation") de
`docs/technical_documentation.md`: despues de entrenar, cada modelo debe
evaluarse con sus propias metricas de **calidad** (perplexity sobre el set
de evaluacion) y de **rendimiento** (tokens/segundo, latencia TTFT/ITL,
P50/P99), para poder compararlos entre si antes de decidir cual promover.

A diferencia de la Seccion 16 (que evalua un endpoint YA DESPLEGADO en
`mx-central-1` con GuideLLM/MLflow contra trafico HTTP real), este script
evalua los adaptadores **recien entrenados, localmente o en la misma
instancia de entrenamiento**, cargando el modelo fusionado
(base + adaptador) directamente con `transformers`. Es el primer gate de
calidad antes de invertir en el pipeline de cuantizacion/despliegue de la
Seccion 10.

Flujo por modelo:
  1. Localiza sus metricas de entrenamiento (metrics_<job_name>.json,
     escrito por launch_training_job.py) para obtener `base_model_id`,
     `model_key` y la ubicacion del adaptador en S3.
  2. Descarga y extrae el adaptador LoRA desde
     s3://<bucket>/models/<job_name>/<job_name>/output/model.tar.gz
     (o usa un directorio local si se pasa --adapters-dir).
  3. Carga el modelo base + adaptador con `peft.PeftModel` (sin fusionar,
     para reflejar el runtime real de inferencia con adaptador).
  4. Calcula la **perplexity** sobre eval.jsonl (exp(mean cross-entropy)).
  5. Bench de **throughput/latencia**: genera continuaciones para un set de
     prompts de benchmark (variando longitud, ver Seccion 16.3), midiendo
     TTFT (tiempo al primer token), tokens/segundo y latencia total.
  6. Escribe `eval_metrics_<job_name>.json` con ambos bloques de metricas.

Uso (evaluar todos los jobs de una corrida en paralelo):
    python evaluate_models.py --run-manifest parallel_run_<timestamp>.json --bucket <data-bucket>

Uso (evaluar un solo job, adaptador ya descargado localmente):
    python evaluate_models.py --job-name <job_name> --base-model-id Qwen/Qwen2.5-1.5B-Instruct \
        --adapter-dir ./adapters/<job_name> --eval-jsonl ../data_pipeline/processed/dataset/eval.jsonl

Requiere `training/eval_requirements.txt` (torch, transformers, peft,
accelerate) instalado en el entorno de ejecucion.
"""
import argparse
import glob
import io
import json
import math
import os
import statistics
import tarfile
import time
from datetime import datetime, timezone

BENCHMARK_PROMPTS = [
    # Prompts cortos, medianos y largos (~128/256/512 tokens aprox. de
    # contexto), replicando la mezcla de longitudes de la Seccion 16.3.
    "Resume en dos frases el objeto de una disposicion de caracter general emitida por la CNBV.",
    (
        "Redacta un checklist de cumplimiento con las obligaciones tipicas que una institucion "
        "de banca multiple debe cumplir frente a una circular de Banxico sobre reportes regulatorios, "
        "incluyendo plazos y sanciones aplicables en caso de incumplimiento."
    ),
    (
        "Actuando como analista de cumplimiento regulatorio, elabora un resumen estructurado (objeto, "
        "sujetos obligados, plazos y sanciones) de una circular hipotetica de la CNBV dirigida a "
        "uniones de credito, que establece nuevos requisitos de reporte de operaciones con partes "
        "relacionadas, e incluye una clasificacion del tipo de norma y el sector regulado al que aplica, "
        "citando el fundamento normativo de forma generica dado que no se cuenta con el texto original."
    ),
]

SYSTEM_PROMPT = (
    "Eres un asistente especializado en cumplimiento regulatorio financiero "
    "mexicano (CNBV y Banxico). Ayudas a evaluar carpetas de cumplimiento y a "
    "estructurar documentos regulatorios de forma precisa y concisa, citando "
    "el fundamento normativo cuando sea posible."
)

MAX_NEW_TOKENS = 256


def _lazy_imports():
    """Importa las dependencias pesadas (torch/transformers/peft) solo cuando se necesitan,
    para que --help y la carga del modulo no requieran GPU/estas librerias instaladas."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, PeftModel, AutoModelForCausalLM, AutoTokenizer


def find_metrics_files(metrics_dir: str, job_names=None):
    if job_names:
        return [os.path.join(metrics_dir, f"metrics_{jn}.json") for jn in job_names]
    return sorted(glob.glob(os.path.join(metrics_dir, "metrics_*.json")))


def load_run_manifest(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    job_names = []
    for _key, job in manifest.get("jobs", {}).items():
        jn = job.get("training_job_name") or job.get("job_name")
        if jn:
            job_names.append(jn)
    return job_names


def download_and_extract_adapter(s3_client, bucket: str, job_name: str, dest_dir: str) -> str:
    key = f"models/{job_name}/{job_name}/output/model.tar.gz"
    adapter_dir = os.path.join(dest_dir, job_name)
    os.makedirs(adapter_dir, exist_ok=True)

    buffer = io.BytesIO()
    s3_client.download_fileobj(bucket, key, buffer)
    buffer.seek(0)
    with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
        tar.extractall(adapter_dir)  # noqa: S202 - artefacto propio, generado por nuestro training job
    return adapter_dir


def compute_perplexity(torch, model, tokenizer, eval_path: str, max_examples: int = 100, max_length: int = 1024) -> dict:
    """
    Perplexity = exp(cross-entropy promedio) sobre el set de evaluacion,
    consistente con el `eval_loss` que reporta trl.SFTTrainer durante el
    entrenamiento (Seccion 8 del notebook), pero calculado aqui de forma
    independiente sobre el modelo con adaptador ya cargado para inferencia.
    """
    model.eval()
    losses = []
    with open(eval_path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f][:max_examples]

    with torch.no_grad():
        for rec in lines:
            text = tokenizer.apply_chat_template(rec["messages"], tokenize=False)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            losses.append(outputs.loss.item())

    mean_loss = statistics.mean(losses) if losses else None
    perplexity = math.exp(mean_loss) if mean_loss is not None else None
    return {
        "eval_examples_used": len(lines),
        "mean_eval_loss": mean_loss,
        "perplexity": perplexity,
    }


def benchmark_throughput(torch, model, tokenizer) -> dict:
    """
    Bench de generacion: para cada prompt de BENCHMARK_PROMPTS, mide TTFT
    (tiempo hasta el primer token generado), tiempo total de generacion y
    tokens/segundo. Aproxima TTFT generando primero max_new_tokens=1 y
    despues completando la generacion, ya que `generate()` no expone un
    callback nativo de streaming sin usar TextIteratorStreamer.
    """
    ttft_seconds = []
    total_latency_seconds = []
    tokens_per_second = []

    for prompt in BENCHMARK_PROMPTS:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # TTFT: tiempo para producir el primer token generado.
        start = time.time()
        with torch.no_grad():
            first_token_out = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        ttft = time.time() - start
        ttft_seconds.append(ttft)

        # Generacion completa, para throughput sostenido.
        start_full = time.time()
        with torch.no_grad():
            full_out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        total_latency = time.time() - start_full
        generated_tokens = full_out.shape[-1] - inputs["input_ids"].shape[-1]
        tok_s = generated_tokens / total_latency if total_latency > 0 else None

        total_latency_seconds.append(total_latency)
        if tok_s is not None:
            tokens_per_second.append(tok_s)

    def _pctl(values, p):
        if not values:
            return None
        values_sorted = sorted(values)
        idx = min(int(len(values_sorted) * p), len(values_sorted) - 1)
        return values_sorted[idx]

    return {
        "num_prompts": len(BENCHMARK_PROMPTS),
        "max_new_tokens": MAX_NEW_TOKENS,
        "ttft_seconds_avg": statistics.mean(ttft_seconds) if ttft_seconds else None,
        "ttft_seconds_p50": _pctl(ttft_seconds, 0.5),
        "ttft_seconds_p99": _pctl(ttft_seconds, 0.99),
        "total_latency_seconds_avg": statistics.mean(total_latency_seconds) if total_latency_seconds else None,
        "total_latency_seconds_p50": _pctl(total_latency_seconds, 0.5),
        "total_latency_seconds_p99": _pctl(total_latency_seconds, 0.99),
        "tokens_per_second_avg": statistics.mean(tokens_per_second) if tokens_per_second else None,
    }


def evaluate_one(
    torch,
    PeftModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    base_model_id: str,
    adapter_dir: str,
    eval_jsonl: str,
    model_key: str,
    job_name: str,
    device_map: str,
) -> dict:
    print(f"[{job_name}] Cargando modelo base {base_model_id} + adaptador de {adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map=device_map
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    perplexity_metrics = compute_perplexity(torch, model, tokenizer, eval_jsonl)
    print(f"[{job_name}] Perplexity: {perplexity_metrics['perplexity']}")

    throughput_metrics = benchmark_throughput(torch, model, tokenizer)
    print(f"[{job_name}] Tokens/s promedio: {throughput_metrics['tokens_per_second_avg']}")

    return {
        "job_name": job_name,
        "model_key": model_key,
        "base_model_id": base_model_id,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "device": str(model.device),
        "quality": perplexity_metrics,
        "performance": throughput_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Evalua calidad (perplexity) y rendimiento (tok/s, latencia) de adaptadores LoRA entrenados")
    parser.add_argument("--metrics-dir", default=os.path.dirname(__file__),
                         help="Directorio donde buscar metrics_<job_name>.json (default: training/)")
    parser.add_argument("--run-manifest", default=None,
                         help="Ruta a parallel_run_<timestamp>.json; evalua todos los jobs de esa corrida")
    parser.add_argument("--job-name", default=None, help="Evalua un unico job por nombre (modo manual)")
    parser.add_argument("--base-model-id", default=None, help="Requerido junto con --job-name si no hay metrics_<job_name>.json disponible")
    parser.add_argument("--adapter-dir", default=None, help="Directorio local con el adaptador ya extraido (evita descargar de S3)")
    parser.add_argument("--adapters-download-dir", default=os.path.join(os.path.dirname(__file__), "_adapters_eval"))
    parser.add_argument("--bucket", default=None, help="Bucket S3 de donde descargar el adaptador (models/<job_name>/.../model.tar.gz)")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--eval-jsonl", default=os.path.join(
        os.path.dirname(__file__), "..", "data_pipeline", "processed", "dataset", "eval.jsonl"
    ))
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    torch, PeftModel, AutoModelForCausalLM, AutoTokenizer = _lazy_imports()

    jobs_to_eval = []  # lista de dicts: {job_name, model_key, base_model_id, adapter_dir}

    if args.job_name:
        base_model_id = args.base_model_id
        model_key = args.job_name
        metrics_path = os.path.join(args.metrics_dir, f"metrics_{args.job_name}.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, encoding="utf-8") as f:
                m = json.load(f)
            base_model_id = base_model_id or m.get("base_model_id")
            model_key = m.get("model_key", model_key)
        if not base_model_id:
            raise SystemExit("--base-model-id es requerido si no hay metrics_<job_name>.json disponible")
        jobs_to_eval.append({
            "job_name": args.job_name,
            "model_key": model_key,
            "base_model_id": base_model_id,
            "adapter_dir": args.adapter_dir,
        })
    else:
        job_names = load_run_manifest(args.run_manifest) if args.run_manifest else None
        for metrics_path in find_metrics_files(args.metrics_dir, job_names):
            if not os.path.exists(metrics_path):
                print(f"Aviso: no existe {metrics_path}, se omite")
                continue
            with open(metrics_path, encoding="utf-8") as f:
                m = json.load(f)
            jobs_to_eval.append({
                "job_name": m.get("training_job_name"),
                "model_key": m.get("model_key", m.get("training_job_name")),
                "base_model_id": m.get("base_model_id"),
                "adapter_dir": None,
            })

    if not jobs_to_eval:
        raise SystemExit("No se encontraron jobs para evaluar. Usa --job-name, --run-manifest, o revisa --metrics-dir.")

    s3_client = None
    if args.bucket:
        import boto3
        s3_client = boto3.Session(region_name=args.region).client("s3")

    results = {}
    for job in jobs_to_eval:
        job_name = job["job_name"]
        adapter_dir = job["adapter_dir"] or args.adapter_dir
        if adapter_dir is None:
            if s3_client is None:
                raise SystemExit(f"[{job_name}] Falta --adapter-dir o --bucket para descargar el adaptador")
            adapter_dir = download_and_extract_adapter(s3_client, args.bucket, job_name, args.adapters_download_dir)

        result = evaluate_one(
            torch, PeftModel, AutoModelForCausalLM, AutoTokenizer,
            base_model_id=job["base_model_id"],
            adapter_dir=adapter_dir,
            eval_jsonl=args.eval_jsonl,
            model_key=job["model_key"],
            job_name=job_name,
            device_map=args.device_map,
        )
        results[job_name] = result

        out_path = os.path.join(args.metrics_dir, f"eval_metrics_{job_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[{job_name}] Metricas de evaluacion guardadas en: {out_path}")

    summary_path = os.path.join(
        args.metrics_dir, f"eval_summary_{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H-%M-%S')}.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Resumen de evaluacion guardado en: {summary_path}")


if __name__ == "__main__":
    main()
