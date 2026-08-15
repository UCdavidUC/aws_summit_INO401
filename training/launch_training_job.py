"""
Lanza fine-tuning QLoRA de uno o varios SLMs en SageMaker Training Jobs,
usando boto3 directamente (sin el SDK de alto nivel `sagemaker`, para
mantener el entorno local ligero).

Soporta dos modos:

  1. **Single-job** (comportamiento original): `--model-id` sobreescribe el
     modelo a entrenar en un unico job. Sigue funcionando exactamente igual
     que antes (mismo CLI, mismo archivo `metrics_<job_name>.json`).

  2. **Multi-modelo en paralelo** (nuevo, `--model-keys`): lanza un
     training job independiente por cada modelo del catalogo
     (`training/model_catalog.py`, alineado a la Seccion 3/7.1/17.1 de
     `docs/technical_documentation.md`), todos sobre el mismo dataset SFT,
     y los espera **concurrentemente** con un `ThreadPoolExecutor`. Esto
     permite comparar tiempo de entrenamiento, memoria, throughput y loss
     entre modelos bajo las mismas condiciones de datos/hiperparametros.

Pasos por job:
  1. Empaqueta training/source/ (train_qlora.py + requirements.txt) en un
     sourcedir.tar.gz y lo sube a S3 (los SageMaker Deep Learning Containers
     de HuggingFace saben descomprimir y ejecutar automaticamente un paquete
     con esta convencion, via las variables de entorno SAGEMAKER_*).
  2. Sube los datasets train.jsonl / eval.jsonl a S3 bajo un prefijo propio
     del job (capa "gold" del datalake, Seccion 1.1 del notebook: inmutable
     por convencion, nunca se sobreescribe).
  3. Llama a create_training_job con el HuggingFace Training DLC (PyTorch +
     Transformers, imagen GPU) y espera a que finalice.
  4. Al terminar, descarga metrics.json del artefacto de salida (model.tar.gz,
     que train_qlora.py escribe junto al adaptador) y lo combina con las
     metricas de duracion/costo/recursos obtenidas de describe_training_job
     y CloudWatch (CPUUtilization, MemoryUtilization, GPUUtilization,
     GPUMemoryUtilization del namespace /aws/sagemaker/TrainingJobs).

Uso (single-job, igual que antes):
    python launch_training_job.py \
        --bucket <data-bucket> \
        --role-arn <SlmTrainingStack TrainingExecutionRoleArn> \
        --region us-west-2

Uso (multi-modelo en paralelo):
    python launch_training_job.py \
        --bucket <data-bucket> \
        --role-arn <SlmTrainingStack TrainingExecutionRoleArn> \
        --region us-west-2 \
        --model-keys qwen2.5-1.5b,qwen3-0.6b \
        --max-workers 2
"""
import argparse
import concurrent.futures
import io
import json
import os
import tarfile
import time
from datetime import datetime, timezone

import boto3

from model_catalog import DEFAULT_MODEL_KEYS, ModelSpec, get_model_spec, list_model_specs

# HuggingFace PyTorch Training DLC (GPU, cuenta publica de AWS Deep Learning
# Containers). Incluye PyTorch 2.9 + transformers 5.3, base sobre la que se
# instalan peft/trl/bitsandbytes/accelerate/datasets via requirements.txt.
HF_TRAINING_IMAGE_URI_TEMPLATE = (
    "763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-training:"
    "2.9.0-transformers5.3.0-gpu-py312-cu130-ubuntu22.04"
)

SOURCE_DIR = os.path.join(os.path.dirname(__file__), "source")

# Precios on-demand de SageMaker Training en us-west-2, obtenidos de la AWS
# Price List API (servicio AmazonSageMaker, usagetype USW2-Train:<instancia>).
# Se usan unicamente para el costo estimado en las metricas; no reflejan el
# precio en tiempo real (volver a consultar la API para cifras exactas).
#
# Default recomendado: familia G6 (GPU NVIDIA L4, 24GB) en vez de G5
# (A10G, 24GB): mismo VRAM, ~15-20% mas barata por hora (ver Seccion 17.1 de
# docs/technical_documentation.md, "Cost Optimization Strategies"). Las
# entradas de G5 se conservan para compatibilidad con jobs/scripts previos
# que fijen `--instance-type ml.g5.*` explicitamente.
INSTANCE_HOURLY_COST_USD = {
    "ml.g6.xlarge": 1.13,
    "ml.g6.2xlarge": 1.22,
    "ml.g6.4xlarge": 1.65,
    "ml.g6.8xlarge": 2.52,
    "ml.g6.12xlarge": 5.75,
    "ml.g5.xlarge": 1.41,
    "ml.g5.2xlarge": 1.52,
    "ml.g5.4xlarge": 2.03,
    "ml.g5.12xlarge": 7.09,
    "ml.p3.2xlarge": 3.83,
    "ml.p4d.24xlarge": 37.69,
    "ml.p5.48xlarge": 98.32,
}

# Metricas de sistema publicadas por SageMaker en CloudWatch para cada
# training job, bajo el namespace /aws/sagemaker/TrainingJobs con dimension
# Host=<job-name>/algo-1. Se recolectan como evidencia de "memoria
# utilizada" / utilizacion de CPU y GPU durante el entrenamiento.
CLOUDWATCH_RESOURCE_METRICS = [
    "CPUUtilization",
    "MemoryUtilization",
    "GPUUtilization",
    "GPUMemoryUtilization",
]


def package_source_dir() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for fname in os.listdir(SOURCE_DIR):
            fpath = os.path.join(SOURCE_DIR, fname)
            tar.add(fpath, arcname=fname)
    buffer.seek(0)
    return buffer.read()


def upload_source_dir(s3_client, bucket: str, prefix: str) -> str:
    data = package_source_dir()
    key = f"{prefix}/sourcedir.tar.gz"
    s3_client.put_object(Bucket=bucket, Key=key, Body=data)
    return f"s3://{bucket}/{key}"


def ensure_dataset_uploaded(s3_client, bucket: str, local_path: str, s3_prefix: str) -> str:
    key = f"{s3_prefix}/{os.path.basename(local_path)}"
    s3_client.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{s3_prefix}/"


def create_training_job(
    sm_client,
    job_name: str,
    role_arn: str,
    image_uri: str,
    source_dir_s3_uri: str,
    train_s3_uri: str,
    eval_s3_uri: str,
    output_s3_uri: str,
    instance_type: str,
    hyperparameters: dict,
    max_run_seconds: int,
):
    sm_hyperparameters = {k: json.dumps(v) if not isinstance(v, str) else v for k, v in hyperparameters.items()}
    # Convenciones del HuggingFace/PyTorch Training Toolkit para localizar y
    # ejecutar el entry point empaquetado en sourcedir.tar.gz.
    sm_hyperparameters["sagemaker_program"] = json.dumps("train_qlora.py")
    sm_hyperparameters["sagemaker_submit_directory"] = json.dumps(source_dir_s3_uri)

    response = sm_client.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
        },
        RoleArn=role_arn,
        HyperParameters=sm_hyperparameters,
        InputDataConfig=[
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": train_s3_uri,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            },
            {
                "ChannelName": "eval",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": eval_s3_uri,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            },
        ],
        OutputDataConfig={"S3OutputPath": output_s3_uri},
        ResourceConfig={
            "InstanceType": instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 100,
        },
        StoppingCondition={"MaxRuntimeInSeconds": max_run_seconds},
    )
    return response


def wait_for_job(sm_client, job_name: str, poll_seconds: int = 30) -> dict:
    while True:
        desc = sm_client.describe_training_job(TrainingJobName=job_name)
        status = desc["TrainingJobStatus"]
        secondary = desc.get("SecondaryStatus", "")
        print(f"[{job_name}] Estado: {status} ({secondary})", flush=True)
        if status in ("Completed", "Failed", "Stopped"):
            return desc
        time.sleep(poll_seconds)


def collect_cloudwatch_resource_metrics(cw_client, job_name: str, start, end) -> dict:
    """
    Recolecta utilizacion de CPU/memoria/GPU publicada por SageMaker en
    CloudWatch durante la ventana [start, end] del job. Best-effort: si el
    namespace no tiene datapoints (job muy corto, region sin soporte, o los
    datos aun no se han propagado), se devuelve None para esas metricas en
    vez de fallar el recolector de metricas completo.
    """
    results = {}
    if not start or not end:
        return {f"{m}_avg": None for m in CLOUDWATCH_RESOURCE_METRICS} | {
            f"{m}_max": None for m in CLOUDWATCH_RESOURCE_METRICS
        }
    for metric_name in CLOUDWATCH_RESOURCE_METRICS:
        avg_key = f"{metric_name}_avg"
        max_key = f"{metric_name}_max"
        try:
            resp = cw_client.get_metric_statistics(
                Namespace="/aws/sagemaker/TrainingJobs",
                MetricName=metric_name,
                Dimensions=[{"Name": "Host", "Value": f"{job_name}/algo-1"}],
                StartTime=start,
                EndTime=end,
                Period=60,
                Statistics=["Average", "Maximum"],
            )
            datapoints = resp.get("Datapoints", [])
            if datapoints:
                results[avg_key] = sum(d["Average"] for d in datapoints) / len(datapoints)
                results[max_key] = max(d["Maximum"] for d in datapoints)
            else:
                results[avg_key] = None
                results[max_key] = None
        except Exception as exc:  # pragma: no cover - resiliencia ante permisos/latencia de CloudWatch
            print(f"[{job_name}] Aviso: no se pudo leer la metrica de CloudWatch '{metric_name}': {exc}")
            results[avg_key] = None
            results[max_key] = None
    return results


def download_container_metrics(s3_client, bucket: str, job_name: str) -> dict:
    """
    Descarga y extrae metrics.json desde el model.tar.gz que train_qlora.py
    guarda junto al adaptador LoRA (ver train_qlora.py: escribe una copia de
    metrics.json en SM_MODEL_DIR ademas de SM_OUTPUT_DATA_DIR). Devuelve {}
    si el artefacto no existe o no contiene metrics.json (p.ej. job fallido).
    """
    key = f"models/{job_name}/{job_name}/output/model.tar.gz"
    try:
        buffer = io.BytesIO()
        s3_client.download_fileobj(bucket, key, buffer)
        buffer.seek(0)
        with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
            member = next((m for m in tar.getmembers() if m.name.endswith("metrics.json")), None)
            if member is None:
                return {}
            extracted = tar.extractfile(member)
            return json.load(extracted) if extracted else {}
    except Exception as exc:  # pragma: no cover - artefacto puede no existir si el job fallo
        print(f"[{job_name}] Aviso: no se pudo descargar metrics.json de s3://{bucket}/{key}: {exc}")
        return {}


def collect_metrics(desc: dict) -> dict:
    creation = desc.get("CreationTime")
    start = desc.get("TrainingStartTime")
    end = desc.get("TrainingEndTime")

    duration_seconds = None
    if start and end:
        duration_seconds = (end - start).total_seconds()

    billable_seconds = desc.get("BillableTimeInSeconds")
    instance_type = desc["ResourceConfig"]["InstanceType"]

    hourly_cost = INSTANCE_HOURLY_COST_USD.get(instance_type)
    estimated_cost_usd = (
        round(billable_seconds / 3600 * hourly_cost, 2)
        if billable_seconds is not None and hourly_cost is not None
        else None
    )

    return {
        "training_job_name": desc["TrainingJobName"],
        "training_job_status": desc["TrainingJobStatus"],
        "instance_type": instance_type,
        "creation_time": creation.isoformat() if creation else None,
        "training_start_time": start.isoformat() if start else None,
        "training_end_time": end.isoformat() if end else None,
        "duration_seconds": duration_seconds,
        "billable_time_seconds": billable_seconds,
        "estimated_cost_usd": estimated_cost_usd,
        "final_metric_data_list": desc.get("FinalMetricDataList", []),
        "failure_reason": desc.get("FailureReason"),
    }


def run_single_training_job(
    session: boto3.Session,
    bucket: str,
    role_arn: str,
    region: str,
    instance_type: str,
    train_jsonl: str,
    eval_jsonl: str,
    epochs: int,
    max_run_hours: float,
    job_name: str,
    model_id: str,
    model_key: str,
    lora_target_modules: list,
    wait: bool,
    metrics_out_dir: str,
) -> dict:
    """Ejecuta el ciclo completo (empaquetar, subir, crear, esperar, recolectar) para UN job."""
    s3_client = session.client("s3")
    sm_client = session.client("sagemaker")
    cw_client = session.client("cloudwatch")

    print(f"[{job_name}] Empaquetando y subiendo codigo de entrenamiento...")
    source_dir_s3_uri = upload_source_dir(s3_client, bucket, f"code/{job_name}")
    print(f"[{job_name}] Codigo subido a {source_dir_s3_uri}")

    print(f"[{job_name}] Subiendo datasets...")
    train_s3_uri = ensure_dataset_uploaded(s3_client, bucket, train_jsonl, f"datasets/{job_name}/train")
    eval_s3_uri = ensure_dataset_uploaded(s3_client, bucket, eval_jsonl, f"datasets/{job_name}/eval")
    print(f"[{job_name}] train: {train_s3_uri}, eval: {eval_s3_uri}")

    output_s3_uri = f"s3://{bucket}/models/{job_name}/"
    image_uri = HF_TRAINING_IMAGE_URI_TEMPLATE.format(region=region)

    hyperparameters = {
        "model-id": model_id,
        "model-key": model_key,
        "epochs": epochs,
        "learning-rate": 2e-4,
        "per-device-train-batch-size": 4,
        "per-device-eval-batch-size": 4,
        "gradient-accumulation-steps": 4,
        "max-seq-length": 1024,
        "lora-r": 16,
        "lora-alpha": 32,
        "lora-dropout": 0.05,
        "lora-target-modules": ",".join(lora_target_modules),
        "seed": 42,
    }

    print(f"[{job_name}] Creando training job (modelo: {model_id}, instancia: {instance_type})")
    create_training_job(
        sm_client=sm_client,
        job_name=job_name,
        role_arn=role_arn,
        image_uri=image_uri,
        source_dir_s3_uri=source_dir_s3_uri,
        train_s3_uri=train_s3_uri,
        eval_s3_uri=eval_s3_uri,
        output_s3_uri=output_s3_uri,
        instance_type=instance_type,
        hyperparameters=hyperparameters,
        max_run_seconds=int(max_run_hours * 3600),
    )
    print(f"[{job_name}] Training job creado")

    if not wait:
        return {"job_name": job_name, "model_key": model_key, "model_id": model_id, "status": "InProgress"}

    desc = wait_for_job(sm_client, job_name)
    metrics = collect_metrics(desc)
    metrics["model_key"] = model_key

    resource_metrics = collect_cloudwatch_resource_metrics(
        cw_client, job_name, desc.get("TrainingStartTime"), desc.get("TrainingEndTime")
    )
    metrics["resource_utilization"] = resource_metrics

    if desc["TrainingJobStatus"] == "Completed":
        container_metrics = download_container_metrics(s3_client, bucket, job_name)
        metrics.update(container_metrics)

    metrics_out_path = os.path.join(metrics_out_dir, f"metrics_{job_name}.json")
    with open(metrics_out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    print(f"[{job_name}] Job finalizado con estado: {desc['TrainingJobStatus']}")
    print(f"[{job_name}] Metricas guardadas en: {metrics_out_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Lanza uno o varios training jobs QLoRA en SageMaker")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--instance-type", default=None,
                         help="Sobreescribe la instancia recomendada del catalogo para TODOS los modelos lanzados")
    parser.add_argument("--train-jsonl", default="../data_pipeline/processed/dataset/train.jsonl")
    parser.add_argument("--eval-jsonl", default="../data_pipeline/processed/dataset/eval.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-run-hours", type=float, default=6.0)
    parser.add_argument("--job-name", default=None,
                         help="Nombre de job explicito. Solo valido en modo single-job (un unico modelo).")
    parser.add_argument("--wait", action="store_true", default=True)
    parser.add_argument("--no-wait", dest="wait", action="store_false")

    # Modo single-job (compatibilidad con el flujo original): si se pasa
    # --model-id, se ignora el catalogo y se lanza UN solo job con ese modelo,
    # exactamente como en la version original del script.
    parser.add_argument("--model-id", default=None,
                         help="Si se especifica, lanza un unico job con este model-id (modo original/single-job)")
    parser.add_argument("--model-key", default="qwen2.5-1.5b",
                         help="Clave de model_catalog.py a usar junto con --model-id en modo single-job")

    # Modo multi-modelo en paralelo (nuevo).
    parser.add_argument("--model-keys", default=None,
                         help="Lista separada por comas de claves de model_catalog.py a entrenar EN PARALELO "
                              "(p.ej. qwen2.5-1.5b,qwen3-0.6b). "
                              f"Si se omite junto con --model-id, se usa el catalogo por defecto: {','.join(DEFAULT_MODEL_KEYS)}")
    parser.add_argument("--max-workers", type=int, default=4,
                         help="Numero maximo de training jobs a lanzar/esperar en paralelo")

    args = parser.parse_args()

    session = boto3.Session(region_name=args.region)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
    metrics_out_dir = os.path.dirname(__file__)

    if args.model_id is not None:
        # ---- Modo single-job (compatibilidad retro) ----
        job_name = args.job_name or f"cnbv-banxico-{args.model_key.replace('.', '-')}-qlora-{timestamp}"
        try:
            spec = get_model_spec(args.model_key)
            lora_target_modules = spec.lora_target_modules
        except KeyError:
            lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
            ]
        instance_type = args.instance_type or "ml.g6.2xlarge"

        metrics = run_single_training_job(
            session=session,
            bucket=args.bucket,
            role_arn=args.role_arn,
            region=args.region,
            instance_type=instance_type,
            train_jsonl=args.train_jsonl,
            eval_jsonl=args.eval_jsonl,
            epochs=args.epochs,
            max_run_hours=args.max_run_hours,
            job_name=job_name,
            model_id=args.model_id,
            model_key=args.model_key,
            lora_target_modules=lora_target_modules,
            wait=args.wait,
            metrics_out_dir=metrics_out_dir,
        )
        print(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))
        return

    # ---- Modo multi-modelo en paralelo ----
    model_keys = [k.strip() for k in args.model_keys.split(",")] if args.model_keys else DEFAULT_MODEL_KEYS
    specs = list_model_specs(model_keys)

    print(f"[launch_training_job] Lanzando {len(specs)} training jobs en paralelo: {[s.key for s in specs]}")

    run_timestamp = timestamp
    job_names = {spec.key: f"cnbv-banxico-{spec.key.replace('.', '-')}-qlora-{run_timestamp}" for spec in specs}

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_key = {}
        for spec in specs:
            instance_type = args.instance_type or spec.recommended_instance_type
            future = executor.submit(
                run_single_training_job,
                session=session,
                bucket=args.bucket,
                role_arn=args.role_arn,
                region=args.region,
                instance_type=instance_type,
                train_jsonl=args.train_jsonl,
                eval_jsonl=args.eval_jsonl,
                epochs=args.epochs,
                max_run_hours=args.max_run_hours,
                job_name=job_names[spec.key],
                model_id=spec.model_id,
                model_key=spec.key,
                lora_target_modules=spec.lora_target_modules,
                wait=args.wait,
                metrics_out_dir=metrics_out_dir,
            )
            future_to_key[future] = spec.key

        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                print(f"[{job_names[key]}] ERROR: {exc}")
                results[key] = {"model_key": key, "job_name": job_names[key], "error": str(exc)}

    parallel_run_path = os.path.join(metrics_out_dir, f"parallel_run_{run_timestamp}.json")
    with open(parallel_run_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_timestamp": run_timestamp,
                "model_keys": model_keys,
                "jobs": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print(f"[launch_training_job] Resumen de la corrida en paralelo guardado en: {parallel_run_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
