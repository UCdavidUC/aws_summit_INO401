"""
Preparacion de datos para fine-tuning, orientada a S3, disparada por el
Step Function `DocumentSyncStateMachine` cada vez que la sincronizacion
documental (data_pipeline/sync/sync_documents.py) detecta documentos
nuevos o actualizados.

Reutiliza la logica de extract_text.py (extraccion de texto de PDFs con
pypdf), chunk_documents.py (chunking por parrafos) y build_dataset.py
(generacion de pares instruccion/respuesta con Amazon Bedrock) sobre una
copia local temporal de s3://<bucket>/raw/<source>/, y sube los resultados
a:

    s3://<bucket>/processed/text/<source>/*.txt
    s3://<bucket>/processed/text/manifest_<source>.jsonl
    s3://<bucket>/processed/chunks.jsonl
    s3://<bucket>/processed/dataset/{train,eval,dataset_with_metadata,generation_log}.jsonl

El corpus de documentos regulatorios es de tamano modesto (cientos de PDFs),
por lo que en cada corrida se re-procesa el contenido completo de raw/ en
lugar de intentar un merge incremental de manifests/chunks; esto es mas
simple y evita inconsistencias si un documento fue removido o renombrado en
la fuente original.

El paso de generacion de dataset (build_dataset.py) invoca Amazon Bedrock por
cada chunk seleccionado y puede omitirse (GENERATE_DATASET=false) si solo se
necesita refrescar el texto/chunks sin regenerar el dataset de fine-tuning.

Variables de entorno esperadas:
    DATA_BUCKET             Bucket S3 (obligatorio)
    SOURCES                 Lista separada por comas de fuentes a preparar
                             (por defecto: "cnbv,banxico")
    GENERATE_DATASET        "true"/"false" (por defecto "true")
    BEDROCK_REGION          Region de Bedrock para build_dataset.py
                             (por defecto "us-west-2")
    MAX_CHUNKS_PER_DOC       (por defecto 10)
    MAX_TOTAL_CHUNKS         (opcional, sin limite por defecto)
    NUM_EXAMPLES_PER_CHUNK   (por defecto 2)
"""
import logging
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from processing.extract_text import process_directory as extract_text_directory  # noqa: E402
from processing.chunk_documents import run as chunk_documents_run  # noqa: E402
from processing.build_dataset import run as build_dataset_run  # noqa: E402
from common.s3_utils import download_prefix, upload_directory, get_s3_client  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_data_prep")


def run(bucket: str, sources: list, generate_dataset: bool = True) -> dict:
    s3 = get_s3_client()
    workdir = tempfile.mkdtemp(prefix="data_prep_")
    text_out_dir = os.path.join(workdir, "processed", "text")
    os.makedirs(text_out_dir, exist_ok=True)

    stats = {}
    try:
        for source in sources:
            raw_dir = os.path.join(workdir, "raw", source)
            num_downloaded = download_prefix(bucket, f"raw/{source}/", raw_dir, s3=s3)
            logger.info("Descargados %d objetos de raw/%s/", num_downloaded, source)

            manifest_path = extract_text_directory(raw_dir, source, text_out_dir)
            stats[source] = {"num_raw_objects": num_downloaded, "manifest": manifest_path}

        chunks_path = os.path.join(workdir, "processed", "chunks.jsonl")
        chunk_documents_run(text_out_dir, chunks_path)

        num_uploaded = upload_directory(bucket, text_out_dir, "processed/text", s3=s3)
        s3.upload_file(chunks_path, bucket, "processed/chunks.jsonl")

        logger.info(
            "Extraccion/chunking finalizado. Archivos de texto/manifests subidos: %d, chunks.jsonl actualizado",
            num_uploaded,
        )

        num_dataset_files = 0
        if generate_dataset:
            dataset_out_dir = os.path.join(workdir, "processed", "dataset")
            max_total = os.environ.get("MAX_TOTAL_CHUNKS")
            build_dataset_run(
                chunks_path=chunks_path,
                out_dir=dataset_out_dir,
                max_chunks_per_doc=int(os.environ.get("MAX_CHUNKS_PER_DOC", "10")),
                max_total_chunks=int(max_total) if max_total else None,
                num_examples_per_chunk=int(os.environ.get("NUM_EXAMPLES_PER_CHUNK", "2")),
                region=os.environ.get("BEDROCK_REGION", "us-west-2"),
                eval_fraction=float(os.environ.get("EVAL_FRACTION", "0.1")),
                seed=int(os.environ.get("SEED", "42")),
                max_workers=int(os.environ.get("MAX_WORKERS", "8")),
            )
            num_dataset_files = upload_directory(bucket, dataset_out_dir, "processed/dataset", s3=s3)
            logger.info("Dataset de fine-tuning actualizado. Archivos subidos: %d", num_dataset_files)

        return {
            "sources": stats,
            "num_text_objects_uploaded": num_uploaded,
            "num_dataset_files_uploaded": num_dataset_files,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    bucket = os.environ["DATA_BUCKET"]
    sources = [s.strip() for s in os.environ.get("SOURCES", "cnbv,banxico").split(",") if s.strip()]
    generate_dataset = os.environ.get("GENERATE_DATASET", "true").strip().lower() != "false"
    run(bucket, sources, generate_dataset=generate_dataset)


if __name__ == "__main__":
    main()
