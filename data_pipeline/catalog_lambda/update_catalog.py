"""
Lambda invocada por el Step Function DocumentSyncStateMachine para volcar el
resumen de la sincronizacion documental (escrito por
data_pipeline/sync/sync_documents.py en S3) a la tabla DynamoDB
`finance_document_catalog`.

Un item por documento conocido (nuevo, actualizado o sin cambios), con
clave primaria `doc_id` (formato "<source>#<s3_key>"), de forma que el
catalogo siempre refleje el estado mas reciente conocido de cada documento
regulatorio y su ubicacion en el bucket de datos (incluyendo el
`s3_version_id` de S3 para poder recuperar versiones anteriores si el
bucket tiene versionado habilitado, como es el caso).

Entrada esperada (event):
    {
        "bucket": "<data bucket>",
        "summary_s3_key": "sync-runs/latest_summary.json"
    }

Variables de entorno:
    CATALOG_TABLE_NAME   Nombre de la tabla DynamoDB (por defecto
                          "finance_document_catalog")
"""
import json
import logging
import os

import boto3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("update_catalog")

dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")

TABLE_NAME = os.environ.get("CATALOG_TABLE_NAME", "finance_document_catalog")


def _to_item(record: dict) -> dict:
    item = {
        "doc_id": record["doc_id"],
        "source": record["source"],
        "s3_key": record["s3_key"],
        "s3_version_id": record.get("s3_version_id") or "null",
        "url": record.get("url", ""),
        "status": record.get("status", "unchanged"),
        "last_synced_at": record.get("last_synced_at", ""),
        "nombre": record.get("nombre", ""),
    }
    # Campos opcionales especificos de CNBV.
    for key in ("sector", "tipo", "fecha_publicacion_dof", "doc_id_origen", "kind"):
        source_key = "doc_id" if key == "doc_id_origen" else key
        if source_key in record and source_key != "doc_id":
            item[key] = record[source_key]
    return {k: v for k, v in item.items() if v is not None}


def handler(event, context):
    bucket = event["bucket"]
    summary_s3_key = event.get("summary_s3_key", "sync-runs/latest_summary.json")

    obj = s3.get_object(Bucket=bucket, Key=summary_s3_key)
    summary = json.loads(obj["Body"].read())

    catalog_records = summary.get("catalog_records", [])
    table = dynamodb.Table(TABLE_NAME)

    written = 0
    with table.batch_writer(overwrite_by_pkeys=["doc_id"]) as batch:
        for record in catalog_records:
            batch.put_item(Item=_to_item(record))
            written += 1

    logger.info("Catalogo actualizado: %d items escritos en %s", written, TABLE_NAME)
    return {"catalog_items_written": written, "any_changes": summary.get("any_changes", False)}
