"""
Utilidades S3 compartidas por los contenedores de sincronizacion
(data_pipeline/sync/) y de preparacion de datos (data_pipeline/processing/)
que corren como tareas de Amazon ECS Fargate orquestadas por el Step
Function de sincronizacion documental.
"""
import json
import logging
import os
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger("s3_utils")


def get_s3_client():
    return boto3.client("s3")


def read_json_if_exists(bucket: str, key: str, s3=None) -> Optional[Any]:
    s3 = s3 or get_s3_client()
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise


def write_json(bucket: str, key: str, data: Any, s3=None) -> None:
    s3 = s3 or get_s3_client()
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def download_prefix(bucket: str, prefix: str, local_dir: str, s3=None) -> int:
    """Descarga todos los objetos bajo `prefix` a `local_dir`, preservando la
    estructura relativa. Devuelve el numero de archivos descargados."""
    s3 = s3 or get_s3_client()
    os.makedirs(local_dir, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel_path = key[len(prefix):].lstrip("/")
            if not rel_path:
                continue
            dest_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            s3.download_file(bucket, key, dest_path)
            count += 1
    logger.info("Descargados %d objetos de s3://%s/%s a %s", count, bucket, prefix, local_dir)
    return count


def upload_directory(bucket: str, local_dir: str, prefix: str, s3=None) -> int:
    """Sube recursivamente todos los archivos de `local_dir` a
    s3://bucket/prefix/, preservando la estructura relativa. Devuelve el
    numero de archivos subidos."""
    s3 = s3 or get_s3_client()
    count = 0
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, local_dir)
            key = f"{prefix.rstrip('/')}/{rel_path.replace(os.sep, '/')}"
            s3.upload_file(local_path, bucket, key)
            count += 1
    logger.info("Subidos %d archivos de %s a s3://%s/%s", count, local_dir, bucket, prefix)
    return count
