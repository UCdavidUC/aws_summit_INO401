"""
Orquestador de sincronizacion documental CNBV/Banxico contra S3.

Reutiliza la logica de descubrimiento de scraping/scrape_cnbv.py y
scraping/scrape_banxico.py (paginas indice -> lista de documentos con su URL
de descarga), pero en lugar de escribir a un directorio local:

1. Para cada documento descubierto, hace un HEAD request y compara
   Content-Length/ETag/Last-Modified contra el estado conocido guardado en
   s3://<bucket>/raw/<source>/_sync_state.json.
2. Si el HEAD indica un posible cambio (o no hay estado previo), descarga el
   contenido completo, calcula su SHA-256 y confirma si es un documento
   nuevo o actualizado.
3. Sube (PUT, aprovechando el versionado del bucket) unicamente los
   documentos nuevos/actualizados a s3://<bucket>/raw/<source>/..., y
   actualiza el estado de sincronizacion.
4. Escribe un resumen de la corrida en s3://<bucket>/<SUMMARY_S3_KEY>, que es
   consumido por el Step Function `DocumentSyncStateMachine` para actualizar
   el catalogo en DynamoDB y decidir si se debe disparar la preparacion de
   datos para el fine-tuning.

Disenado para ejecutarse como tarea de Amazon ECS Fargate (ver
data_pipeline/sync/Dockerfile e infra/stacks/document_sync_stack.py).

Variables de entorno esperadas:
    DATA_BUCKET      Bucket S3 destino (obligatorio)
    SOURCES          Lista separada por comas de fuentes a sincronizar
                      (por defecto: "cnbv,banxico")
    SUMMARY_S3_KEY   Key donde se escribe el resumen de la corrida
                      (por defecto: "sync-runs/latest_summary.json")
"""
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scraping.scrape_cnbv import (  # noqa: E402
    SECTOR_PAGES as CNBV_SECTOR_PAGES,
    fetch_sector_documents as cnbv_fetch_sector_documents,
    make_session as cnbv_make_session,
)
from scraping.scrape_banxico import (  # noqa: E402
    iter_documents as banxico_iter_documents,
    make_session as banxico_make_session,
)
from common.s3_utils import get_s3_client, read_json_if_exists, write_json  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sync_documents")

REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_REQUESTS = 0.2
DEFAULT_SUMMARY_S3_KEY = "sync-runs/latest_summary.json"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sync_state_key(source: str) -> str:
    return f"raw/{source}/_sync_state.json"


def _record_url(source: str, record) -> str:
    return record.url if source == "cnbv" else record.pdf_url


def _head_looks_unchanged(head_meta: dict, previous: dict) -> bool:
    """Heuristica barata para decidir si un documento probablemente no
    cambio, sin descargarlo por completo: compara Content-Length y, si esta
    disponible, ETag/Last-Modified contra lo observado en la corrida
    anterior. Si no hay suficiente informacion, se asume que SI pudo haber
    cambiado (mejor sobre-verificar que perder una actualizacion)."""
    if not previous:
        return False
    if head_meta.get("content_length") is None:
        return False
    if str(head_meta.get("content_length")) != str(previous.get("size")):
        return False
    if head_meta.get("etag") and previous.get("etag"):
        return head_meta["etag"] == previous["etag"]
    if head_meta.get("last_modified") and previous.get("last_modified"):
        return head_meta["last_modified"] == previous["last_modified"]
    # Mismo tamano, sin ETag/Last-Modified para comparar: se acepta como
    # senal suficiente para evitar re-descargar el corpus completo cada
    # semana; la verificacion criptografica ocurre igual la primera vez que
    # cambie el tamano.
    return True


def _head_document(session, url: str) -> dict:
    try:
        resp = session.head(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if resp.status_code >= 400:
            return {}
        return {
            "content_length": resp.headers.get("Content-Length"),
            "etag": resp.headers.get("ETag"),
            "last_modified": resp.headers.get("Last-Modified"),
        }
    except Exception as exc:
        logger.warning("HEAD fallido para %s: %s", url, exc)
        return {}


def sync_source_documents(s3, bucket: str, source: str, session, records) -> dict:
    state_key = _sync_state_key(source)
    state = read_json_if_exists(bucket, state_key, s3=s3) or {}

    changed_docs = []
    catalog_records = []
    seen_urls = set()
    total_new, total_updated, total_unchanged, total_failed = 0, 0, 0, 0

    for record in records:
        url = _record_url(source, record)
        if url in seen_urls:
            continue
        seen_urls.add(url)

        s3_key = f"raw/{source}/{record.local_path}".replace("\\", "/")
        previous = state.get(s3_key)

        head_meta = _head_document(session, url)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

        if previous and _head_looks_unchanged(head_meta, previous):
            total_unchanged += 1
            catalog_records.append(_catalog_entry(source, s3_key, url, record, "unchanged", previous))
            continue

        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except Exception as exc:
            logger.error("Error descargando %s: %s", url, exc)
            total_failed += 1
            continue
        time.sleep(SLEEP_BETWEEN_REQUESTS)

        content = resp.content
        content_hash = _sha256(content)

        if previous and previous.get("sha256") == content_hash:
            total_unchanged += 1
            catalog_records.append(_catalog_entry(source, s3_key, url, record, "unchanged", previous))
            continue

        status = "new" if previous is None else "updated"
        if status == "new":
            total_new += 1
        else:
            total_updated += 1

        put_resp = s3.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=content,
            ContentType="application/pdf",
        )
        entry = {
            "sha256": content_hash,
            "size": len(content),
            "etag": head_meta.get("etag"),
            "last_modified": head_meta.get("last_modified"),
            "version_id": put_resp.get("VersionId", "null"),
            "last_synced_at": datetime.now(timezone.utc).isoformat(),
        }
        state[s3_key] = entry

        record_dict = asdict(record)
        changed_docs.append({"source": source, "s3_key": s3_key, "status": status, "url": url, **record_dict})
        catalog_records.append(_catalog_entry(source, s3_key, url, record, status, entry))
        logger.info("%s: %s (%s, %d bytes)", status.upper(), s3_key, source, len(content))

    write_json(bucket, state_key, state, s3=s3)

    return {
        "source": source,
        "total_documents": len(seen_urls),
        "total_new": total_new,
        "total_updated": total_updated,
        "total_unchanged": total_unchanged,
        "total_failed": total_failed,
        "changed_docs": changed_docs,
        "catalog_records": catalog_records,
    }


def _catalog_entry(source: str, s3_key: str, url: str, record, status: str, state_entry: dict) -> dict:
    record_dict = asdict(record)
    return {
        "doc_id": f"{source}#{s3_key}",
        "source": source,
        "s3_key": s3_key,
        "s3_version_id": state_entry.get("version_id", ""),
        "url": url,
        "status": status,
        "last_synced_at": state_entry.get("last_synced_at", ""),
        **record_dict,
    }


def discover_cnbv_records(session):
    for sector, url in CNBV_SECTOR_PAGES.items():
        try:
            yield from cnbv_fetch_sector_documents(session, sector, url)
        except Exception as exc:
            logger.error("Error obteniendo indice de sector %s: %s", sector, exc)


def discover_banxico_records(session):
    yield from banxico_iter_documents(session)


def run(bucket: str, sources: list, summary_s3_key: str) -> dict:
    s3 = get_s3_client()
    results = {}

    for source in sources:
        logger.info("Sincronizando fuente: %s", source)
        if source == "cnbv":
            session = cnbv_make_session()
            records = discover_cnbv_records(session)
        elif source == "banxico":
            session = banxico_make_session()
            records = discover_banxico_records(session)
        else:
            logger.warning("Fuente desconocida, se omite: %s", source)
            continue

        results[source] = sync_source_documents(s3, bucket, source, session, records)

    any_changes = any(r["total_new"] > 0 or r["total_updated"] > 0 for r in results.values())
    all_changed_docs = [d for r in results.values() for d in r["changed_docs"]]
    all_catalog_records = [c for r in results.values() for c in r["catalog_records"]]

    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "any_changes": any_changes,
        "sources": {
            src: {k: v for k, v in r.items() if k not in ("changed_docs", "catalog_records")}
            for src, r in results.items()
        },
        "changed_docs": all_changed_docs,
        "catalog_records": all_catalog_records,
    }

    write_json(bucket, summary_s3_key, summary, s3=s3)
    logger.info(
        "Sincronizacion finalizada. any_changes=%s, total_changed_docs=%d, total_catalog_records=%d",
        any_changes,
        len(all_changed_docs),
        len(all_catalog_records),
    )
    return summary


def main():
    bucket = os.environ["DATA_BUCKET"]
    sources = [s.strip() for s in os.environ.get("SOURCES", "cnbv,banxico").split(",") if s.strip()]
    summary_s3_key = os.environ.get("SUMMARY_S3_KEY", DEFAULT_SUMMARY_S3_KEY)
    summary = run(bucket, sources, summary_s3_key)
    print(json.dumps({"any_changes": summary["any_changes"]}))


if __name__ == "__main__":
    main()
