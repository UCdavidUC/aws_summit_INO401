"""
Scraper de documentos normativos de Banxico (Banco de Mexico).

Flujo en dos pasos:
1. Descarga la pagina indice cronologica de normativa emitida por el Banco de
   Mexico (agrupa ~300 circulares/disposiciones desde 1969 hasta la fecha),
   y extrae los enlaces a las "landing pages" de cada circular/disposicion.
2. Para cada landing page, extrae el enlace directo al PDF real de la norma
   y lo descarga.

Banxico responde HTTP 403 a clientes sin un User-Agent de navegador real, por
lo que se usa un User-Agent de Chrome. Ademas, banxico.org.mx no envia el
certificado intermedio de su CA (GoDaddy), por lo que se usa el mismo bundle
CA combinado que en scrape_cnbv.py (ver ca_bundle.py).

Uso:
    python scrape_banxico.py --out-dir ./raw_banxico

Salida:
    <out-dir>/<slug-del-documento>.pdf
    <out-dir>/metadata.jsonl
"""
import argparse
import json
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass, asdict
from typing import Iterator, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

try:
    from ca_bundle import get_ca_bundle_path
except ImportError:
    from scraping.ca_bundle import get_ca_bundle_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scrape_banxico")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_REQUESTS = 0.4

BASE_URL = "https://www.banxico.org.mx"
INDEX_URL = f"{BASE_URL}/marco-normativo/normativa-agrupada-por-ano-cr.html"
CIRCULAR_PATH_PREFIX = "/marco-normativo/normativa-emitida-por-el-banco-de-mexico/"


@dataclass
class DocumentRecord:
    source: str
    doc_id: str
    nombre: str
    landing_url: str
    pdf_url: str
    local_path: str


def slugify(text: str, max_len: int = 120) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[\s_]+", "-", text)
    return text[:max_len] or "documento"


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "es-MX,es;q=0.9",
        }
    )
    session.verify = get_ca_bundle_path()
    return session


def fetch_landing_page_urls(session: requests.Session) -> list:
    logger.info("Descargando indice cronologico de Banxico: %s", INDEX_URL)
    resp = session.get(INDEX_URL, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    urls = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith(CIRCULAR_PATH_PREFIX) and href.endswith(".html"):
            urls.add(urljoin(BASE_URL, href))

    logger.info("Landing pages de circulares/disposiciones encontradas: %d", len(urls))
    return sorted(urls)


def fetch_pdf_url_from_landing(
    session: requests.Session, landing_url: str
) -> Optional[str]:
    try:
        resp = session.get(landing_url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Error obteniendo landing page %s: %s", landing_url, exc)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            return urljoin(landing_url, href)
    return None


def build_document_name(landing_url: str) -> str:
    """Deriva un nombre legible a partir del slug de la URL de landing,
    p.ej. '.../circular-3-2025/operaciones-caja-corresponsalia-dispo.html'
    -> 'circular-3-2025-operaciones-caja-corresponsalia-dispo'."""
    path = landing_url[len(BASE_URL):]
    path = path[len(CIRCULAR_PATH_PREFIX):] if path.startswith(CIRCULAR_PATH_PREFIX) else path
    path = path.rstrip("/")
    parts = [p for p in path.split("/") if p]
    name = "-".join(parts)
    return name.replace(".html", "")


def iter_documents(session: requests.Session) -> Iterator[DocumentRecord]:
    landing_urls = fetch_landing_page_urls(session)

    for idx, landing_url in enumerate(landing_urls, start=1):
        pdf_url = fetch_pdf_url_from_landing(session, landing_url)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

        if not pdf_url:
            logger.warning("Sin PDF encontrado en landing page: %s", landing_url)
            continue

        nombre = build_document_name(landing_url)
        slug = slugify(nombre)
        local_path = f"{slug}.pdf"

        yield DocumentRecord(
            source="banxico",
            doc_id=str(idx),
            nombre=nombre,
            landing_url=landing_url,
            pdf_url=pdf_url,
            local_path=local_path,
        )


def download_document(
    session: requests.Session, record: DocumentRecord, out_dir: str
) -> Optional[DocumentRecord]:
    dest_path = os.path.join(out_dir, record.local_path)
    os.makedirs(os.path.dirname(dest_path) or out_dir, exist_ok=True)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        logger.info("Ya existe, se omite: %s", dest_path)
        return record

    try:
        resp = session.get(record.pdf_url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type.lower() and not record.pdf_url.lower().endswith(".pdf"):
            logger.warning(
                "Contenido no-PDF (%s) para %s, se descarta", content_type, record.pdf_url
            )
            return None
        with open(dest_path, "wb") as f:
            f.write(resp.content)
        logger.info(
            "Descargado: %s -> %s (%d bytes)", record.pdf_url, dest_path, len(resp.content)
        )
        return record
    except Exception as exc:
        logger.error("Error descargando %s: %s", record.pdf_url, exc)
        return None


def run(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    session = make_session()

    metadata_path = os.path.join(out_dir, "metadata.jsonl")
    total_ok = 0
    total_fail = 0

    with open(metadata_path, "w", encoding="utf-8") as meta_f:
        for record in iter_documents(session):
            downloaded = download_document(session, record, out_dir)
            time.sleep(SLEEP_BETWEEN_REQUESTS)

            if downloaded is not None:
                meta_f.write(json.dumps(asdict(downloaded), ensure_ascii=False) + "\n")
                total_ok += 1
            else:
                total_fail += 1

    logger.info(
        "Scraping Banxico finalizado. Documentos descargados: %d, fallidos: %d",
        total_ok,
        total_fail,
    )
    return metadata_path


def main():
    parser = argparse.ArgumentParser(description="Scraper de normativa de Banxico")
    parser.add_argument("--out-dir", default="./raw_banxico", help="Directorio local de salida")
    args = parser.parse_args()
    run(args.out_dir)


if __name__ == "__main__":
    main()
