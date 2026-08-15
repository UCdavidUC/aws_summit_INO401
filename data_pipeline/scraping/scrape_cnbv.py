"""
Scraper de documentos de normatividad de la CNBV (Comision Nacional Bancaria
y de Valores).

Recorre las paginas de "Normatividad" de cada sector supervisado por la
CNBV. Cada pagina contiene una tabla HTML (generada por el webpart
"Cnbv.Webpart.Normatividad") con una fila por norma, con metadata (nombre,
tipo, fecha de publicacion en el DOF, sectores a los que aplica) y un enlace
directo de descarga al PDF.

Adicionalmente, para cada norma con un ID valido (> 0) se consulta el
endpoint AJAX de "Resoluciones y Anexos" para descubrir PDFs relacionados
(resoluciones modificatorias y anexos), que tambien son de valor para el
corpus de fine-tuning.

Uso:
    python scrape_cnbv.py --out-dir ./raw_cnbv [--upload-s3 s3://bucket/raw/cnbv/]

Salida:
    <out-dir>/<sector>/<slug-del-documento>.pdf
    <out-dir>/metadata.jsonl   (una linea JSON por documento descargado)
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

import requests
from bs4 import BeautifulSoup

try:
    from ca_bundle import get_ca_bundle_path
except ImportError:  # ejecucion como script suelto fuera del paquete
    from scraping.ca_bundle import get_ca_bundle_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scrape_cnbv")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_REQUESTS = 0.5

# Paginas de "Normatividad" por sector supervisado. Cada URL renderiza una
# tabla HTML con las normas de ese sector y su PDF de descarga.
SECTOR_PAGES = {
    "asesores_en_inversiones": "https://www.cnbv.gob.mx/SECTORES-SUPERVISADOS/ASESORES_EN_INVERSIONES/Paginas/Normatividad.aspx",
    "banca_multiple": "https://www.cnbv.gob.mx/SECTORES-SUPERVISADOS/BANCA-MULTIPLE/paginas/normatividad.aspx",
    "participantes_redes_medios_disposicion": "https://www.cnbv.gob.mx/SECTORES-SUPERVISADOS/PARTICIPANTES_EN_REDES_DE_MEDIOS_DE_DISPOSICI%C3%93N/Paginas/Normatividad.aspx",
    "sociedades_de_inversion": "https://www.cnbv.gob.mx/SECTORES-SUPERVISADOS/SOCIEDADES-DE-INVERSION/Paginas/Normatividad.aspx",
    "uniones_de_credito": "https://www.cnbv.gob.mx/SECTORES-SUPERVISADOS/UNIONES-DE-CREDITO/Paginas/Normatividad.aspx",
    "banca_de_desarrollo": "https://www.cnbv.gob.mx/SECTORES-SUPERVISADOS/BANCA-DE-DESARROLLO/Normatividad/Paginas/Banca-de-Desarrollo.aspx",
    "bursatil": "https://www.cnbv.gob.mx/SECTORES-SUPERVISADOS/BURS%C3%81TIL/Normatividad/Paginas/Casas-de-Bolsa.aspx",
    "otros_supervisados": "https://www.cnbv.gob.mx/SECTORES-SUPERVISADOS/OTROS-SUPERVISADOS/Normatividad/Paginas/Organizaciones-y-Actividades-Auxiliares-de-Cr%C3%A9dito.aspx",
    "sector_popular": "https://www.cnbv.gob.mx/SECTORES-SUPERVISADOS/SECTOR-POPULAR/Normatividad/Paginas/Sociedades-Cooperativas-de-Ahorro-y-Pr%C3%A9stamo.aspx",
}

RESOLUCIONES_ANEXOS_ENDPOINT = (
    "https://www.cnbv.gob.mx/_vti_bin/Cnbv.Webpart.Normatividad/"
    "NormatividadAjax.svc/ResolucionesYAnexos"
)


@dataclass
class DocumentRecord:
    source: str
    sector: str
    doc_id: str
    nombre: str
    tipo: str
    fecha_publicacion_dof: str
    sectores_aplicables: list
    url: str
    local_path: str
    kind: str  # "principal" | "resolucion" | "anexo"


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
    # www.cnbv.gob.mx no envia el certificado intermedio de su CA; ver
    # scraping/ca_bundle.py para el detalle.
    session.verify = get_ca_bundle_path()
    return session


def fetch_sector_documents(
    session: requests.Session, sector: str, url: str
) -> Iterator[DocumentRecord]:
    logger.info("Descargando indice de normatividad: %s (%s)", sector, url)
    resp = session.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table", id="normatividadTable")
    if table is None:
        logger.warning("No se encontro tabla de normatividad en %s", url)
        return

    body = table.find("tbody")
    if body is None:
        return

    for row in body.find_all("tr", recursive=False):
        cells = row.find_all("td", recursive=False)
        if len(cells) < 6:
            continue

        doc_id = cells[0].get_text(strip=True)
        nombre = cells[1].get_text(strip=True)
        tipo = cells[2].get_text(strip=True)
        fecha = cells[3].get_text(strip=True)
        sectores = [li.get_text(strip=True) for li in cells[4].find_all("li")]
        link_tag = cells[5].find("a", href=True)

        if link_tag is None:
            continue

        pdf_url = link_tag["href"]
        slug = slugify(nombre)
        local_path = os.path.join(sector, f"{slug}.pdf")

        yield DocumentRecord(
            source="cnbv",
            sector=sector,
            doc_id=doc_id,
            nombre=nombre,
            tipo=tipo,
            fecha_publicacion_dof=fecha,
            sectores_aplicables=sectores,
            url=pdf_url,
            local_path=local_path,
            kind="principal",
        )

        if doc_id and doc_id != "-1":
            yield from fetch_related_documents(session, sector, doc_id, nombre)


def fetch_related_documents(
    session: requests.Session, sector: str, doc_id: str, base_nombre: str
) -> Iterator[DocumentRecord]:
    """Consulta el endpoint AJAX de resoluciones y anexos para un doc_id
    dado, y produce un DocumentRecord por cada PDF adicional encontrado."""
    try:
        resp = session.get(
            RESOLUCIONES_ANEXOS_ENDPOINT,
            params={"normaId": doc_id},
            timeout=REQUEST_TIMEOUT,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning(
            "No se pudieron obtener resoluciones/anexos para doc_id=%s (%s): %s",
            doc_id,
            base_nombre,
            exc,
        )
        return

    for kind_key, kind_label in (("Resoluciones", "resolucion"), ("Anexos", "anexo")):
        for item in data.get(kind_key, []) or []:
            url = item.get("URL")
            descripcion = item.get("Descripcion") or f"{kind_label}-{doc_id}"
            if not url:
                continue
            slug = slugify(f"{base_nombre}-{descripcion}")
            local_path = os.path.join(sector, f"{slug}.pdf")
            yield DocumentRecord(
                source="cnbv",
                sector=sector,
                doc_id=doc_id,
                nombre=descripcion,
                tipo=kind_label,
                fecha_publicacion_dof="",
                sectores_aplicables=[],
                url=url,
                local_path=local_path,
                kind=kind_label,
            )


def download_document(
    session: requests.Session, record: DocumentRecord, out_dir: str
) -> Optional[DocumentRecord]:
    dest_path = os.path.join(out_dir, record.local_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        logger.info("Ya existe, se omite: %s", dest_path)
        return record

    try:
        resp = session.get(record.url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type.lower() and not record.url.lower().endswith(".pdf"):
            logger.warning(
                "Contenido no-PDF (%s) para %s, se descarta", content_type, record.url
            )
            return None
        with open(dest_path, "wb") as f:
            f.write(resp.content)
        logger.info("Descargado: %s -> %s (%d bytes)", record.url, dest_path, len(resp.content))
        return record
    except Exception as exc:
        logger.error("Error descargando %s: %s", record.url, exc)
        return None


def run(out_dir: str, sectors: Optional[list] = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    session = make_session()

    metadata_path = os.path.join(out_dir, "metadata.jsonl")
    seen_urls = set()
    total_ok = 0
    total_fail = 0

    target_sectors = sectors or list(SECTOR_PAGES.keys())

    with open(metadata_path, "w", encoding="utf-8") as meta_f:
        for sector in target_sectors:
            url = SECTOR_PAGES[sector]
            try:
                records = list(fetch_sector_documents(session, sector, url))
            except Exception as exc:
                logger.error("Error obteniendo indice de sector %s: %s", sector, exc)
                continue

            for record in records:
                if record.url in seen_urls:
                    continue
                seen_urls.add(record.url)

                downloaded = download_document(session, record, out_dir)
                time.sleep(SLEEP_BETWEEN_REQUESTS)

                if downloaded is not None:
                    meta_f.write(json.dumps(asdict(downloaded), ensure_ascii=False) + "\n")
                    total_ok += 1
                else:
                    total_fail += 1

    logger.info(
        "Scraping CNBV finalizado. Documentos descargados: %d, fallidos: %d",
        total_ok,
        total_fail,
    )
    return metadata_path


def main():
    parser = argparse.ArgumentParser(description="Scraper de normatividad CNBV")
    parser.add_argument("--out-dir", default="./raw_cnbv", help="Directorio local de salida")
    parser.add_argument(
        "--sectors",
        nargs="*",
        default=None,
        help="Subconjunto de sectores a descargar (por defecto, todos)",
    )
    args = parser.parse_args()
    run(args.out_dir, args.sectors)


if __name__ == "__main__":
    main()
