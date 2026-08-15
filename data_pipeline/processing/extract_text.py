"""
Extraccion de texto de los PDFs descargados de CNBV y Banxico.

Lee los PDFs desde un directorio local (ver scraping/scrape_cnbv.py y
scrape_banxico.py), extrae el texto de cada uno con `pypdf`, aplica una
limpieza basica (normalizacion de espacios, eliminacion de saltos de pagina
repetidos) y escribe el resultado como archivos de texto plano en
processed/text/<source>/<slug>.txt, junto con un manifest.jsonl que registra
metadatos (fuente, documento, num_paginas, num_caracteres, ruta).

Uso:
    python extract_text.py --raw-dir ../raw_cnbv --source cnbv --out-dir ../processed/text
    python extract_text.py --raw-dir ../raw_banxico --source banxico --out-dir ../processed/text
"""
import argparse
import json
import logging
import os
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("extract_text")

MIN_CHARS_VALID = 200  # documentos con menos texto que esto se consideran vacios/escaneados


def clean_text(raw_text: str) -> str:
    text = raw_text.replace("\x0c", "\n")  # form feed -> salto de linea
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +\n", "\n", text)
    return text.strip()


def extract_pdf_text(pdf_path: str) -> tuple:
    """Devuelve (texto_limpio, num_paginas). Lanza excepcion si el PDF esta
    corrupto o no se puede abrir."""
    reader = PdfReader(pdf_path)
    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception as exc:
            logger.warning("Error extrayendo pagina de %s: %s", pdf_path, exc)
    full_text = "\n\n".join(pages_text)
    return clean_text(full_text), len(reader.pages)


def process_directory(raw_dir: str, source: str, out_dir: str) -> str:
    text_out_dir = os.path.join(out_dir, source)
    os.makedirs(text_out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, f"manifest_{source}.jsonl")

    pdf_paths = []
    for root, _, files in os.walk(raw_dir):
        for fname in files:
            if fname.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, fname))

    logger.info("Encontrados %d PDFs en %s", len(pdf_paths), raw_dir)

    total_ok, total_empty, total_error = 0, 0, 0

    with open(manifest_path, "w", encoding="utf-8") as manifest_f:
        for pdf_path in sorted(pdf_paths):
            rel_path = os.path.relpath(pdf_path, raw_dir)
            slug = re.sub(r"[/\\]", "__", rel_path)[:-4]  # sin extension .pdf
            txt_path = os.path.join(text_out_dir, f"{slug}.txt")

            try:
                text, num_pages = extract_pdf_text(pdf_path)
            except (PdfReadError, Exception) as exc:
                logger.error("Error abriendo %s: %s", pdf_path, exc)
                total_error += 1
                continue

            if len(text) < MIN_CHARS_VALID:
                logger.warning(
                    "Texto insuficiente (%d chars) en %s, probablemente escaneado sin OCR",
                    len(text),
                    pdf_path,
                )
                total_empty += 1
                continue

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            manifest_f.write(
                json.dumps(
                    {
                        "source": source,
                        "original_pdf": rel_path,
                        "text_path": os.path.relpath(txt_path, out_dir),
                        "num_pages": num_pages,
                        "num_chars": len(text),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            total_ok += 1

    logger.info(
        "Extraccion de %s finalizada. OK=%d, vacios/escaneados=%d, errores=%d",
        source,
        total_ok,
        total_empty,
        total_error,
    )
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Extraccion de texto de PDFs CNBV/Banxico")
    parser.add_argument("--raw-dir", required=True, help="Directorio con los PDFs de origen")
    parser.add_argument("--source", required=True, choices=["cnbv", "banxico"])
    parser.add_argument("--out-dir", default="../processed/text", help="Directorio de salida")
    args = parser.parse_args()
    process_directory(args.raw_dir, args.source, args.out_dir)


if __name__ == "__main__":
    main()
