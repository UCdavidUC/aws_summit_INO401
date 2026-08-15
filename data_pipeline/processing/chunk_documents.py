"""
Chunking de los documentos de texto extraidos (CNBV/Banxico) en fragmentos
de tamano manejable para servir de contexto a la generacion de pares de
instruccion (build_dataset.py).

Estrategia: split por parrafos (doble salto de linea), acumulando parrafos
hasta un tamano objetivo (~3000 caracteres), sin cortar parrafos a la mitad.
Fragmentos menores al minimo se descartan (ruido, encabezados sueltos).

Uso:
    python chunk_documents.py --text-dir ../processed/text --out ../processed/chunks.jsonl
"""
import argparse
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("chunk_documents")

TARGET_CHUNK_CHARS = 3000
MIN_CHUNK_CHARS = 400
MAX_CHUNK_CHARS = 4500


def split_into_chunks(text: str) -> list:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > MAX_CHUNK_CHARS and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

        current.append(para)
        current_len += len(para) + 2

        if current_len >= TARGET_CHUNK_CHARS:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

    if current:
        chunks.append("\n\n".join(current))

    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


def run(text_dir: str, out_path: str):
    manifests = [
        os.path.join(text_dir, "manifest_cnbv.jsonl"),
        os.path.join(text_dir, "manifest_banxico.jsonl"),
    ]

    total_chunks = 0
    total_docs = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for manifest_path in manifests:
            if not os.path.exists(manifest_path):
                continue
            with open(manifest_path, "r", encoding="utf-8") as manifest_f:
                for line in manifest_f:
                    doc = json.loads(line)
                    full_text_path = os.path.join(text_dir, os.path.basename(doc["text_path"]))
                    # text_path en el manifest es relativo a --out-dir de extract_text.py,
                    # que es el mismo text_dir aqui; pero incluye el subdirectorio source/.
                    full_text_path = os.path.join(text_dir, doc["source"], os.path.basename(doc["text_path"]))
                    if not os.path.exists(full_text_path):
                        # fallback: el text_path ya es relativo correcto
                        full_text_path = os.path.join(text_dir, doc["text_path"])

                    if not os.path.exists(full_text_path):
                        logger.warning("No se encontro archivo de texto: %s", full_text_path)
                        continue

                    with open(full_text_path, "r", encoding="utf-8") as tf:
                        text = tf.read()

                    chunks = split_into_chunks(text)
                    total_docs += 1

                    for i, chunk in enumerate(chunks):
                        out_f.write(
                            json.dumps(
                                {
                                    "source": doc["source"],
                                    "original_pdf": doc["original_pdf"],
                                    "chunk_index": i,
                                    "num_chunks_in_doc": len(chunks),
                                    "text": chunk,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        total_chunks += 1

    logger.info(
        "Chunking finalizado. Documentos procesados=%d, chunks generados=%d",
        total_docs,
        total_chunks,
    )


def main():
    parser = argparse.ArgumentParser(description="Chunking de documentos CNBV/Banxico")
    parser.add_argument("--text-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.text_dir, args.out)


if __name__ == "__main__":
    main()
