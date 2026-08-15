# Contenedor de sincronizacion documental CNBV/Banxico, ejecutado como tarea
# de Amazon ECS Fargate por el Step Function DocumentSyncStateMachine (ver
# infra/stacks/document_sync_stack.py).
FROM python:3.12-slim

WORKDIR /app

COPY sync/requirements.txt ./sync-requirements.txt
RUN pip install --no-cache-dir -r sync-requirements.txt

# Se copia todo data_pipeline/ (no solo sync/) porque sync_documents.py
# reutiliza los modulos de scraping/ (descubrimiento de documentos) y
# common/ (utilidades S3).
COPY common ./common
COPY scraping ./scraping
COPY certs ./certs
COPY sync ./sync

ENV PYTHONUNBUFFERED=1

CMD ["python", "sync/sync_documents.py"]
