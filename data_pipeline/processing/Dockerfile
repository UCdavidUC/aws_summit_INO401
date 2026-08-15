# Contenedor de preparacion de datos (extraccion de texto + chunking) para
# el fine-tuning del SLM, ejecutado como tarea de Amazon ECS Fargate por el
# Step Function DocumentSyncStateMachine tras detectar documentos
# nuevos/actualizados (ver infra/stacks/document_sync_stack.py).
FROM python:3.12-slim

WORKDIR /app

COPY processing/requirements.txt ./processing-requirements.txt
RUN pip install --no-cache-dir -r processing-requirements.txt

COPY common ./common
COPY processing ./processing

ENV PYTHONUNBUFFERED=1

CMD ["python", "processing/run_data_prep.py"]
