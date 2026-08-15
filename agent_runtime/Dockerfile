# Amazon Bedrock AgentCore Runtime requiere imagenes linux/arm64.
FROM --platform=linux/arm64 python:3.12-slim

WORKDIR /app

COPY requirements.txt .
# Se instala torch desde el indice CPU-only de PyTorch para evitar arrastrar
# las dependencias de CUDA/NVIDIA (varios GB), innecesarias en esta imagen
# ARM64 sin GPU (AgentCore Runtime no ofrece instancias con GPU).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8080

# bedrock_agentcore.BedrockAgentCoreApp expone /invocations (POST) y /ping
# (GET) en el puerto 8080, tal como lo requiere AgentCore Runtime.
CMD ["python", "app.py"]
