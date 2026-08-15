#!/usr/bin/env python3
import os

import aws_cdk as cdk

from stacks.data_pipeline_stack import DataPipelineStack
from stacks.training_stack import TrainingStack
from stacks.agent_runtime_stack import AgentRuntimeStack
from stacks.document_sync_stack import DocumentSyncStack

# Region fija: us-west-2. Es donde Amazon Bedrock AgentCore Runtime esta
# soportado y donde la cuenta ya tiene cuotas de GPU (ml.g6.*) habilitadas
# para el SageMaker Training Job. mx-central-1 no soporta AgentCore Runtime
# ni Custom Model Import al momento de este diseno; ver
# docs/portability_mx_central_1.md.
REGION = "us-west-2"

app = cdk.App()

env = cdk.Environment(
    account=os.getenv("CDK_DEFAULT_ACCOUNT"),
    region=REGION,
)

data_pipeline_stack = DataPipelineStack(
    app,
    "SlmDataPipelineStack",
    env=env,
    description="Buckets S3 y roles IAM para el pipeline de datos del SLM CNBV/Banxico",
)

training_stack = TrainingStack(
    app,
    "SlmTrainingStack",
    data_pipeline_stack=data_pipeline_stack,
    env=env,
    description="Rol de ejecucion de SageMaker para el fine-tuning QLoRA del SLM CNBV/Banxico",
)

agent_runtime_stack = AgentRuntimeStack(
    app,
    "SlmAgentRuntimeStack",
    data_pipeline_stack=data_pipeline_stack,
    env=env,
    description="Bedrock AgentCore Runtime de prueba para el SLM CNBV/Banxico fine-tuneado",
)

document_sync_stack = DocumentSyncStack(
    app,
    "SlmDocumentSyncStack",
    data_pipeline_stack=data_pipeline_stack,
    env=env,
    description="Step Function semanal que sincroniza documentos CNBV/Banxico, actualiza el catalogo DynamoDB y dispara la preparacion de datos",
)

app.synth()
