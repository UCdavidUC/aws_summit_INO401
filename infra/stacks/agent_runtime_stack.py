"""
AgentRuntimeStack
------------------
Despliegue de prueba del SLM fine-tuneado (Qwen2.5-1.5B-Instruct + adaptador
QLoRA fusionado) sobre Amazon Bedrock AgentCore Runtime, en us-west-2 (region
donde AgentCore Runtime esta soportado; mx-central-1 no lo esta al momento de
este diseno, ver docs/portability_mx_central_1.md).

La imagen del agente se construye desde el directorio local `agent_runtime/`
(Dockerfile) usando `AgentRuntimeArtifact.from_asset`, lo que hace que el CDK
toolkit construya la imagen (arquitectura arm64, requerida por AgentCore
Runtime) y la publique en un repositorio ECR administrado por CDK.
"""
import os

from aws_cdk import (
    Stack,
    Duration,
    aws_iam as iam,
    aws_bedrockagentcore as agentcore,
    CfnOutput,
)
from constructs import Construct

from stacks.data_pipeline_stack import DataPipelineStack

AGENT_RUNTIME_CODE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "agent_runtime"
)


class AgentRuntimeStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        data_pipeline_stack: DataPipelineStack,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        agent_runtime_artifact = agentcore.AgentRuntimeArtifact.from_asset(
            AGENT_RUNTIME_CODE_DIR
        )

        self.runtime = agentcore.Runtime(
            self,
            "SlmComplianceAgentRuntime",
            runtime_name="cnbv_banxico_slm_agent",
            agent_runtime_artifact=agent_runtime_artifact,
            description=(
                "Runtime de prueba para el SLM (Qwen2.5-1.5B fine-tuneado) "
                "especializado en cumplimiento regulatorio CNBV/Banxico"
            ),
            environment_variables={
                "MODEL_BUCKET": data_pipeline_stack.data_bucket.bucket_name,
                "MODEL_S3_PREFIX": "models/latest/",
            },
            lifecycle_configuration=agentcore.LifecycleConfiguration(
                idle_runtime_session_timeout=Duration.minutes(15),
                max_lifetime=Duration.hours(8),
            ),
        )

        # El runtime necesita descargar el modelo fine-tuneado (adaptador
        # fusionado) desde el bucket de datos al iniciar el contenedor.
        self.runtime.add_to_role_policy(
            iam.PolicyStatement(
                sid="ReadModelArtifacts",
                actions=["s3:GetObject", "s3:ListBucket"],
                resources=[
                    data_pipeline_stack.data_bucket.bucket_arn,
                    data_pipeline_stack.data_bucket.arn_for_objects("models/*"),
                ],
            )
        )

        CfnOutput(
            self,
            "AgentRuntimeArn",
            value=self.runtime.agent_runtime_arn,
            description="ARN del AgentCore Runtime de prueba",
        )
        CfnOutput(
            self,
            "AgentRuntimeId",
            value=self.runtime.agent_runtime_id,
            description="ID del AgentCore Runtime de prueba",
        )
