"""
TrainingStack
-------------
Rol de ejecucion de SageMaker para el Training Job de fine-tuning (QLoRA)
del SLM Qwen2.5-1.5B-Instruct sobre el corpus CNBV/Banxico.

El lanzamiento del Training Job en si se realiza fuera de CDK (via
`training/launch_training_job.py` con boto3/SageMaker SDK), porque un
training job es una ejecucion puntual y no un recurso de infraestructura de
larga duracion. CDK provisiona unicamente el rol de ejecucion y los permisos
necesarios.

Ademas, este stack crea un secreto en Secrets Manager (`slm/notebook-config`)
con las variables que el notebook necesita para operar (bucket de datos, ARN
del rol de entrenamiento, cuenta y region). Esto reemplaza el patron anterior
de un archivo `.env` local: el notebook lee estos valores directamente desde
Secrets Manager (Seccion 0.1), y el acceso se controla por IAM en vez de por
un archivo en disco.
"""
import json

from aws_cdk import (
    Stack,
    SecretValue,
    aws_iam as iam,
    aws_secretsmanager as secretsmanager,
    CfnOutput,
)
from constructs import Construct

from stacks.data_pipeline_stack import DataPipelineStack

NOTEBOOK_CONFIG_SECRET_NAME = "slm/notebook-config"


class TrainingStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        data_pipeline_stack: DataPipelineStack,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.execution_role = iam.Role(
            self,
            "SlmSageMakerTrainingRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Rol de ejecucion para el SageMaker Training Job de fine-tuning QLoRA del SLM CNBV/Banxico",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess"
                ),
                data_pipeline_stack.pipeline_access_policy,
            ],
        )

        # Permite al job leer/escribir logs y metricas en CloudWatch
        # (ya cubierto por AmazonSageMakerFullAccess, se deja explicito por
        # claridad y por si se decide reducir el alcance de la managed policy
        # en el futuro).
        self.execution_role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudWatchLogsAndMetrics",
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                    "cloudwatch:PutMetricData",
                ],
                resources=["*"],
            )
        )

        CfnOutput(
            self,
            "TrainingExecutionRoleArn",
            value=self.execution_role.role_arn,
            description="ARN del rol de ejecucion para el SageMaker Training Job",
        )

        # ------------------------------------------------------------------
        # Secreto con las variables que el notebook necesita (Seccion 0.1):
        # sustituye al archivo .env local. El valor se fija en el momento del
        # despliegue con los recursos ya creados (bucket, rol), por lo que no
        # requiere una actualizacion manual posterior.
        # ------------------------------------------------------------------
        self.notebook_config_secret = secretsmanager.Secret(
            self,
            "SlmNotebookConfigSecret",
            secret_name=NOTEBOOK_CONFIG_SECRET_NAME,
            description=(
                "Variables de configuracion del notebook de fine-tuning SLM "
                "CNBV/Banxico (bucket de datos, rol de entrenamiento, cuenta, region)"
            ),
            secret_string_value=SecretValue.unsafe_plain_text(
                json.dumps(
                    {
                        "data_bucket_name": data_pipeline_stack.data_bucket.bucket_name,
                        "training_role_arn": self.execution_role.role_arn,
                        "account_id": Stack.of(self).account,
                        "region": Stack.of(self).region,
                    }
                )
            ),
        )

        # Managed policy para adjuntar al rol de ejecucion de SageMaker
        # Studio (o cualquier otro consumidor) que necesite leer este secreto.
        self.notebook_config_read_policy = iam.ManagedPolicy(
            self,
            "SlmNotebookConfigSecretReadPolicy",
            description="Permite leer el secreto de configuracion del notebook SLM CNBV/Banxico",
            statements=[
                iam.PolicyStatement(
                    sid="ReadNotebookConfigSecret",
                    actions=["secretsmanager:GetSecretValue"],
                    resources=[self.notebook_config_secret.secret_arn],
                )
            ],
        )

        CfnOutput(
            self,
            "NotebookConfigSecretArn",
            value=self.notebook_config_secret.secret_arn,
            description="ARN del secreto de Secrets Manager con la configuracion del notebook",
        )
        CfnOutput(
            self,
            "NotebookConfigSecretName",
            value=NOTEBOOK_CONFIG_SECRET_NAME,
            description="Nombre del secreto de Secrets Manager con la configuracion del notebook",
        )
        CfnOutput(
            self,
            "NotebookConfigSecretReadPolicyArn",
            value=self.notebook_config_read_policy.managed_policy_arn,
            description=(
                "ARN de la managed policy que otorga lectura del secreto "
                "slm/notebook-config (adjuntar al rol de ejecucion de Studio)"
            ),
        )
