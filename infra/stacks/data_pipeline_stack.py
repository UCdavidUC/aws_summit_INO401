"""
DataPipelineStack
------------------
Recursos de datos para el pipeline de fine-tuning del SLM de cumplimiento
regulatorio CNBV/Banxico:

- Bucket S3 con prefijos logicos: raw/, processed/, datasets/, models/
- Rol IAM para procesos locales/jobs que necesitan leer/escribir en el bucket
  (usado por los scripts de scraping, procesamiento y por el training job de
  SageMaker a traves de TrainingStack).

El bucket tiene versionado y bloqueo de acceso publico habilitados por
defecto, y cifrado S3-managed (SSE-S3).

Nota sobre KMS: algunos objetos escritos por SageMaker (p.ej. el artefacto
`model.tar.gz` de un Training Job) quedan cifrados con SSE-KMS usando la
llave administrada por AWS `aws/s3`, en vez de con el cifrado SSE-S3 por
defecto del bucket. Sin `kms:Decrypt` en la identity policy, S3 devuelve un
404 generico (no un 403/AccessDenied) al intentar leer esos objetos, aunque
el objeto exista y el caller tenga `s3:GetObject`. Por eso
`pipeline_access_policy` incluye el patron estandar de AWS para acceso a
`aws/s3`: permitir las acciones KMS con `Resource: "*"`, acotado por
`kms:ViaService` a S3 en esta cuenta/region (la llave `aws/s3` es
administrada por AWS y no tiene un ARN fijo utilizable en el codigo).
"""
from aws_cdk import (
    Stack,
    RemovalPolicy,
    aws_s3 as s3,
    aws_iam as iam,
    CfnOutput,
)
from constructs import Construct


class DataPipelineStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.data_bucket = s3.Bucket(
            self,
            "SlmDataBucket",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="abort-incomplete-multipart-uploads",
                    abort_incomplete_multipart_upload_after=None,
                    enabled=True,
                    expired_object_delete_marker=False,
                )
            ],
        )

        # Rol asumible por operadores/roles locales o por otros servicios
        # (p.ej. el rol de ejecucion de SageMaker) para leer y escribir en
        # las carpetas del pipeline de datos.
        self.pipeline_access_policy = iam.ManagedPolicy(
            self,
            "SlmDataPipelineAccessPolicy",
            description="Acceso de lectura/escritura al bucket de datos del pipeline SLM CNBV/Banxico",
            statements=[
                iam.PolicyStatement(
                    sid="ListBucket",
                    actions=["s3:ListBucket"],
                    resources=[self.data_bucket.bucket_arn],
                ),
                iam.PolicyStatement(
                    sid="ReadWriteObjects",
                    actions=[
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:DeleteObject",
                    ],
                    resources=[self.data_bucket.arn_for_objects("*")],
                ),
                # Descifrar/cifrar objetos SSE-KMS escritos con la llave
                # administrada por AWS `aws/s3` (ver nota en el docstring del
                # modulo). El condicional `kms:ViaService` limita el permiso
                # a llamadas hechas a traves de S3 en esta cuenta/region, que
                # es el patron documentado por AWS para este escenario.
                iam.PolicyStatement(
                    sid="S3ManagedKmsKeyAccess",
                    actions=[
                        "kms:Decrypt",
                        "kms:GenerateDataKey*",
                        "kms:DescribeKey",
                    ],
                    resources=["*"],
                    conditions={
                        "StringEquals": {
                            "kms:ViaService": f"s3.{Stack.of(self).region}.amazonaws.com",
                            "kms:CallerAccount": Stack.of(self).account,
                        }
                    },
                ),
            ],
        )

        CfnOutput(
            self,
            "DataBucketName",
            value=self.data_bucket.bucket_name,
            description="Bucket S3 con los prefijos raw/, processed/, datasets/, models/",
        )
        CfnOutput(
            self,
            "PipelineAccessPolicyArn",
            value=self.pipeline_access_policy.managed_policy_arn,
            description=(
                "ARN de la managed policy con acceso S3+KMS al bucket de datos "
                "(adjuntar al rol de ejecucion de Studio para leer artefactos SSE-KMS)"
            ),
        )
