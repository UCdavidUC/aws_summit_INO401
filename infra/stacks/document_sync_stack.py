"""
DocumentSyncStack
------------------
Step Function que mantiene sincronizado el corpus de documentos
regulatorios (CNBV/Banxico) con sus fuentes publicas originales, cataloga
cada documento en DynamoDB, y dispara la pipeline de preparacion de datos
para el fine-tuning cuando hay contenido nuevo o actualizado.

Se ejecuta automaticamente cada domingo a las 2:00 AM hora de Ciudad de
Mexico (UTC-6, sin horario de verano desde 2022) mediante una regla de
EventBridge con expresion cron en UTC.

Flujo del Step Function `DocumentSyncStateMachine`:

    1. SyncDocuments (ECS Fargate)
       Recorre las paginas indice de CNBV y Banxico (mismas fuentes que
       data_pipeline/scraping/), compara cada documento contra el estado
       conocido en S3 y sube (PUT, con versionado S3) los documentos
       nuevos/actualizados a raw/<source>/. Escribe un resumen de la corrida
       en sync-runs/latest_summary.json.
    2. UpdateDocumentCatalog (Lambda)
       Lee ese resumen y actualiza la tabla DynamoDB
       `finance_document_catalog` con el estado de cada documento conocido
       (nuevo, actualizado o sin cambios), incluyendo su S3 key y version.
    3. Choice: HasChanges?
       - Si no hubo documentos nuevos/actualizados: termina (Success).
       - Si hubo cambios: TriggerDataPrep (ECS Fargate) descarga raw/,
         extrae texto, genera chunks y el dataset de instruccion/respuesta
         (ver data_pipeline/processing/run_data_prep.py), y sube los
         resultados a processed/.

Los dos pasos de computo pesado (sincronizacion y preparacion de datos)
corren como tareas de Amazon ECS Fargate (no Lambda) porque:
  - El scraping/descarga de cientos de PDFs y la generacion de texto/chunks
    puede exceder comodamente el limite de 15 minutos de Lambda.
  - Las dependencias (pypdf, beautifulsoup4, boto3 con reintentos largos)
    son mas simples de empaquetar como imagen de contenedor que como layer
    de Lambda.
"""
import os

from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    aws_dynamodb as dynamodb,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecr_assets as ecr_assets,
    aws_events as events,
    aws_events_targets as events_targets,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_logs as logs,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as sfn_tasks,
    CfnOutput,
)
from constructs import Construct

from stacks.data_pipeline_stack import DataPipelineStack

DATA_PIPELINE_CODE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data_pipeline")

# Fuentes documentales sincronizadas en cada corrida. Debe coincidir con las
# fuentes soportadas por data_pipeline/sync/sync_documents.py.
SOURCES = "cnbv,banxico"

SUMMARY_S3_KEY = "sync-runs/latest_summary.json"

CATALOG_TABLE_NAME = "finance_document_catalog"


class DocumentSyncStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        data_pipeline_stack: DataPipelineStack,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        data_bucket = data_pipeline_stack.data_bucket

        # ------------------------------------------------------------------
        # Catalogo de documentos (DynamoDB)
        # ------------------------------------------------------------------
        self.catalog_table = dynamodb.Table(
            self,
            "FinanceDocumentCatalogTable",
            table_name=CATALOG_TABLE_NAME,
            partition_key=dynamodb.Attribute(
                name="doc_id", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True
            ),
            removal_policy=RemovalPolicy.RETAIN,
        )

        # ------------------------------------------------------------------
        # Red y cluster de ECS Fargate
        # ------------------------------------------------------------------
        vpc = ec2.Vpc(
            self,
            "DocumentSyncVpc",
            max_azs=2,
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                )
            ],
        )

        cluster = ecs.Cluster(self, "DocumentSyncCluster", vpc=vpc)

        task_security_group = ec2.SecurityGroup(
            self,
            "DocumentSyncTaskSecurityGroup",
            vpc=vpc,
            description="Egress-only SG para las tareas Fargate de sincronizacion/preparacion de datos",
            allow_all_outbound=True,
        )

        log_group = logs.LogGroup(
            self,
            "DocumentSyncLogGroup",
            log_group_name="/aws/ecs/document-sync",
            retention=logs.RetentionDays.THREE_MONTHS,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # ------------------------------------------------------------------
        # Imagenes de contenedor (build local via Docker, publicadas a ECR
        # administrado por CDK)
        # ------------------------------------------------------------------
        sync_image_asset = ecr_assets.DockerImageAsset(
            self,
            "SyncDocumentsImage",
            directory=DATA_PIPELINE_CODE_DIR,
            file="sync/Dockerfile",
        )

        data_prep_image_asset = ecr_assets.DockerImageAsset(
            self,
            "DataPrepImage",
            directory=DATA_PIPELINE_CODE_DIR,
            file="processing/Dockerfile",
        )

        # ------------------------------------------------------------------
        # Task definitions
        # ------------------------------------------------------------------
        sync_task_role = iam.Role(
            self,
            "SyncDocumentsTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description="Rol de la tarea Fargate que sincroniza documentos CNBV/Banxico hacia raw/",
            managed_policies=[data_pipeline_stack.pipeline_access_policy],
        )

        sync_task_def = ecs.FargateTaskDefinition(
            self,
            "SyncDocumentsTaskDef",
            cpu=1024,
            memory_limit_mib=2048,
            task_role=sync_task_role,
        )
        sync_container = sync_task_def.add_container(
            "SyncDocumentsContainer",
            image=ecs.ContainerImage.from_docker_image_asset(sync_image_asset),
            logging=ecs.LogDrivers.aws_logs(stream_prefix="sync-documents", log_group=log_group),
            environment={
                "DATA_BUCKET": data_bucket.bucket_name,
                "SOURCES": SOURCES,
                "SUMMARY_S3_KEY": SUMMARY_S3_KEY,
            },
        )

        data_prep_task_role = iam.Role(
            self,
            "DataPrepTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description="Rol de la tarea Fargate que prepara los datos (texto/chunks/dataset) para el fine-tuning",
            managed_policies=[data_pipeline_stack.pipeline_access_policy],
        )
        data_prep_task_role.add_to_policy(
            iam.PolicyStatement(
                sid="InvokeBedrockForDatasetGeneration",
                actions=["bedrock:InvokeModel"],
                resources=["*"],
            )
        )

        data_prep_task_def = ecs.FargateTaskDefinition(
            self,
            "DataPrepTaskDef",
            cpu=2048,
            memory_limit_mib=4096,
            task_role=data_prep_task_role,
        )
        data_prep_container = data_prep_task_def.add_container(
            "DataPrepContainer",
            image=ecs.ContainerImage.from_docker_image_asset(data_prep_image_asset),
            logging=ecs.LogDrivers.aws_logs(stream_prefix="data-prep", log_group=log_group),
            environment={
                "DATA_BUCKET": data_bucket.bucket_name,
                "SOURCES": SOURCES,
                "BEDROCK_REGION": self.region,
            },
        )

        # ------------------------------------------------------------------
        # Lambda: actualizacion del catalogo DynamoDB
        # ------------------------------------------------------------------
        update_catalog_fn = lambda_.Function(
            self,
            "UpdateDocumentCatalogFunction",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="update_catalog.handler",
            code=lambda_.Code.from_asset(
                os.path.join(DATA_PIPELINE_CODE_DIR, "catalog_lambda")
            ),
            timeout=Duration.minutes(2),
            memory_size=256,
            environment={"CATALOG_TABLE_NAME": self.catalog_table.table_name},
        )
        self.catalog_table.grant_write_data(update_catalog_fn)
        data_bucket.grant_read(update_catalog_fn, "sync-runs/*")

        # ------------------------------------------------------------------
        # Step Function
        # ------------------------------------------------------------------
        sync_documents_task = sfn_tasks.EcsRunTask(
            self,
            "SyncDocuments",
            cluster=cluster,
            task_definition=sync_task_def,
            launch_target=sfn_tasks.EcsFargateLaunchTarget(
                platform_version=ecs.FargatePlatformVersion.LATEST
            ),
            assign_public_ip=True,
            security_groups=[task_security_group],
            subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            container_overrides=[
                sfn_tasks.ContainerOverride(
                    container_definition=sync_container,
                )
            ],
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
            result_path="$.syncResult",
            comment="Sincroniza documentos CNBV/Banxico contra sus fuentes publicas y los sube a raw/",
        )

        update_catalog_task = sfn_tasks.LambdaInvoke(
            self,
            "UpdateDocumentCatalog",
            lambda_function=update_catalog_fn,
            payload=sfn.TaskInput.from_object(
                {
                    "bucket": data_bucket.bucket_name,
                    "summary_s3_key": SUMMARY_S3_KEY,
                }
            ),
            payload_response_only=True,
            result_path="$.catalogResult",
            comment=f"Actualiza la tabla DynamoDB {CATALOG_TABLE_NAME} con el estado de cada documento",
        )

        trigger_data_prep_task = sfn_tasks.EcsRunTask(
            self,
            "TriggerDataPrep",
            cluster=cluster,
            task_definition=data_prep_task_def,
            launch_target=sfn_tasks.EcsFargateLaunchTarget(
                platform_version=ecs.FargatePlatformVersion.LATEST
            ),
            assign_public_ip=True,
            security_groups=[task_security_group],
            subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            container_overrides=[
                sfn_tasks.ContainerOverride(
                    container_definition=data_prep_container,
                )
            ],
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
            result_path="$.dataPrepResult",
            comment="Prepara raw/ nuevo/actualizado para el fine-tuning: texto, chunks y dataset",
        )

        no_changes = sfn.Pass(
            self,
            "NoChangesDetected",
            comment="No se encontraron documentos nuevos ni actualizados en esta corrida",
        )

        has_changes_choice = (
            sfn.Choice(self, "HasChanges?")
            .when(
                sfn.Condition.boolean_equals("$.catalogResult.any_changes", True),
                trigger_data_prep_task,
            )
            .otherwise(no_changes)
        )

        definition = sync_documents_task.next(update_catalog_task).next(has_changes_choice)

        self.state_machine = sfn.StateMachine(
            self,
            "DocumentSyncStateMachine",
            state_machine_name="finance-document-sync",
            definition_body=sfn.DefinitionBody.from_chainable(definition),
            timeout=Duration.hours(2),
            logs=sfn.LogOptions(
                destination=logs.LogGroup(
                    self,
                    "DocumentSyncStateMachineLogGroup",
                    log_group_name="/aws/vendedlogs/states/document-sync",
                    retention=logs.RetentionDays.THREE_MONTHS,
                    removal_policy=RemovalPolicy.DESTROY,
                ),
                level=sfn.LogLevel.ALL,
            ),
        )

        # ------------------------------------------------------------------
        # Programacion: cada domingo 02:00 hora de Ciudad de Mexico (UTC-6)
        # EventBridge solo acepta expresiones cron en UTC, por lo que
        # 02:00 UTC-6 equivale a 08:00 UTC.
        # ------------------------------------------------------------------
        self.schedule_rule = events.Rule(
            self,
            "WeeklyDocumentSyncSchedule",
            schedule=events.Schedule.cron(
                minute="0", hour="8", week_day="SUN", month="*", year="*"
            ),
            description="Dispara la sincronizacion documental CNBV/Banxico cada domingo 02:00 (UTC-6)",
            targets=[events_targets.SfnStateMachine(self.state_machine)],
        )

        CfnOutput(
            self,
            "DocumentSyncStateMachineArn",
            value=self.state_machine.state_machine_arn,
            description="ARN del Step Function de sincronizacion documental",
        )
        CfnOutput(
            self,
            "FinanceDocumentCatalogTableName",
            value=self.catalog_table.table_name,
            description="Tabla DynamoDB con el catalogo de documentos regulatorios",
        )
