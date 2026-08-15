import aws_cdk as core
import aws_cdk.assertions as assertions

from stacks.data_pipeline_stack import DataPipelineStack
from stacks.training_stack import TrainingStack
from stacks.agent_runtime_stack import AgentRuntimeStack
from stacks.document_sync_stack import DocumentSyncStack


def _env():
    return core.Environment(account="123456789012", region="us-west-2")


def test_data_pipeline_stack_creates_bucket():
    app = core.App()
    stack = DataPipelineStack(app, "TestDataPipelineStack", env=_env())
    template = assertions.Template.from_stack(stack)

    template.has_resource_properties(
        "AWS::S3::Bucket",
        {
            "BucketEncryption": {
                "ServerSideEncryptionConfiguration": assertions.Match.any_value()
            },
        },
    )
    template.resource_count_is("AWS::S3::Bucket", 1)


def test_training_stack_creates_execution_role():
    app = core.App()
    data_stack = DataPipelineStack(app, "TestDataPipelineStack2", env=_env())
    training_stack = TrainingStack(
        app, "TestTrainingStack", data_pipeline_stack=data_stack, env=_env()
    )
    template = assertions.Template.from_stack(training_stack)

    template.has_resource_properties(
        "AWS::IAM::Role",
        {
            "AssumeRolePolicyDocument": {
                "Statement": [
                    assertions.Match.object_like(
                        {
                            "Principal": {"Service": "sagemaker.amazonaws.com"},
                        }
                    )
                ]
            }
        },
    )


def test_agent_runtime_stack_creates_runtime():
    app = core.App()
    data_stack = DataPipelineStack(app, "TestDataPipelineStack3", env=_env())
    agent_stack = AgentRuntimeStack(
        app, "TestAgentRuntimeStack", data_pipeline_stack=data_stack, env=_env()
    )
    template = assertions.Template.from_stack(agent_stack)

    template.resource_count_is("AWS::BedrockAgentCore::Runtime", 1)


def test_document_sync_stack_creates_expected_resources():
    app = core.App()
    data_stack = DataPipelineStack(app, "TestDataPipelineStack4", env=_env())
    sync_stack = DocumentSyncStack(
        app, "TestDocumentSyncStack", data_pipeline_stack=data_stack, env=_env()
    )
    template = assertions.Template.from_stack(sync_stack)

    # Catalogo DynamoDB
    template.has_resource_properties(
        "AWS::DynamoDB::Table",
        {
            "TableName": "finance_document_catalog",
            "KeySchema": [{"AttributeName": "doc_id", "KeyType": "HASH"}],
        },
    )

    # Step Function orquestador
    template.resource_count_is("AWS::StepFunctions::StateMachine", 1)
    template.has_resource_properties(
        "AWS::StepFunctions::StateMachine",
        {"StateMachineName": "finance-document-sync"},
    )

    # Regla de EventBridge: domingos 02:00 UTC-6 == 08:00 UTC
    template.has_resource_properties(
        "AWS::Events::Rule",
        {"ScheduleExpression": "cron(0 8 ? * SUN *)"},
    )

    # Lambda de actualizacion de catalogo
    template.has_resource_properties(
        "AWS::Lambda::Function",
        {"Handler": "update_catalog.handler"},
    )

    # Dos tareas de ECS Fargate (sync + data prep)
    template.resource_count_is("AWS::ECS::TaskDefinition", 2)
