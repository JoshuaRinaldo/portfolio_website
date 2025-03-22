from aws_cdk import aws_iam as iam
from aws_cdk import aws_sagemaker as sagemaker
from constructs import Construct
import uuid
from typing import Any

# The region dict allows us to use the AWS region name to find the
# correct docker repo
region_dict = {
    "af-south-1": "626614931356",
    "ap-east-1": "871362719292",
    "ap-northeast-1": "763104351884",
    "ap-northeast-2": "763104351884",
    "ap-northeast-3": "364406365360",
    "ap-south-1": "763104351884",
    "ap-southeast-1": "763104351884",
    "ap-southeast-2": "763104351884",
    "ca-central-1": "763104351884",
    "cn-north-1": "727897471807",
    "cn-northwest-1": "727897471807",
    "eu-central-1": "763104351884",
    "eu-north-1": "763104351884",
    "eu-south-1": "692866216735",
    "eu-west-1": "763104351884",
    "eu-west-2": "763104351884",
    "eu-west-3": "763104351884",
    "me-south-1": "217643126080",
    "sa-east-1": "763104351884",
    "us-east-1": "763104351884",
    "us-east-2": "763104351884",
    "us-gov-west-1": "442386744353",
    "us-iso-east-1": "886529160074",
    "us-west-1": "763104351884",
    "us-west-2": "763104351884",
}


def get_image_uri(
        region,
        transformers_version,
        pytorch_version,
        ubuntu_version,
        use_gpu,
    ):
    """
    Formats a string to correspond to a huggingface-provided docker
    image.
    """
    repository = f"{region_dict[region]}.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-inference"
    tag = f"{pytorch_version}-transformers{transformers_version}-{'gpu-py36-cu111' if use_gpu == True else 'cpu-py39'}-ubuntu{ubuntu_version}"
    return f"{repository}:{tag}"


class SagemakerHuggingface(Construct):
    """
    Deploys an off-the-shelf huggingface model from the huggingface hub
    or a finetuned model from a model data file.

    Args:
        region (str): The region of the cdk stack.

        construct_id (str): The model's construct id.

        endpoint_name (str): The name that the endpoint will assume
        once deployed.

        model_task (str): The task of the model.

        undesired_class (str): The name of the undesired class. This
        tells the function which tokens to mask.

        serverless_config (dict): The config for serverless endpoints,
        which is only used if the endpoint is serverles.

        production_variants (dict): The production variants of the
        endpoint.

        model_data_url (str): The location of a huggingface model data
        file. This is only used if the model is a custom finetuned
        model.

        model_name (str): The name of the model. The model name is used
        to pull a huggingface model off the shelf.

        use_gpu (bool): Whether to pull a gpu-compatible docker image
        for the model container.

        transformers_version (str): The transformers version to use
        when pulling the docker image.

        pytorch_version (str): The pytorch version to use when pulling
        the docker image.

        ubuntu_version (str): The ubuntu version to use when pulling
        the docker image.

    Returns:
        None

    """
    def __init__(
            self,
            scope: Construct,
            region: str,
            construct_id: str,
            endpoint_name: str,
            model_task: str = None,
            serverless_config: dict = None,
            production_variants: dict[str, Any] = {},
            model_data_url: str = None,
            model_name: str = None,
            use_gpu: bool = False,
            transformers_version: str = "4.26",
            pytorch_version: str = "1.13",
            ubuntu_version: str = "20.04",
            ) -> None:
        
        super().__init__(scope, construct_id)
        endpoint_name = f"{endpoint_name}-{str(uuid.uuid4())}"[:62]

        if model_name is None and model_data_url is None:
            raise ValueError(
                "One of model_name or model_data_url must be defined when"
                " creating huggingface endpoints."
            )
        
        # Determine image uri and create container definition
        image_uri = get_image_uri(
            region,
            transformers_version,
            pytorch_version,
            ubuntu_version,
            use_gpu,
        )

        container_environment = {}
        if model_task:
            container_environment["HF_TASK"] = model_task
        if model_name:
            container_environment["HF_MODEL_ID"] = model_name
        container = sagemaker.CfnModel.ContainerDefinitionProperty(
             environment=container_environment,
             image=image_uri,
             model_data_url=model_data_url,
             )

        # Set endpoint role/permissions
        role = iam.Role(
            self, f"{construct_id}-sm-role", assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com")
        )
        role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSagemakerFullAccess")
            )
        if model_data_url:
            role.add_to_policy(
                iam.PolicyStatement(
                    resources = [
                        f"arn:aws:s3:::{model_data_url[5:]}"
                    ],
                    actions=["s3:GetObject", "s3:ListBucket"]
                )
            )
        
        model = sagemaker.CfnModel(
            self,
            f"model-{endpoint_name}",
            execution_role_arn=role.role_arn,
            primary_container=container,
            model_name=f'model-{endpoint_name}',
        )
        endpoint_configuration = sagemaker.CfnEndpointConfig(
            self,
            f"endpoint-config-{endpoint_name}",
            endpoint_config_name=f'config-{endpoint_name}',
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    model_name=model.model_name,
                    variant_name=str(uuid.uuid4()),
                    **production_variants,
                    serverless_config=sagemaker.CfnEndpointConfig.ServerlessConfigProperty(**serverless_config) if serverless_config else None,
                )
            ],
        )
        self.endpoint_id = f"endpoint-{endpoint_name}"
        self.endpoint = sagemaker.CfnEndpoint(
            self,
            self.endpoint_id,
            endpoint_name=endpoint_name,
            endpoint_config_name=endpoint_configuration.endpoint_config_name,
        )

        # Add dependencies so the endpoint isn't created before the
        # model or endpoint config
        model.node.add_dependency(role)
        endpoint_configuration.node.add_dependency(model)
        self.endpoint.node.add_dependency(endpoint_configuration)
    
    def return_name(self):
        """
        Returns the endpoint name and arn.
        """
        return self.endpoint.endpoint_name

class SagemakerFromImageAndModelData(Construct):
    """
    Deploys a sagemaker model based on a custom docker image and a
    model data file.

    Args:
        region (str): The region of the cdk stack.

        account (str): The account number where the ECR repo is located.

        construct_id (str): The construct id.

        endpoint_name (str): The name that the endpoint will assume
        once deployed.

        image_repo_name (str): The name of the ECR repo from which the
        image will be pulled.

        image_tag (str): The tag of the desired image.

        model_data_bucket (str): The S3 bucket where the model data
        file is located.

        serverless_config (dict): The config for serverless endpoints,
        which is only used if the endpoint is serverles.

        production_variants (dict): The production variants of the
        endpoint.

        container_environment (dict): The environment variables to
        start the container with.

    Returns:
        None

    """
    def __init__(
            self,
            scope: Construct,
            region: str,
            account: str,
            construct_id: str,
            endpoint_name: str,
            image_repo_name: str,
            image_tag: str,
            model_data_bucket: str = None,
            serverless_config: dict = None,
            production_variants: dict[str, Any] = {},
            container_environment: dict[str, Any] = {},
            ) -> None:
        
        super().__init__(scope, construct_id)
        endpoint_name = f"{endpoint_name}-{str(uuid.uuid4())}"[:50]

        # Set endpoint role/permissions, define Sagemaker Model
        role = iam.Role(
            self, f"{construct_id}-sm-role", assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com")
        )
        role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSagemakerFullAccess")
            )
        if model_data_bucket:
            role.add_to_policy(
                iam.PolicyStatement(
                    resources = [
                        f"arn:aws:s3:::{model_data_bucket}/*"
                    ],
                    actions=["s3:GetObject", "s3:ListBucket"]
                )
            )
        
        model = sagemaker.CfnModel(
            self,
            f"model-{endpoint_name}",
            execution_role_arn=role.role_arn,
            model_name=f'model-{endpoint_name}',
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                environment=container_environment,
                image=f"{account}.dkr.ecr.{region}.amazonaws.com/{image_repo_name}:{image_tag}",
            ),
        )

        # Creates SageMaker Endpoint configurations
        endpoint_configuration = sagemaker.CfnEndpointConfig(
            self,
            f"endpoint-config-{endpoint_name}",
            endpoint_config_name=f'config-{endpoint_name}',
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    model_name=model.model_name,
                    variant_name=model.model_name,
                    **production_variants,
                    serverless_config=sagemaker.CfnEndpointConfig.ServerlessConfigProperty(**serverless_config) if serverless_config else None,
                )
            ],
        )

        # Creates Real-Time Endpoint
        self.endpoint_id = f"endpoint-{endpoint_name}"
        self.endpoint = sagemaker.CfnEndpoint(
            self,
            self.endpoint_id,
            endpoint_name=endpoint_name,
            endpoint_config_name=endpoint_configuration.endpoint_config_name,
        )

        # adds depends on for different resources
        model.node.add_dependency(role)
        endpoint_configuration.node.add_dependency(model)
        self.endpoint.node.add_dependency(endpoint_configuration)
    
    def return_name(self):
        """
        Returns the endpoint name and arn.
        """
        return self.endpoint.endpoint_name
