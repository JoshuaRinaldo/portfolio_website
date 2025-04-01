from aws_cdk import (
    aws_ecr_assets as ecrassets,
    aws_ecs as ecs,
    aws_iam as iam,
    aws_lambda as _lambda,
    Duration,
)
from constructs import Construct
from typing import List

class LambdaFunctionFromDockerImage(Construct):
    """
    Deploys a lambda function from a docker image.

    Args:
        construct_id (str): The function's construct id.

        lambda_folder (str): The folder name of the lambda function.
        The lambda folder must be in the lambda_functions folder. This
        argument is only used if the docker image is being built during
        deployment.

        platform (str): The CPU platform/architecture to use.

        ecr_repo (str): The name of the ECR repo containing the docker
        image. This argument is only used if the docker image is being
        pulled from an ECR repository.

        tag (str): The tag of the docker image in the ECR repo. Must be
        a valid tag in the ECR repo passed in the ecr_repo argument.

        timeout (int): The time (in minutes) before the lambda function
        times out.

        memory_size (int): The memory size in MB.
        
        policy_statements (List[Dict[str, List[str]]]): A list of
        extra policy statements to add to the lambda function. By
        default, the function has the basic Lambda role. If additional
        permissions are required, they are added through this argument.
        Structure additional policy statements as follows:

        ```
        {
            "resources": ["the resources to include in the policy"],
            "actions": ["the actions to include in the policy"]
        }
        ```

    Returns:
        None

    """
    def __init__(
            self,
            scope: Construct,
            construct_id: str,
            lambda_folder: str=None,
            platform: str = None,
            ecr_repo: str = None,
            tag: str = None,
            timeout: int = 5,
            memory_size: int = 512,
            policy_statements: List[dict[str, List[str]]] = [],
            ) -> None:
        
        super().__init__(scope, construct_id)

        lambda_role = iam.Role(self, "My Role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com")
        )

        if lambda_folder:
            lambda_docker_image = _lambda.DockerImageCode.from_image_asset(

                # Directory relative to where you execute cdk deploy
                # contains a Dockerfile with build instructions
                directory=f"lambda_functions/{lambda_folder}/.",
                platform=(
                    ecrassets.Platform.LINUX_ARM64 if platform == "arm64"
                    else ecrassets.Platform.LINUX_AMD64
                )
            )
        elif ecr_repo and tag:
            lambda_docker_image = _lambda.DockerImageCode.from_ecr(
                repository=ecr_repo,
                tag=tag,
            )
        else:
            raise ValueError(
                "Either lambda_folder or ecr_repo AND tag must be"
                " defined."
            )

        self.lambda_function = _lambda.DockerImageFunction(
            scope=self,
            id="ExampleDockerLambda",
            role=lambda_role,
            timeout=Duration.minutes(timeout),
            code=lambda_docker_image,
            memory_size=memory_size,
            architecture=(
            _lambda.Architecture.ARM64 if platform == "arm64"
            else _lambda.Architecture.X86_64
        )
        )
        lambda_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
            )
        
        for policy_n in policy_statements:
            lambda_role.add_to_policy(iam.PolicyStatement(
                resources=policy_n["resources"],
                actions=policy_n["actions"],
            ))
    
    def return_name(self):
        """
        Returns the function name and arn.
        """
        return self.lambda_function.function_name, self.lambda_function.function_arn
