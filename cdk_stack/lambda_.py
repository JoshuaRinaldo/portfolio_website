from aws_cdk import (
    aws_ecr_assets as ecrassets,
    aws_iam as iam,
    aws_lambda as _lambda,
    Duration,
)
from constructs import Construct
from typing import List

class LambdaFunctionFromDockerImage(Construct):
    def __init__(
            self,
            scope: Construct,
            construct_id: str,
            lambda_folder: str=None,
            platform: str = None,
            ecr_repo: str = None,
            tag: str = None,
            timeout: int = 5,
            policy_statements: List[dict[str, List]] = {},
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

        self.counterfactual_lambda = _lambda.DockerImageFunction(
            scope=self,
            id="ExampleDockerLambda",
            role=lambda_role,
            timeout=Duration.minutes(timeout),
            code=lambda_docker_image,
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
        return self.counterfactual_lambda.function_name, self.counterfactual_lambda.function_arn
