from aws_cdk import (
    aws_iam as iam,
    Stack,
    CfnOutput,
    Duration,
    RemovalPolicy,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_iam as iam,
    aws_ecr_assets as ecrassets,
    aws_elasticloadbalancingv2 as elbv2,
    aws_ecs_patterns as ecspatterns,
    aws_route53 as route53,
    aws_certificatemanager as certificatemanager
)
from .sagemaker import SagemakerHuggingface, SagemakerFromImageAndModelData
from .lambda_ import LambdaFunctionFromDockerImage
from constructs import Construct
from pathlib import Path


class StreamlitSite(Stack):
    """
    The StreamlitSite class provisions a collection of resources that
    can be used to serve interactive machine learning models using a
    streamlit app.

    Args:

        account (str): The AWS account number of the AWS aacount to
        deploy resources in.

        region (str): The region of the cdk stack.

        sagemaker_endpoints (List[Dict]): A list of dictionaries
        containing arguments for a SageMaker endpoint class. Each
        dictionary in the list represents an individual SageMaker 
        endpoint. Each dictionary must contain a key "endpoint_type"
        that maps to a class in sagemaker.py, and an
        environment_variable_name, which will be used to pass the
        endpoint name to the streamlit app as an environment variable.
        The remaining keys in the endpoint's dictionary are dependent
        on the endpoint class. See the example below:
        ```
        {
            "endpoint_type": "huggingface",
            "environment_variable_name": "SENTIMENT_UNMASKING_MODEL",
            "serverless_config": {
                "memory_size_in_mb": 2048,
                "max_concurrency": 1
            },
            "model_data_url": "an s3 uri"
        }
        ```
        See the sagemaker.py file for supported endpoint classes and
        a breakdown of their arguments.

        lambda_functions (List[Dict]): A list of dictionaries
        containing arguments for a Lambda function class. Each
        dictionary in the list represents an individual Lambda
        function. Each dictionary must contain a key "function_type"
        that maps to a class in lambda_.py, and an
        environment_variable_name, which will be used to pass the
        lambda function name to the streamlit app as an environment
        variable. The remaining keys in the Lambda function's
        dictionary are dependent on the function's class. See the
        example below:
        ```
        {
            "function_type": "from_docker_image",
            "environment_variable_name": "COUNTERFACTUAL_LAMBDA",
            "folder_name": "text_counterfactuals",
            "policy_statements": [
                {
                "resources": ["*"],
                "actions": ["sagemaker:InvokeEndpoint"]
                }
            ]
        }
        ```
        See the lambda_.py file for supported function classes and a
        breakdown of their arguments.

        environment (str): The environment of the deployment. This
        allows for separate testing and production environments. If the
        deployment is not in the "prod" environment, the environment
        name is included in the deployment's domain name.

        hosted_zone_id (str): The hosted zone id of the website's
        domain name.

        domain_name (str): The domain name of the website.

        platform (str): The cpu platform/architecture to run the
        streamlit app on. Can either be "arm64" or "amd64". Defaults to
        "amd64", but please take note of your local machine's
        compatibility when modifying this argument.

        streamlit_environment_variables (Dict[str, Any]): Extra
        environment variables to pass to the streamlit app. By default
        this variable initializes as empty and is filled with the names
        of external resources (lambda functions and sagemaker
        endpoints).

        ecs_policy_statements (List[Dict[str, List[str]]]): A list of
        extra policy statements to add to the streamlit app. By
        default the app is allowed to invoke SageMaker endpoints and
        Lambda functions. If additional permissions are required, they
        are added through this argument. Structure additional policy
        statements as follows:

        ```
        {
            "resources": ["the resources to include in the policy"],
            "actions": ["the actions to include in the policy"]
        }
        ```

    Returns:
        None
    """
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        account = self.node.try_get_context("account")
        region = self.node.try_get_context("region")
        sagemaker_endpoints = self.node.try_get_context("sagemaker_endpoints")
        lambda_functions = self.node.try_get_context("lambda_functions")
        environment = self.node.try_get_context("environment")
        hosted_zone_id = self.node.try_get_context("hosted_zone_id")
        domain_name = self.node.try_get_context("domain_name")
        platform = self.node.try_get_context("platform")
        streamlit_environment_variables = self.node.try_get_context("streamlit_environment_variables")
        ecs_policy_statements = self.node.try_get_context("ecs_policy_statements")


        if not streamlit_environment_variables:
            streamlit_environment_variables = {}

        cpu_platform = (
            ecrassets.Platform.LINUX_ARM64 if platform == "arm64"
            else ecrassets.Platform.LINUX_AMD64
        )
        cpu_architecture = (
            ecs.CpuArchitecture.ARM64 if platform == "arm64"
            else ecs.CpuArchitecture.X86_64
        )

        # Create Lambda functions
        lambda_arns = []
        for lambda_function_n in lambda_functions:
            lambda_env_var_name = lambda_function_n["environment_variable_name"]
            if lambda_function_n["function_type"] == "from_docker_image":
                lambda_function = LambdaFunctionFromDockerImage(
                    scope = self,
                    construct_id = f"{construct_id}-{lambda_env_var_name}",
                    lambda_folder = lambda_function_n.get("folder_name", None),
                    ecr_repo = lambda_function_n.get("ecr_repo", None),
                    tag = lambda_function_n.get("tag", None),
                    platform = lambda_function_n.get("platform"),
                    timeout = lambda_function_n.get("timeout", 5),
                    policy_statements = lambda_function_n.get("policy_statements", {}),
                )

            lambda_name, lambda_arn = lambda_function.return_name()
            streamlit_environment_variables[lambda_env_var_name] = lambda_name
            lambda_arns.append(lambda_arn)


        # Create SageMaker endpoints
        sagemaker_arns = []
        for endpoint_n in sagemaker_endpoints:
            endpoint_env_var_name = endpoint_n.get("environment_variable_name")
            if endpoint_n["endpoint_type"] == "image_and_model_data":
                
                sagemaker_endpoint = SagemakerFromImageAndModelData(
                    scope=self,
                    region = region,
                    account = account,
                    construct_id = f"{construct_id}-{endpoint_env_var_name}",
                    endpoint_name = endpoint_n["endpoint_name"],
                    image_repo_name = endpoint_n["image_repo_name"],
                    image_tag = endpoint_n["image_tag"],
                    model_data_bucket = endpoint_n["model_data_bucket"],
                    serverless_config = endpoint_n.get("serverless_config", {}),
                    container_environment=endpoint_n.get("container_environment", {}),
                )

            elif endpoint_n["endpoint_type"] == "huggingface":
                sagemaker_endpoint = SagemakerHuggingface(
                    scope = self,
                    region = region,
                    construct_id = f"{construct_id}-{endpoint_env_var_name}",
                    endpoint_name = endpoint_n["endpoint_name"],
                    model_task=endpoint_n.get("model_task"),
                    serverless_config = endpoint_n.get("serverless_config", {}),
                    production_variants = endpoint_n.get("production_variants", {}),
                    model_data_url = endpoint_n.get("model_data_url"),
                    model_name = endpoint_n.get("model_name"),
                    use_gpu = endpoint_n.get("use_gpu", False),
                )

            endpoint_name = sagemaker_endpoint.return_name()
            endpoint_arn = f"arn:aws:sagemaker:{region}:{account}:endpoint/{endpoint_name}"
            streamlit_environment_variables[endpoint_env_var_name] = endpoint_name
            sagemaker_arns.append(endpoint_arn)
        
        # If deploying to a test environment, label env in domain name,
        # if prod environment, deploy without env in domain name
        if environment == "prod":
            api_domain_name = domain_name
        else:
            api_domain_name = f"{environment}.{domain_name}"
        
        # VPC for ALB and ECS cluster
        vpc = ec2.Vpc(
            self,
            f"streamlit-app-vpc-{environment}",
            ip_addresses=ec2.IpAddresses.cidr("10.0.0.0/16"),
            max_azs=2,
            vpc_name=f"streamlit-app-vpc-{environment}",
            nat_gateways=1,
        )
        vpc.add_gateway_endpoint(
            f"vpc-s3-endpoint-{environment}",
            service=ec2.GatewayVpcEndpointAwsService.S3
        )
        vpc.add_interface_endpoint("EcrDockerEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER
        )
        vpc.add_interface_endpoint("EcrEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.ECR
        )

        # Configure hosted zone using custom domain name
        hosted_zone = route53.HostedZone.from_hosted_zone_attributes(
            self,
            id=hosted_zone_id,
            hosted_zone_id=hosted_zone_id,
            zone_name=api_domain_name
        )

        route53.CnameRecord(
            self,
            f"{environment}-cname-record",
            record_name=domain_name,
            zone=hosted_zone,
            domain_name=domain_name,
        )

        # Create certificate
        certificate = certificatemanager.Certificate(
            self,
            id=f"{environment}-certificate",
            domain_name=domain_name,
            validation=certificatemanager.CertificateValidation.from_dns(
            hosted_zone=hosted_zone
            ),
            subject_alternative_names=[api_domain_name],
        )
        certificate.apply_removal_policy(RemovalPolicy.DESTROY)

        # Create ecs cluster, iam policy, make cluster dependent on
        # policy
        ecs_cluster = ecs.Cluster(
            self,
            id=f"{environment}-streamlit-app-ecs-cluster",
            vpc=vpc,
        )
        ecs_task_role_iam_policy = iam.Policy(
            self,
            id=f"{environment}-streamlit-app-ecs-iam-policy",
            statements=[
                iam.PolicyStatement(
                actions=[
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage"
                ],
                resources=["arn:aws:ecr:us-east-1:324014232528:repository/*"]
                ),
                iam.PolicyStatement(
                actions=[
                "sagemaker:InvokeEndpoint",
                ],
                resources=sagemaker_arns
                ),
                iam.PolicyStatement(
                actions=[
                "lambda:InvokeFunction"
                ],
                resources=lambda_arns
                ),
            ]
        )

        if ecs_policy_statements:
            for policy_statement_n in ecs_policy_statements:
                ecs_task_role_iam_policy.add_to_policy(
                iam.PolicyStatement(
                    resources = policy_statement_n["resources"],
                    actions=policy_statement_n["actions"]
                )
            )

        ecs_cluster.node.add_dependency(
            ecs_task_role_iam_policy
        )

        # Build Dockerfile from local folder and push to ECR
        docker_image = ecrassets.DockerImageAsset(
            self,
            id=f"{environment}-streamlit-app-docker-image",
            directory=str(Path(__file__).parent.parent / "streamlit_app"),
            file="Dockerfile",
            platform=cpu_platform
        )

        # create alb/ecs cluster
        alb_ecs_service = ecspatterns.ApplicationLoadBalancedFargateService(
            self,
            id=f"{environment}-application-load-balanced-fargate-service",
            certificate=certificate,
            cpu=256,
            memory_limit_mib=2048,
            cluster=ecs_cluster,
            desired_count=1,
            domain_name=api_domain_name,
            domain_zone=hosted_zone,
            public_load_balancer=True,
            open_listener=False,
            load_balancer_name=f"{environment}-streamlit-app-load-balancer",
            ssl_policy=elbv2.SslPolicy.FORWARD_SECRECY_TLS12_RES,
            task_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
            runtime_platform=ecs.RuntimePlatform(
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
                cpu_architecture=cpu_architecture,
            ),
            task_image_options=ecspatterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_docker_image_asset(docker_image),
                container_port=8080,
                environment=streamlit_environment_variables,
            ),
        )

        # Create security group for load balancer
        load_balancer = alb_ecs_service.load_balancer
        security_group = ec2.SecurityGroup(
            self,
            f"streamlit_app_security_group_{environment}",
            vpc=vpc,
            security_group_name=f"streamlit_app_security_group_{environment}",
        )

        # Give access to all incoming traffic
        security_group.add_ingress_rule(
            peer=ec2.Peer.any_ipv4(),
            connection=ec2.Port.tcp(443)
        )
        security_group.add_ingress_rule(
            peer=ec2.Peer.any_ipv6(),
            connection=ec2.Port.tcp(443)
        )
        load_balancer.add_security_group(
            security_group
        )

        # Attach iam policy to alb/ecs cluster
        alb_ecs_service.task_definition.task_role.attach_inline_policy(
            ecs_task_role_iam_policy
        )

        # Configure health check using streamlit's default health check
        # endpoint
        alb_ecs_service.target_group.configure_health_check(
            path="/healthz",
            interval=Duration.seconds(60),
            unhealthy_threshold_count=5,
            )

        CfnOutput(
            self,
            "api_url",
            value=f"https://{api_domain_name}",
        )
