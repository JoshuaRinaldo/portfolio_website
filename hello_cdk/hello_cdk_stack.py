from aws_cdk import (
    aws_iam as iam,
    Stack,
    aws_lambda as _lambda, # Import the Lambda module
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
from .sagemaker import HuggingfaceSagemaker, SagemakerFromImageAndModelData
from constructs import Construct
from pathlib import Path


class HelloCdkStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        region = self.node.try_get_context("region")
        account = self.node.try_get_context("account")
        sagemaker_endpoints = self.node.try_get_context("sagemaker_endpoints")
        environment = self.node.try_get_context("environment")
        hosted_zone_id = self.node.try_get_context("hosted_zone_id")
        domain_name = self.node.try_get_context("domain_name")
        platform = self.node.try_get_context("platform")

        cpu_platform = (
            ecrassets.Platform.LINUX_ARM64 if platform == "arm64"
            else ecrassets.Platform.LINUX_AMD64
        )
        cpu_architecture = (
            ecs.CpuArchitecture.ARM64 if platform == "arm64"
            else ecs.CpuArchitecture.AMD64
        )
        

        # Define the Lambda function resource
        counterfactual_lambda_role = iam.Role(self, "My Role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com")
        )
        counterfactual_lambda = _lambda.Function(self, "MyFunction",
            runtime=_lambda.Runtime.PYTHON_3_13,
            handler="unmask_models.handler",
            code=_lambda.Code.from_asset("lambda_functions/unmask_models/"),
            role=counterfactual_lambda_role,
            timeout=Duration.minutes(5)
        )
        counterfactual_lambda_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
            )
        
        counterfactual_lambda_role.add_to_policy(iam.PolicyStatement(
            resources=["*"],
            actions=["sagemaker:InvokeEndpoint"]
        ))

        counterfactual_lambda_name = counterfactual_lambda.function_name



        # Create SageMaker endpoints
        for endpoint_n in sagemaker_endpoints:
            if endpoint_n["endpoint_type"] == "image_and_model_data":
                SagemakerFromImageAndModelData(
                    scope=self,
                    region = region,
                    account = account,
                    construct_id = endpoint_n["endpoint_name"],
                    endpoint_name = endpoint_n["endpoint_name"],
                    image_repo_name = endpoint_n["image_repo_name"],
                    image_tag = endpoint_n["image_tag"],
                    model_data_bucket = endpoint_n["model_data_bucket"],
                    serverless_config = endpoint_n.get("serverless_config", {}),
                    container_environment=endpoint_n.get("container_environment", {})
                )

            elif endpoint_n["endpoint_type"] == "huggingface":
                HuggingfaceSagemaker(
                    scope = self,
                    region = region,
                    construct_id = endpoint_n["endpoint_name"],
                    endpoint_name = endpoint_n["endpoint_name"],
                    model_task=endpoint_n.get("model_task"),
                    serverless_config = endpoint_n.get("serverless_config", {}),
                    production_variants = endpoint_n.get("production_variants", {}),
                    model_data_url = endpoint_n.get("model_data_url"),
                    model_name = endpoint_n.get("model_name"),
                    use_gpu = endpoint_n.get("use_gpu", False),
                )
            
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
                resources=["*"]
                ),


                iam.PolicyStatement(
                actions=[
                "lambda:InvokeFunction"
                ],
                resources=["*"]
                ),

            ]
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
                environment={
                    "COUNTERFACTUAL_LAMBDA": counterfactual_lambda_name,
                    "SENTIMENT_UNMASKING_MODEL": "sentiment-mlm",
                    "SENTIMENT_CLASSIFICATION_MODEL": "sentiment-classifier",
                }
            ),
        )

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

        # Configure health check using streamlit's default health check
        # endpoint
        alb_ecs_service.target_group.configure_health_check(
            path="/healthz",
            interval=Duration.seconds(60),
            unhealthy_threshold_count=5,
            )

        # Attach iam policy to alb/ecs cluster
        alb_ecs_service.task_definition.task_role.attach_inline_policy(
            ecs_task_role_iam_policy
        )

        CfnOutput(
            self,
            "api_url",
            value=f"https://{domain_name}",
        )