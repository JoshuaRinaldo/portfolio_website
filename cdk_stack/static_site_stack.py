from aws_cdk import (
    Stack,
    CfnOutput,
    Duration,
    RemovalPolicy,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_certificatemanager as certificatemanager,
    aws_route53 as route53,
    aws_route53_targets as targets,
    aws_apigateway as apigw,
)
from .sagemaker import SagemakerHuggingface, SagemakerFromImageAndModelData
from .lambda_ import LambdaFunctionFromDockerImage
from constructs import Construct
from pathlib import Path
import json


class StaticSite(Stack):
    """
    StaticSite provisions a portfolio website using S3, CloudFront,
    and API Gateway. From this template you can serve live demos.

    Args:
        account (str): AWS account number
        region (str): AWS region
        sagemaker_endpoints (List[Dict]): SageMaker endpoint configurations
        lambda_functions (List[Dict]): Lambda function configurations
        environment (str): Environment name
        hosted_zone_id (str): Route53 hosted zone ID
        domain_name (str): Domain name for the website
        classification_models (Dict): Model type to label mappings

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
        classification_models = self.node.try_get_context("classification_models")

        # Store Lambda configs that need endpoint names as env vars
        lambda_arns = []
        lambda_functions_map = {}  # Maps route_path -> lambda function
        warmup_lambda_config = None
        invoke_model_lambda_config = None

        for lambda_function_n in lambda_functions:
            lambda_env_var_name = lambda_function_n["environment_variable_name"]
            route_path = lambda_function_n.get("route_path")

            # Skip warmup and invoke_model Lambdas
            if lambda_env_var_name == "WARMUP_ENDPOINTS_LAMBDA":
                warmup_lambda_config = lambda_function_n
                continue

            if lambda_env_var_name == "INVOKE_MODEL_LAMBDA":
                invoke_model_lambda_config = lambda_function_n
                continue

            if lambda_function_n["function_type"] == "from_docker_image":
                lambda_function = LambdaFunctionFromDockerImage(
                    scope=self,
                    construct_id=f"{construct_id}-{lambda_env_var_name}",
                    ecr_repo=lambda_function_n.get("ecr_repo", None),
                    tag=lambda_function_n.get("tag", None),
                    lambda_folder=lambda_function_n.get("folder_name", None),
                    platform=lambda_function_n.get("platform", "amd64"),
                    timeout=lambda_function_n.get("timeout", 5),
                    policy_statements=lambda_function_n.get("policy_statements", {}),
                    memory_size=512,
                )

            lambda_name, lambda_arn = lambda_function.return_name()
            lambda_arns.append(lambda_arn)

            # If route_path is specified, add to map for API Gateway integration
            if route_path:
                lambda_functions_map[route_path] = lambda_function.lambda_function

        # Create SageMaker endpoints and build endpoint name mapping
        sagemaker_arns = []
        endpoint_names = {}

        for endpoint_n in sagemaker_endpoints:
            endpoint_env_var_name = endpoint_n.get("environment_variable_name")

            if endpoint_n["endpoint_type"] == "image_and_model_data":
                sagemaker_endpoint = SagemakerFromImageAndModelData(
                    scope=self,
                    region=region,
                    account=account,
                    construct_id=f"{construct_id}-{endpoint_env_var_name}",
                    endpoint_name=endpoint_n["endpoint_name"],
                    image_repo_name=endpoint_n.get("image_repo_name"),
                    image_tag=endpoint_n.get("image_repo_name"),
                    dockerfile_folder=endpoint_n.get("dockerfile_folder"),
                    platform=endpoint_n.get("platform"),
                    model_data_bucket=endpoint_n.get("model_data_bucket"),
                    serverless_config=endpoint_n.get("serverless_config", {}),
                    container_environment=endpoint_n.get("container_environment", {}),
                )

            elif endpoint_n["endpoint_type"] == "huggingface":
                sagemaker_endpoint = SagemakerHuggingface(
                    scope=self,
                    region=region,
                    construct_id=f"{construct_id}-{endpoint_env_var_name}",
                    endpoint_name=endpoint_n["endpoint_name"],
                    model_task=endpoint_n.get("model_task"),
                    serverless_config=endpoint_n.get("serverless_config", {}),
                    production_variants=endpoint_n.get("production_variants", {}),
                    model_data_url=endpoint_n.get("model_data_url"),
                    model_name=endpoint_n.get("model_name"),
                    use_gpu=endpoint_n.get("use_gpu", False),
                )

            endpoint_name = sagemaker_endpoint.return_name()
            endpoint_arn = f"arn:aws:sagemaker:{region}:{account}:endpoint/{endpoint_name}"
            sagemaker_arns.append(endpoint_arn)
            endpoint_names[endpoint_env_var_name] = endpoint_name

        # Create warmup lambda
        if warmup_lambda_config:

            warmup_env_vars = {}

            # Add each endpoint name as an environment variable
            for env_var_name, endpoint_name in endpoint_names.items():
                warmup_env_vars[f"{env_var_name}_ENDPOINT_NAME"] = endpoint_name

            warmup_lambda = LambdaFunctionFromDockerImage(
                scope=self,
                construct_id=f"{construct_id}-{warmup_lambda_config['environment_variable_name']}",
                ecr_repo=warmup_lambda_config.get("ecr_repo", None),
                tag=warmup_lambda_config.get("tag", None),
                lambda_folder=warmup_lambda_config.get("folder_name", None),
                platform=warmup_lambda_config.get("platform", "amd64"),
                timeout=warmup_lambda_config.get("timeout", 5),
                policy_statements=warmup_lambda_config.get("policy_statements", []),
                memory_size=512,
                environment=warmup_env_vars,
            )

            # Add warmup Lambda to the routing map
            route_path = warmup_lambda_config.get("route_path")
            if route_path:
                lambda_functions_map[route_path] = warmup_lambda.lambda_function

        # Create invoke_model Lambda with endpoint names as environment variables for validation
        if invoke_model_lambda_config:
            # Build environment variables with endpoint names for allowlist validation
            invoke_model_env_vars = {}
            for env_var_name, endpoint_name in endpoint_names.items():
                invoke_model_env_vars[env_var_name] = endpoint_name

            invoke_model_lambda = LambdaFunctionFromDockerImage(
                scope=self,
                construct_id=f"{construct_id}-{invoke_model_lambda_config['environment_variable_name']}",
                ecr_repo=invoke_model_lambda_config.get("ecr_repo", None),
                tag=invoke_model_lambda_config.get("tag", None),
                lambda_folder=invoke_model_lambda_config.get("folder_name", None),
                platform=invoke_model_lambda_config.get("platform", "amd64"),
                timeout=invoke_model_lambda_config.get("timeout", 5),
                policy_statements=invoke_model_lambda_config.get("policy_statements", []),
                memory_size=512,
                environment=invoke_model_env_vars,
            )

            # Add invoke_model Lambda to the routing map
            route_path = invoke_model_lambda_config.get("route_path")
            if route_path:
                lambda_functions_map[route_path] = invoke_model_lambda.lambda_function

        # Set domain name based on environment
        if environment == "prod":
            api_domain_name = domain_name
        else:
            api_domain_name = f"{environment}.{domain_name}"

        # Create API Gateway with Lambda proxy integration
        api = apigw.RestApi(
            self,
            f"{environment}-api",
            rest_api_name=f"{environment}-api",
            description="API for portfolio website services",
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=[f"https://{api_domain_name}"],  # Restrict to portfolio domain only
                allow_methods=["POST", "OPTIONS"],
                allow_headers=["Content-Type", "Authorization"],
            ),

            # Add default throttling to prevent abuse
            deploy_options=apigw.StageOptions(
                throttling_rate_limit=2,
                throttling_burst_limit=5,
            ),
        )

        # Automatically create API Gateway routes for all Lambdas with route_path
        for route_path, lambda_func in lambda_functions_map.items():
            # Create Lambda integration
            lambda_integration = apigw.LambdaIntegration(
                lambda_func,
                proxy=True, 
                timeout=Duration.seconds(90)
            )

            # Add API resource and method
            resource = api.root.add_resource(route_path)
            resource.add_method("POST", lambda_integration)

        # Configure hosted zone
        hosted_zone = route53.HostedZone.from_hosted_zone_attributes(
            self,
            id=hosted_zone_id,
            hosted_zone_id=hosted_zone_id,
            zone_name=api_domain_name
        )

        # Create certificate for CloudFront (must be in us-east-1)
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

        # Create S3 bucket for static website
        # Note: Bucket is private - CloudFront accesses it via OAI (Origin Access Identity)
        # We don't use S3 website hosting mode - CloudFront serves the files directly
        website_bucket = s3.Bucket(
            self,
            f"{environment}-website-bucket",
            bucket_name=f"{environment}-{domain_name.replace('.', '-')}-website",
            public_read_access=False,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # Create CloudFront Origin Access Identity
        oai = cloudfront.OriginAccessIdentity(
            self,
            f"{environment}-oai",
            comment=f"OAI for {domain_name}"
        )

        # Grant CloudFront access to S3 bucket
        website_bucket.grant_read(oai)

        # Create CloudFront distribution
        distribution = cloudfront.Distribution(
            self,
            f"{environment}-distribution",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3Origin(
                    website_bucket,
                    origin_access_identity=oai
                ),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED,
            ),
            domain_names=[api_domain_name],
            certificate=certificate,
            default_root_object="index.html",
            error_responses=[
                # Handle 404s - return index.html for client-side routing
                cloudfront.ErrorResponse(
                    http_status=404,
                    response_http_status=200,
                    response_page_path="/index.html",
                    ttl=Duration.minutes(5),
                ),
                # Handle 403s (forbidden) - this happens when accessing a "directory"
                # CloudFront returns 403 when OAI tries to list objects
                cloudfront.ErrorResponse(
                    http_status=403,
                    response_http_status=200,
                    response_page_path="/index.html",
                    ttl=Duration.minutes(5),
                ),
            ],
        )

        # Build API endpoint URLs manually using the API ID
        # This avoids CDK token resolution issues
        api_endpoint = f"https://{api.rest_api_id}.execute-api.{region}.amazonaws.com/{environment}/invoke"
        warmup_endpoint = f"https://{api.rest_api_id}.execute-api.{region}.amazonaws.com/{environment}/warmup"

        # Generate config.js content
        config_js_content = f"""// Configuration generated during CDK deployment
const API_ENDPOINT = '{api_endpoint}';
const WARMUP_ENDPOINT = '{warmup_endpoint}';

const CLASSIFICATION_MODELS = {json.dumps(classification_models, indent=2)};

const ENDPOINT_NAMES = {json.dumps(endpoint_names, indent=2)};
"""

        # Deploy website files to S3
        website_dir = Path(__file__).parent.parent / "website"

        s3deploy.BucketDeployment(
            self,
            f"{environment}-website-deployment",
            sources=[
                s3deploy.Source.asset(str(website_dir)),
                s3deploy.Source.data("js/config.js", config_js_content),
            ],
            destination_bucket=website_bucket,
            distribution=distribution,
            distribution_paths=["/*"],
        )

        # Create Route53 record pointing to CloudFront
        route53.ARecord(
            self,
            f"{environment}-alias-record",
            zone=hosted_zone,
            target=route53.RecordTarget.from_alias(
                targets.CloudFrontTarget(distribution)
            ),
            record_name=api_domain_name,
        )

        # Output API endpoint
        CfnOutput(
            self,
            "ApiUrl",
            value=api.url,
            description="API Gateway URL",
        )

        CfnOutput(
            self,
            "WebsiteUrl",
            value=f"https://{api_domain_name}",
            description="Website URL",
        )

        CfnOutput(
            self,
            "CloudFrontUrl",
            value=f"https://{distribution.distribution_domain_name}",
            description="CloudFront distribution URL",
        )
