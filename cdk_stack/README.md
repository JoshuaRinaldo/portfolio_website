# CDK Stack

This CDK stack provisions resources and deploys a static website with serverless backend infrastructure. The [`static_site_stack.py`](static_site_stack.py) file deploys the main static site and uses classes defined in [`sagemaker.py`](sagemaker.py) and [`lambda_.py`](lambda_.py) to deploy SageMaker endpoints and Lambda functions that have been defined in [`cdk.json`](../cdk.json).

## Table of Contents

 - [The Main Stack](#the-main-stack)
 - [Using cdk.json](#using-cdkjson)
 - [sagemaker.py](#sagemakerpy)
 - [lambda_.py](#lambda_py)


### The Main Stack

[`static_site_stack.py`](static_site_stack.py) is the main file from which all resources are provisioned. It reads the context from [`cdk.json`](../cdk.json) to determine the deployment configuration. The file will create the following resources:

1. Lambda functions (based on classes in [`lambda_.py`](lambda_.py)).

2. SageMaker endpoints (based on classes in [`sagemaker.py`](sagemaker.py)).

3. An S3 bucket for hosting static website files.

4. A CloudFront distribution for content delivery.

5. An API Gateway REST API for Lambda function invocation.

6. A CNAME Record and Certificate, using the hosted zone and domain name provided in [`cdk.json`](../cdk.json).

7. Route 53 DNS configuration for the custom domain.

### Using [`cdk.json`](../cdk.json)

Each of the arguments supported by [`cdk.json`](../cdk.json) are described below:

 - `account` `(str)`: The AWS account number of the AWS account to deploy resources in.

 - `region` `(str)`: The region of the CDK stack.

 - `sagemaker_endpoints` `(List[Dict])`: A list of dictionaries containing arguments for a SageMaker endpoint class. Each dictionary in the list represents an individual SageMaker endpoint. Each dictionary must contain a key "endpoint_type" that maps to a class in sagemaker.py, and an environment_variable_name, which will be used to pass the endpoint name to the frontend as a configuration variable. The remaining keys in the endpoint's dictionary are dependent on the endpoint class. See the example below:
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
   See the sagemaker.py file for supported endpoint classes and a breakdown of their arguments.

 - `lambda_functions` `(List[Dict])`: A list of dictionaries containing arguments for a Lambda function class. Each dictionary in the list represents an individual Lambda function. Each dictionary must contain a key "function_type" that maps to a class in lambda_.py, and an environment_variable_name which will be used to reference the Lambda function. Optionally, include a "route_path" to automatically create an API Gateway route for the Lambda function. The remaining keys in the Lambda function's dictionary are dependent on the function's class. See the example below:
   ```
   {
      "function_type": "from_docker_image",
      "environment_variable_name": "INVOKE_MODEL_LAMBDA",
      "folder_name": "invoke_model",
      "route_path": "invoke",
      "policy_statements": [
            {
            "resources": ["*"],
            "actions": ["sagemaker:InvokeEndpoint"]
            }
      ]
   }
   ```
   See the lambda_.py file for supported function classes and a breakdown of their arguments.

 - `classification_models` `(Dict)`: A dictionary mapping model types to their desired and undesired labels. This configuration is used by the frontend to properly format and display model predictions.

 - `environment` `(str)`: The environment of the deployment. This allows for separate testing and production environments. If the deployment is not in the "prod" environment, the environment name is included in the deployment's domain name.

 - `hosted_zone_id` `(str)`: The hosted zone id of the website's domain name.

 - `domain_name` `(str)`: The domain name of the website.


## [`sagemaker.py`](sagemaker.py)

[`sagemaker.py`](sagemaker.py) provides options for deploying custom and off-the-shelf SageMaker models. It currently supports the following SageMaker model types:

#### `SagemakerHuggingface`
- `SagemakerHuggingface` allows users to deploy Hugging Face models and supports the deployment of off-the-shelf models (by specifying the name of the model) or fine-tuned models (by specifying the model data location).

#### `SagemakerFromImageAndModelData`
- `SagemakerFromImageAndModelData` deploys an endpoint that is based on a docker image and model data files. The docker image can either be built during deployment (by specifying a folder with a Dockerfile in it) or pulled from an Elastic Container Registry (ECR) repository (by specifying the repo name and image tag).

## [`lambda_.py`](lambda_.py)

[`lambda_.py`](lambda_.py) provides options for deploying Lambda functions. It currently supports the following function types:

#### `LambdaFunctionFromDockerImage`
- `LambdaFunctionFromDockerImage` deploys a Lambda function that is based on a docker image. The docker image can either be built during deployment (by specifying the folder location of the Lambda function) or pulled from an ECR repository (by specifying the ECR repo name and image tag).
