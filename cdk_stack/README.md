# CDK Stack

This cdk stack provisions resources and deploys a streamlit app as a
Fargate ECS cluster. The [`streamlit_site_stack.py`](streamlit_site_stack.py)
file deploys the main streamlit app and uses classes defined in
[`sagemaker.py`](sagemaker.py) and [`lambda_.py`](lambda_.py) to deploy
SageMaker endpoints and Lambda functions that have been defined in
[`cdk.json`](../cdk.json).

## Table of Contents

 - [The Main Stack](#the-main-stack)
 - [Using cdk.json](#using-cdkjson)
 - [sagemaker.py](#sagemakerpy)
 - [lambda_.py](#lambda_py)


### The Main Stack

[`streamlit_site_stack.py`](streamlit_site_stack.py) is the main file
from which all resources are provisioned. It reads the context from
[`cdk.json`](../cdk.json) to determine the deployment configuration.
The file will create the following resources:

1. Lambda functions (based on classes in [`lambda_.py`](lambda_.py)).

2. Sagemaker endpoints (based on classes in
   [`sagemaker.py`](sagemaker.py)).

3. A Virtual Private Cloud (VPC).

4. A Cname Record and Certificate, using the hosted zone and domain
   name provided in [`cdk.json`](../cdk.json).

5. An Elastic Container Service (ECS).

6. A docker image from the [`Dockerfile`](../streamlit_app/Dockerfile)
   that is located in the `streamlit_app` folder. This docker image is
   then used to create containers which the streamlit app runs on when
   it is deployed.

7. A Fargate service running on an ECS cluster fronted by an
   Application Load Balancer (ALB). The ALB handles incoming traffic
   and distributes it to instances in the ECS cluster which run the
   streamlit app.

8. A security group allowing all incoming traffic. This allows anybody
   to access the webpage.

### Using [`cdk.json`](../cdk.json)

Each of the arguments supported by [`cdk.json`](../cdk.json) are
described below:

 - `account` `(str)`: The AWS account number of the AWS aacount to
deploy resources in.

 - `region` `(str)`: The region of the cdk stack.

 - `sagemaker_endpoints` `(List[Dict])`: A list of dictionaries
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

 - `lambda_functions` `(List[Dict])`: A list of dictionaries
   containing arguments for a Lambda function class. Each
   dictionary in the list represents an individual Lambda
   function. Each dictionary must contain a key "function_type"
   that maps to a class in lambda_.py, and an
   environment_variable_name which will be used to pass the
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

 - `environment` `(str)`: The environment of the deployment. This
   allows for separate testing and production environments. If the
   deployment is not in the "prod" environment, the environment
name is included in the deployment's domain name.

 - `hosted_zone_id` `(str)`: The hosted zone id of the website's
   domain name.

 - `domain_name` `(str)`: The domain name of the website.

 - `platform` `(str)`: The cpu platform/architecture to run the
   streamlit app on. Can either be `"arm64"` or `"amd64"`. Defaults to
   `"amd64"`, but please take note of your local machine's
   compatibility when modifying this argument.

 - `streamlit_environment_variables` `(Dict[str, Any])`: Extra
   environment variables to pass to the streamlit app. By default
   this variable initializes as empty and is filled with the names
   of external resources (lambda functions and sagemaker
   endpoints).

 - `ecs_policy_statements` `(List[Dict[str, List[str]]])`: A list of
   extra policy statements to add to the streamlit app. By
   default, the app is allowed to invoke SageMaker endpoints and
   Lambda functions. If additional permissions are required, they
   are added through this argument. Structure additional policy
   statements as follows:

   ```
   {
      "resources": ["the resources to include in the policy"],
      "actions": ["the actions to include in the policy"]
   }
   ```


## [`sagemaker.py`](sagemaker.py)

[`sagemaker.py`](sagemaker.py) provides options for deploying custom
and off-the-shelf sagemaker models. It currently supports the following
sagemaker model types:

#### `SagemakerHuggingface`
- `SagemakerHuggingface` allows users to deploy huggingface models
  and supports the deployment of off-the-shelf models (by specifying
  the name of the model) or fine-tuned models (by specifying the model
  data location).

#### `SagemakerFromImageAndModelData`
- `SagemakerFromImageAndModelData` deploys an endpoint that is based on
  a docker image and model data files. The docker image can either be
  built during deployment (by specifying a folder with a Dockerfile in
  it) or pulled from an Elastic Container Registry (ECR) repository (by
  specifying the repo name and image tag).

## [`lambda_.py`](lambda_.py)

[`lambda_.py`](lambda_.py) provides options for deploying lambda
functions. It currently supports the following function types:

#### `LambdaFunctionFromDockerImage`
- `LambdaFunctionFromDockerImage` deploys a lambda image that is based
  on a docker image. The docker image can either be built during
  deployment (by specifying the folder location of the lambda function)
  or pulled from an ECR repository (by specifying the ECR repo name and
  image tag).