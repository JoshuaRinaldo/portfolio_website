# CDK Stack

This cdk stack provisions resources and deploys a streamlit app as a
Fargate ECS cluster. The [`streamlit_site_stack.py`](streamlit_site_stack.py)
file deploys the main streamlit app and uses classes defined in
[`sagemaker.py`](sagemaker.py) and [`lambda_.py`](lambda_.py) to deploy
SageMaker endpoints and Lambda functions that have been defined in
[`cdk.json`](../cdk.json).

## [`streamlit_site_stack.py`](streamlit_site_stack.py)

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