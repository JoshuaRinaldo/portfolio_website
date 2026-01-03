# Lambda Functions

This folder contains Lambda functions that are deployed in the CDK stack. To add Lambda functions to a deployment, create a new folder, place the code in the folder, and add the Lambda function to `cdk.json` under the `lambda_functions` list.

## [`invoke_model`](invoke_model/invoke_model.py)

The [`invoke_model`](invoke_model/invoke_model.py) Lambda function provides a generic interface for invoking SageMaker endpoints. It accepts an endpoint name and payload, invokes the specified SageMaker endpoint, and returns the raw response. This allows the frontend to call any SageMaker model without needing separate Lambda functions for each model type.

The function handles both API Gateway events and direct Lambda invocations, making it flexible for various use cases.
