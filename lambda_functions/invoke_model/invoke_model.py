import boto3
import json
import logging
import os

# Set up our logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

runtime_sagemaker_client = boto3.client(service_name='sagemaker-runtime')

# Allowlist of permitted SageMaker endpoints - only these can be invoked
# Populated from environment variables set by CDK deployment
ALLOWED_ENDPOINTS = {
    os.environ.get(key)
    for key in os.environ.keys()
    if key.endswith('_CLASSIFICATION_MODEL')
}
# Remove None values in case any env vars are missing
ALLOWED_ENDPOINTS.discard(None)

logger.info(f"Allowed endpoints: {ALLOWED_ENDPOINTS}")

def handler(event, context):
    """
    Generic Lambda function for invoking SageMaker endpoints.

    This function acts as a proxy between API Gateway and SageMaker endpoints,
    allowing the frontend to invoke any SageMaker model without needing
    separate Lambda functions for each model type.

    This handler supports both API Gateway (HTTP) and direct Lambda invocations.

    Request body format:
        {
            "endpoint_name": str,  # Required: The SageMaker endpoint to invoke
            "payload": dict,       # Required: The payload to send to the endpoint
            "content_type": str    # Optional: Content type (default: "application/json")
        }

    Returns:
        The raw response from the SageMaker endpoint, allowing the frontend
        to handle model-specific response formatting.
    """
    def format_response(data, status_code=200):
        """Format response for API Gateway or direct Lambda invocation."""
        if "body" in event:
            # API Gateway response
            return {
                "statusCode": status_code,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Access-Control-Allow-Methods": "POST, OPTIONS"
                },
                "body": json.dumps(data)
            }
        else:
            # Direct Lambda invocation response
            return data

    try:
        # Detect if this is an API Gateway event and extract body
        if "body" in event:
            # API Gateway event
            try:
                body = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]
            except json.JSONDecodeError:
                return format_response({"error": "Invalid JSON in request body"}, 400)
        else:
            # Direct Lambda invocation
            body = event

        # Extract required parameters
        endpoint_name = body["endpoint_name"]
        payload = body["payload"]
        content_type = body.get("content_type", "application/json")

        # Validate endpoint name against allowlist
        if endpoint_name not in ALLOWED_ENDPOINTS:
            logger.warning(f"Attempted to invoke unauthorized endpoint: {endpoint_name}")
            return format_response({
                "success": False,
                "error": "Invalid or unauthorized endpoint name"
            }, 403)

        logger.info(f"Invoking endpoint: {endpoint_name}")

        # Invoke the SageMaker endpoint
        response = runtime_sagemaker_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType=content_type,
            Body=json.dumps(payload)
        )

        # Parse and return the response
        result = json.loads(response['Body'].read().decode())

        return format_response({
            "success": True,
            "endpoint_name": endpoint_name,
            "result": result
        })

    except KeyError as e:
        logger.error(f"Missing required parameter: {e}")
        return format_response({
            "success": False,
            "error": f"Missing required parameter: {str(e)}"
        }, 400)
    except Exception as e:
        logger.error(f"Error invoking endpoint: {e}")
        return format_response({
            "success": False,
            "error": "An error occurred while invoking the model endpoint"
        }, 500)
