import boto3
import json
import logging

# Set up our logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

runtime_sagemaker_client = boto3.client(service_name='sagemaker-runtime')

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
