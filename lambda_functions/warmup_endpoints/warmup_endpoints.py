import json
import boto3
import os
from datetime import datetime, timezone

sagemaker_runtime = boto3.client('sagemaker-runtime')

# Get all endpoint names from environment variables
ENDPOINT_NAMES = [
    os.environ.get(key)
    for key in os.environ.keys()
    if key.endswith('_ENDPOINT_NAME')
]


def handler(event, context):
    """
    Warm all endpoints by sending a ping request.
    Always warms endpoints regardless of last warmup time.
    This serves as both a warmup trigger and a readiness check.
    """
    try:
        current_time = datetime.now(timezone.utc)
        current_timestamp = int(current_time.timestamp())

        warmup_results = []
        endpoints_warmed = []

        for endpoint_name in ENDPOINT_NAMES:
            if not endpoint_name:
                continue

            # Always warm the endpoint
            try:
                # Send minimal request to wake up the endpoint
                sagemaker_runtime.invoke_endpoint(
                    EndpointName=endpoint_name,
                    ContentType='application/json',
                    Body=json.dumps({
                        "data": "warmup",
                        "explain": False
                    })
                )

                endpoints_warmed.append(endpoint_name)
                warmup_results.append({
                    'endpoint': endpoint_name,
                    'status': 'warmed',
                    'timestamp': current_timestamp
                })

            except Exception as e:
                warmup_results.append({
                    'endpoint': endpoint_name,
                    'status': 'failed',
                    'error': str(e)
                })

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
            },
            'body': json.dumps({
                'success': True,
                'endpoints_warmed': endpoints_warmed,
                'total_endpoints': len(ENDPOINT_NAMES),
                'results': warmup_results
            })
        }

    except Exception as e:
        print(f"Error in warmup handler: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
            },
            'body': json.dumps({
                'success': False,
                'error': str(e)
            })
        }
