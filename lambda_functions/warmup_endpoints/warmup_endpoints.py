import json
import boto3
import os
from datetime import datetime, timezone
from decimal import Decimal

dynamodb = boto3.resource('dynamodb')
sagemaker_runtime = boto3.client('sagemaker-runtime')

WARMUP_TABLE_NAME = os.environ.get('WARMUP_TABLE_NAME')
WARMUP_INTERVAL_SECONDS = int(os.environ.get('WARMUP_INTERVAL_SECONDS', 300))  # Default 5 minutes

# Get all endpoint names from environment variables
ENDPOINT_NAMES = [
    os.environ.get(key)
    for key in os.environ.keys()
    if key.endswith('_ENDPOINT_NAME')
]


def handler(event, context):
    """
    Check if endpoints need warming and warm them if necessary.
    Uses DynamoDB to track last warmup time per endpoint.
    Only warms if >WARMUP_INTERVAL_SECONDS have passed since last warmup.
    """
    try:
        table = dynamodb.Table(WARMUP_TABLE_NAME)
        current_time = datetime.now(timezone.utc)
        current_timestamp = int(current_time.timestamp())

        warmup_results = []
        endpoints_warmed = []

        for endpoint_name in ENDPOINT_NAMES:
            if not endpoint_name:
                continue

            # Check last warmup time from DynamoDB
            response = table.get_item(Key={'endpoint_name': endpoint_name})

            needs_warmup = False
            if 'Item' not in response:
                # No record exists, needs warmup
                needs_warmup = True
            else:
                last_warmup = int(response['Item']['last_warmup_timestamp'])
                time_since_warmup = current_timestamp - last_warmup

                if time_since_warmup >= WARMUP_INTERVAL_SECONDS:
                    needs_warmup = True

            if needs_warmup:
                # Warm the endpoint with a simple ping request
                try:
                    # Send minimal request to wake up the endpoint
                    # Using a simple text classification request
                    sagemaker_runtime.invoke_endpoint(
                        EndpointName=endpoint_name,
                        ContentType='application/json',
                        Body=json.dumps({
                            "data": "warmup",
                            "explain": False
                        })
                    )

                    # Update DynamoDB with new warmup timestamp
                    table.put_item(
                        Item={
                            'endpoint_name': endpoint_name,
                            'last_warmup_timestamp': current_timestamp,
                            'last_warmup_time': current_time.isoformat()
                        }
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
            else:
                warmup_results.append({
                    'endpoint': endpoint_name,
                    'status': 'already_warm',
                    'last_warmup': response['Item']['last_warmup_time']
                })

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
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
            },
            'body': json.dumps({
                'success': False,
                'error': str(e)
            })
        }
