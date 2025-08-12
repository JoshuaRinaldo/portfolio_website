ENDPOINT_NAME="$1"

echo $ENDPOINT_NAME

aws-vault exec $AWS_PROFILE -- aws sagemaker-runtime invoke-endpoint --endpoint-name $ENDPOINT_NAME --content-type application/json --body '{"data": "This is some text", "explain": true}' --cli-binary-format raw-in-base64-out  tmp.json

echo "Endpoint Response: "
cat tmp.json
