import streamlit as st

import boto3
import os
import json

lambda_client = boto3.client("lambda")

counterfactual_lambda = os.getenv("COUNTERFACTUAL_LAMBDA")
sentiment_unmasking_model = os.getenv("SENTIMENT_UNMASKING_MODEL")
sentiment_classification_model = os.getenv("SENTIMENT_CLASSIFICATION_MODEL")


input_text = st.text_input("Insert text to provide counterfactuals for")

if st.button("Press to generate counterfactuals"):

    response = lambda_client.invoke(
        FunctionName=counterfactual_lambda,
        Payload=json.dumps(
            {
                "input_text": input_text,
                "classification_model": sentiment_classification_model,
                "masked_language_model": sentiment_unmasking_model,
                "desired_class": "LABEL_2",
                "undesired_class": "LABEL_0",
            }
        )
    )

    st.json(response)

    payload = json.loads(response['Payload'].read())

    st.json(payload)