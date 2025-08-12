# Import the necessary libraries
import json
import os
import flask
import boto3
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import tarfile
import shap


class ScoringService(object):

    def __init__(self):
        self.pipeline=None
        self.explainer=None

    def get_pipeline(self):
        """
        Read from S3 (if not running locally) and load the model from
        opt/ml/model.
        """
        if self.pipeline == None:
            container_is_local = bool(os.getenv("LOCAL", default=False))
            

            if os.getenv("HUB_MODEL_NAME", default=None):
                model_name = os.getenv("HUB_MODEL_NAME")

                # If tokenizer name is not specified, set to model name
                tokenizer_name = os.getenv("TOKENIZER", default=model_name)
            else:

                # If hub model is not defined, assume the instance
                # is using a model data file
                model_name = "/tmp/model"
                try:
                    tokenizer_name = os.getenv("TOKENIZER")
                except KeyError:
                    raise RuntimeError(
                        "Tokenizer must be defined as an environment"
                        " variable if using a custom model data file"
                    )
            
            if not container_is_local and not os.getenv("HUB_MODEL_NAME", default=None):

                # Load bucket name and model path environment variables
                try:
                    bucket_name = os.getenv("BUCKET_NAME")
                    model_path = os.getenv("MODEL_PATH")
                except KeyError:
                    raise RuntimeError(
                        "BUCKET_NAME and MODEL_PATH environment variables must"
                        " be defined if the model is not being run locally"
                    )
            
                s3_client = boto3.client('s3')
                s3_client.download_file(
                    Bucket=bucket_name,
                    Key=model_path,
                    Filename="/tmp/model.tar.gz"
                )

                # Extract model files
                tar = tarfile.open("/tmp/model.tar.gz", "r:gz")
                tar.extractall(path="/tmp/model")
                tar.close()


            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name)
            self.pipeline = pipeline("text-classification", tokenizer = tokenizer, model = model)
            self.explainer = shap.Explainer(self.pipeline)
        
        return self.pipeline, self.explainer
        
    def predict(self, request):
        data = request["data"]
        return_dict = {}
        clf, explainer = self.get_pipeline()
        if request.get("explain"):
            explanation = explainer([data])
            labels = list(explanation.output_names),
            values = explanation.values.tolist()[0],
            _input = explanation.data[0].tolist(),
            explanation = []

            for token_n in range(len(_input[0])):
                explanation_values = {}
                for label_n in range(len(labels[0])):
                    explanation_values[labels[0][label_n]] = values[0][token_n][label_n]

                explanation.append([_input[0][token_n], explanation_values])

            return_dict["explanation"] = explanation

        prediction = clf(data, top_k=None)
        return_dict["prediction"] = prediction
        return return_dict



# Instantiate Flask app
app = flask.Flask(__name__)

scorer = ScoringService()

# Define an endpoint for health check
@app.route('/ping', methods=['GET'])
def ping():
    return '', 200

@app.route('/invocations', methods=['POST'])
def predict():
    """
    Predicts class of input text, optionally provides an explanation
    using SHAP.

    Args:
        data (str): The starting text which will be passed to the
        pipeline.

        explain (bool): Whether or not to provide an explanation.

    """

    request = flask.request.data.decode("utf-8")
    request = json.loads(request)
    predictions = scorer.predict(request)

    return json.dumps(predictions), 200
