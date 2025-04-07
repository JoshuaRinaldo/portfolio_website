# Text Classification (with Explanations!)

This folder contains files that are required to serve a pretrained
text classification model via the `transformers` library. The model is
served as a [flask](https://flask.palletsprojects.com/en/stable/) app.
In addition to classification, the [SHAP](https://shap.readthedocs.io/en/latest/)
library is used to (optionally) provide model explanations. The
container can be run locally, or as an AWS Sagemaker endpoint.

The code to serve the model was written with the goal of creating an
alternative to SageMaker Clarify's realtime model explanation endpoints
**_that can run serverlessly_** (a feature that is currently note
supported by AWS). Serverless functionality of the flask app is
experimental and may not work in all use cases, especially with large
models.

## Building the Image
  To build the docker image, start a container runtime (such as docker,
  podman, or colima) and run `docker build .` in this directory.

## Environment Variables
  This container uses some custom environment variables that can affect
  the way it behaves:

  - `BUCKET_NAME`: The name of the `S3` bucket to read model data from.
    This variable is only necessary if using a custom finetuned model.
  - `MODEL_PATH`: The path to the model file in the S3 bucket.
  - `HUB_MODEL_NAME`: The name of the `transformers` model to pull from
    the huggingface hub. This can be used instead of `BUCKET_NAME` if
    an off-the-shelf model is sufficient for your classification task.
  - `SERVERLESS`: A boolean value indicating whether the container is
  	being run as a serverless inference endpoint. If `True`, the flask
    app will be started with 1 worker (as recommended by AWS).
    Otherwise, the app will be started with a number of workers equal
    to the number of CPU cores on the machine it is being run on.
  - `TOKENIZER`: The name of the `transformers` tokenizer to be used in
    the inference pipeline.
  - `LOCAL`: tells the container if it is being
    run locally, which effects the way the model is loaded. If `LOCAL`
    is not `True`, (the container is being run on a sagemaker
    instance), the model will be downloaded from an `S3` bucket. If
    `LOCAL` is not `True`, both `BUCKET_NAME` and `MODEL_PATH` must
    be defined. If `LOCAL` is  `True`, you must either specify a
    `HUB_MODEL_NAME` or manually move the model data files to the
    container (as described in [running locally](#running_locally))

## Running Locally
  Once the image is built, the model can be run locally:

  1. Start the container:
  ```
    docker run -p 8080:8080 -e LOCAL=True \
    -e TOKENIZER=google-bert/bert-base-cased -e SERVERLESS=True \
    <image tag>
  ```
    
  **Note**: replace the `TOKENIZER` environment variable with the name
  of the tokenizer that your `transformers` model expects. `SERVERLESS`
  isn't required, but it prevents the model from loading multiple
  times because it launches the flask app with only one worker. If
  the model is being pulled from the huggingface hub using the
  `HUB_MODEL_NAME` environment variable, you do not need to complete
  steps 2 and 3.
    
  2. Get the container ID using `docker ps`.

  3. Copy the model checkpoint file into the container:
  ```
  docker cp model.tar.gz <container ID>:model.tar.gz
  ```
  We do this because usually the model is downloaded from an `S3`
  bucket. Since the model is being run locally, we skip that step and
  instead copy the checkpoint file directly into the container.

  And just like that, the container should be running locally and ready
  to classify text and provide explanations. You can test the model by
  sending requests to your `localhost` at port `8080`. Here is an
  example using `curl`:

  ```
  curl -H 'Content-Type: application/json' \
        -d '{ "data":"This is a sentence","explain":true}' \
        -X POST \
        http://127.0.0.1:8080/invocations
  ```