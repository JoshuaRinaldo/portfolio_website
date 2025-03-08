# Pesonal Website Streamlit App

This repository provides a flexible cdk stack for machine learning
engineers to demonstrate their personal projects. It was originally
authored by [Josh Rinaldo](https://github.com/JoshuaRinaldo).

The resources in this stack can be modified by editing the
[`cdk.json`](cdk.json) file's context. Sagemaker endpoints and Lambda
functions are built based on arguments provided in the context, so
models and other capabilities can be added or removed with relative
ease.

To use this repository as a base for your personal website, simply
clone or fork the repo, change arguments in [`cdk.json`](cdk.json)
to deploy your custom machine learning endpoints, then modify the
[streamlit app](streamlit_app/README.md) and make it your own!

Please see the cdk stack's [README.md](cdk_stack/README.md) for a
breakdown of the deployment architecture and instructions on how to use
[`cdk.json`](cdk.json).