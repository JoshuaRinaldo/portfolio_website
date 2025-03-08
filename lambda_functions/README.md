# Lambda Functions

This folder contains lambda functions that are deployed in the `cdk`
stack. To add lambda functions to a deployment, create a new folder,
place the code in the folder, and add the lambda function to `cdk.json`
under the `lambda_functions` list.

## [`Text Counterfactuals`](text_counterfactuals/text_counterfactuals.py)
The [`text_counterfactuals`](text_counterfactuals/text_counterfactuals.py)
lambda function uses Explainable AI (XAI) and a masked language model
to generate text classification counterfactuals for short strings. 

The function generates counterfactuals by replacing tokens to move strings
away from the `undesired_class` and towards the `desired_class`. To
do this, the function uses a classification model that can provide explanations
using the [SHAP](https://shap.readthedocs.io/en/latest/index.html)
library to determine which tokens had the most impact on the score. The
function then masks those tokens and uses a masked language model to
predict replacement tokens. Preferrably, the masked language model is
finetuned on text examples that are in the `desired_class`.