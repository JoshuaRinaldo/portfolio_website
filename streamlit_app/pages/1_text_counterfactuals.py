import streamlit as st

import asyncio
import base64
import boto3
import os
import json
import re
import time
import threading

lambda_client = boto3.client("lambda")

counterfactual_lambda = os.getenv("COUNTERFACTUAL_LAMBDA")
sentiment_unmasking_model = os.getenv("SENTIMENT_UNMASKING_MODEL")
sentiment_classification_model = os.getenv("SENTIMENT_CLASSIFICATION_MODEL")
toxicity_classification_model = os.getenv("TOXICITY_CLASSIFICATION_MODEL")


classification_models = {
    "sentiment": {
        "model": sentiment_classification_model,
        "desired_label": "positive",
        "undesired_label": "negative",
        },
    "toxicity": {
        "model": toxicity_classification_model,
        "desired_label": "neutral",
        "undesired_label": "toxic",
        },
}

def ping_lambda(counterfactual_endpoint_dict):
    """
    Pings a lambda. This is used to keep lambdas warm and mitigate
    cold-starts
    """
    try:
        lambda_client.invoke(
            FunctionName=counterfactual_lambda,
            InvocationType='Event',
            Payload=json.dumps({
                "input_text": "This is a ping",
                "classification_model": counterfactual_endpoint_dict["model"],
                "masked_language_model": sentiment_unmasking_model,
                "desired_class": counterfactual_endpoint_dict["desired_label"],
                "undesired_class": counterfactual_endpoint_dict["undesired_label"],
                "ping": True,
            })
        )
    except Exception as e:
        print(f"Lambda ping failed: {e}")

def ping_all_lambdas_background():
    for endpoint in classification_models.values():
        threading.Thread(target=ping_lambda, args=(endpoint,), daemon=True).start()

# Ping lambdas at refresh and every reload that occurs >300 ms after
# last ping
if 'last_ping_time' not in st.session_state or time.time() - st.session_state['last_ping_time'] > 300:
    ping_all_lambdas_background()
    st.session_state['last_ping_time'] = time.time()

if "display_text" not in st.session_state:
    st.session_state.display_text = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = sentiment_classification_model


def create_hex_css(explanation, desired_label, undesired_label):
    """
    Renders a model explanation, displaying the shapley values for each
    token on a discrete colorscale from red to white to green.

    Args:
        explanation (dict): The explanation returned by the text
        counterfactual lambda function.

        desired_label (str): The desired label. Shapley values that
        contribute to the desired label will be rendered as green.

        undesired_label (str): The undesired label. Shapley values that
        contribute to the undesired label will be rendered as red.
    """
    greens = ["d6e6d5", "e1ffe0", "c8ffc7", "a8faa7", "90ff8f", "6bff69", "40ff3d", "07fc03"]
    reds = ["e6d5d5", "facfcf", "ffbaba", "ffa1a1", "ff8585", "ff6666", "ff3d3d", "ff1919"]
    base_css = '<mark style="background-color: #{hex_color};">{text}</mark>'
    output_str = ""
    
    for token_explanation_n in explanation:
        token_n = token_explanation_n[0]

        # We collapse the shapley values to simplify coloring
        collapsed_value = token_explanation_n[1][desired_label] - token_explanation_n[1][undesired_label]

        color_index = min(int(abs(collapsed_value)*20), 7)
        if collapsed_value > 0:
            color = greens[color_index]
        elif collapsed_value < 0:
            color = reds[color_index]
        else:
            color = "fcfcfc"

        token_css_string = base_css.format(hex_color=color, text=token_n)
        output_str += token_css_string

    return output_str

st.title("Text Counterfactuals")

file_ = open("static/text_counterfactual_animation.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown(
    f'''<p align="center">
    <img src="data:image/gif;base64,{data_url}" alt="A diagram showing a string being converted from positive to negative sentiment by replacing the word terrible with the word wonderful">
    </p>''',
    unsafe_allow_html=True,
)

st.markdown(
"""
## Welcome!

This is the webpage for my text counterfactuals project. It provides
access to live models which you can use to generate your own
counterfactuals.
"""
)

st.markdown("## Blog Post")

file_ = open("static/text_counterfactuals_blog_post.md", "rb")
blog_post = file_.read().decode("utf-8")
file_.close()

with st.expander("Blog Post"):
    st.markdown(blog_post)

st.markdown("## Try it Yourself")
with st.expander("Try it yourself"):
    st.markdown("Enter a short phrase to generate counterfactuals for")
    input_text = st.text_input(label="Enter Text", value="")

    st.session_state.selected_model = st.selectbox(
        label="model dropdown",
        options = [
            "sentiment",
            "toxicity",
        ],
    )
    st.markdown(
    "This project was built and tested for generating sentiment"
    " classification counterfactuals, but it can work for other"
    " classification tasks. You can choose between generating"
    " sentiment or toxicity text counterfactuals."
    )

    if len(input_text)>100:
        st.markdown(
            "Your input is too long. Your input must be fewer than 100 characters."
            )
    elif len(re.sub("<.*>", "", input_text)) < len(input_text):
            st.markdown(
                "You input was detected as potentially containing HTML. To avoid"
                " HTML injections, this field does not accept any input that is"
                " wrapped in HTML tags: `<>`"
            )
    elif st.button("Press to generate counterfactuals"):

        st.markdown(
            "Counterfactual generation is being done serverlessly and"
            " generating counterfactuals is computationally expensive."
            " You may experience latency, especially if this is your"
            " first invocation."
        )

        desired_label = classification_models[st.session_state.selected_model]["desired_label"]
        undesired_label = classification_models[st.session_state.selected_model]["undesired_label"]

        # Invoke lambda function
        response = lambda_client.invoke(
            FunctionName=counterfactual_lambda,
            Payload=json.dumps(
                {
                    "input_text": input_text,
                    "classification_model": classification_models[st.session_state.selected_model]["model"],
                    "masked_language_model": sentiment_unmasking_model,
                    "desired_class": desired_label,
                    "undesired_class": undesired_label,
                }
            )
        )

        # Parse and render response in markdown
        counterfactual_response = json.loads(response['Payload'].read())

        explanation = counterfactual_response["endpoint_response"]["explanation"]
        explanation_colored = create_hex_css(
            explanation=explanation,
            desired_label=desired_label,
            undesired_label=undesired_label,
            )

        display_text = (
            f"### {counterfactual_response['message']}\n"
            f"### Your Input's Explanation:\n\n<center>\n\n#### {explanation_colored}\n\n</center>\n(green indicates"
            f" a contribution towards the desired label ({desired_label}), red indicates"
            f" a contribution towards the undesired label ({undesired_label}))\n\n"
            " ### Counterfactual Generation Response:\n"
        )

        counterfactual_table = "<center>\n\n| Counterfactual        | Label | Score |\n| :---------------- | :------: | :----: |\n"
        for counterfactual_n in counterfactual_response["result"]:
            counterfactual_table += f"| {counterfactual_n[0]} | {counterfactual_n[1]} | {round(counterfactual_n[2], 3)} |\n"
        counterfactual_table += "\n</center>"
        display_text += counterfactual_table

        # Add rendered text to session state
        st.session_state.display_text = display_text
    
    if st.session_state.display_text:
        st.markdown(
            st.session_state.display_text,
            unsafe_allow_html=True,
        )
