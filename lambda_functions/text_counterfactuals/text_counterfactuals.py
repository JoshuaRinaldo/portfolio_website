import boto3
import json
from typing import List
import re
from random import sample
from statistics import stdev

import spacy
nlp = spacy.load("en_core_web_sm")


runtime_sagemaker_client = boto3.client(service_name='sagemaker-runtime')

def get_predicted_label(endpoint_response):
    """
    Finds and returns the max score based on a response from a
    multiclass endpoint

    Args:
        endpoint_response (dict): The response from a multiclass
        endpoint.

    Returns:
        predicted_label, score: str, float. The highest scoring label
        and its respective score.
    """
    labels = []
    predictions = []
    for label_n in endpoint_response["prediction"]:
        labels.append(label_n["label"])
        predictions.append(label_n["score"])
    score = max(predictions)
    predicted_label = labels[predictions.index(score)]
    return predicted_label, score

def invoke_endpoint(
        request: dict,
        endpoint_name: str,
        ):
    """
    Invokes a sagemaker endpoint

    Args:
        request (dict): The dictionary containing all arguments to pass
        to the endpoint.

        endpoint_name (str): The name of the endpoint to invoke.

    Returns:
        result: Any. The response from the endpoint

    """
    content_type = "application/json"
    payload = json.dumps(request)
    response = runtime_sagemaker_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=content_type,
        Body=payload
    )
    result = json.loads(response['Body'].read().decode())
    return result

def mask_explanation(
        explanation: dict,
        ):
    """
    This function uses Shapley values to mask important tokens that
    are in a string. The function finds which tokens are most
    impactful, then masks them. The function masks all tokens
    that are greater than a cutoff (beginning at the standard
    deviation of the Shapley values multiplied by 0.8), but adjusts
    the cutoff point depending on how many tokens are captured, aiming
    for a number of masks between 1 and 4.

    Args:
        explanation (List): The explanation returned from the
        classification endpoint after being processed to determine if
        each token can be unmasked.

    Returns:
        List[List[str], List[int]]. prediction_tokens is a list of
        strings with a series of masked tokens. mask_indices is each
        masked token's respective index.

    """

    prediction = explanation["prediction"]
    response_dict = {}
    for label_n in prediction:
        response_dict[label_n["label"]] = label_n["score"]

    predicted_class = max(response_dict, key=response_dict.get)

    shap_values = []
    prediction_tokens = []
    allow_unmask = []
    token_pos = []
    for token_n, vals_n, token_pos_n, allow_unmask_n in explanation["explanation"]:
        shap_values.append(vals_n[predicted_class])
        prediction_tokens.append(token_n)
        token_pos.append(token_pos_n)
        allow_unmask.append(allow_unmask_n)

    # Define the cutoff threshold using the standard deviation of the
    # shap values
    standard_deviation = stdev(shap_values)
    std_adjustment = 0.8
    mask_indices = [0]*len(prediction_tokens)

    # We use to counter to avoid infinite loops in the event of an error
    counter = 0
    while counter < 20:
        
        mask_indices = []
        for index, shap_value_n in enumerate(shap_values):

            # Mask the token if the shap value is above the cutoff and
            # the token is allowed to be unmasked
            if shap_value_n > standard_deviation*std_adjustment and allow_unmask[index]:
                mask_indices.append(index)

        if len(mask_indices) == 0:

            # If no tokens are masked, we lower the threshhold for
            # masking tokens
            std_adjustment -= 0.1
        
        elif len(mask_indices) > 4:

            # If more than acceptable # of tokens are masked, we
            # reduce the threshhold for masking tokens
            std_adjustment += 0.1
 
        # If in acceptable range, we mask the tokens are return the
        # result
        else: 
            for index in mask_indices:
                if " " in prediction_tokens[index]:
                    prediction_tokens[index] = "[MASK] "
                else:
                    prediction_tokens[index] = "[MASK]"
            break
        counter += 1
    
    return prediction_tokens, mask_indices


def unmask_string(
        masked_string: List[str],
        mask_indices: List[int],
        desired_class: str,
        classification_model: str,
        masked_language_model: str,
        top_k: int=2,
        all_accepted_unmasks: List[str] = None
        ):
    """
    A function that accepts a masked string and uses a masked language
    model to generate alternative strings. The string is unmasked from
    left to right (iterating over the `masked_indices` variable) and is
    "steered" towards a desired class by selecting unmasked token
    predictions that result in a higher score in the desired class.

    Args:
        masked_string (List[str]): The masked string, broken up into
        a list of its tokens.

        mask_indices (List[int]): A list of indices that point towards
        tokens that will be masked and unmasked.

        desired_class (str): The class that the string will be
        "steered" towards while counterfactuals are generated.

        classification_model (str): The name of the classification
        model that will be used to determine if an unmasked token
        brings the string closer to the desired class.

        masked_language_model (str): The name of the masked language
        model that will unmask the tokens specified in `mask_indices`.

        top_k (int): The number of predicted tokens to keep when
        unmasking a token. If `top_k`. is greater than or equal to the
        number of predictions, all of them are kept. Otherwise only the
        unmasked tokens that moved the predicted class closer to
        `desired_class` are kept.

        all_accepted_unmasks (List[str]): A list of the unmasked
        strings that pass the `top_k` cutoff at their respective
        unmasking step.

    Returns:
        Union[str, List[str]]. A list of strings or an error message.

    """

    # If at the first level of recursion, initialize the list that
    # we will append to and eventually return
    if not all_accepted_unmasks:
        all_accepted_unmasks = []

    # Replace [MASK] with UNK for all but the leftmost mask. We will
    # slowly replace each 
    for mask_n in mask_indices[1:]:
        masked_string[mask_n] = "UNK "

    leftmost_mask = mask_indices[0]
    masked_string[leftmost_mask] = "[MASK] "

    
    # Unmask leftmost mask
    unmask_request = {
        "inputs": "".join(masked_string)
    }
    unmask_response = invoke_endpoint(
        request=unmask_request,
        endpoint_name=masked_language_model,
    )
    
    # Replace masked token with predicted strings, then predict class
    # of text. We keep the top k predictions that have the highest
    # score of the desired class
    unmasked_string_class_prediction = []
    unmasked_strings = []
    for unmasked_label in unmask_response:
        unmasked_str = masked_string.copy()
        unmasked_str[leftmost_mask] = unmasked_label["token_str"] + " "
        unmasked_str_joined = "".join(unmasked_str)

        class_request = {"data": unmasked_str_joined}
        response = invoke_endpoint(class_request, classification_model)["prediction"]

        response_dict = {}
        for label_n in response:
            response_dict[label_n["label"]] = label_n["score"]
        
        unmasked_strings.append(unmasked_str)
        unmasked_string_class_prediction.append(response_dict[desired_class])

    
    # Take top k scores for our desired class
    if top_k >= len(unmasked_strings):
        accepted_unmasks = unmasked_strings
    else:

        accepted_unmasks = []
        while top_k > len(accepted_unmasks):
            max_class = max(unmasked_string_class_prediction)
            max_class_index = unmasked_string_class_prediction.index(max_class)
            accepted_unmasks.append(unmasked_strings[max_class_index])

            unmasked_string_class_prediction.pop(max_class_index)
            unmasked_strings.pop(max_class_index)
    
    if len(mask_indices) > 1:
        all_accepted_unmasks = []
        for accepted_unmasked_str_n in accepted_unmasks:
            unmasked_strings = unmask_string(
                accepted_unmasked_str_n,
                mask_indices[1:],
                desired_class,
                classification_model,
                masked_language_model,
                top_k,
                )
            all_accepted_unmasks.extend(unmasked_strings)
        return all_accepted_unmasks
            
    else:
        for unmasked_str_n in accepted_unmasks:
            all_accepted_unmasks.append("".join(unmasked_str_n))
        return all_accepted_unmasks


def handler(event, context):
    """
    This function generates text counterfactuals using SHAP values
    from a text classification model and a masked language model. The
    function finds SHAP values that indicate which tokens were
    important when making a prediction. For example, "hate" might have
    a large impact on the model's prediction of negative sentiment. The
    tokens which have the highested impact on the undesired class are
    masked with the placeholder "[MASK]". The tokens are then unmasked
    using the masked language model.

    Args:
        input_text (str): The starting text which will have
        counterfactuals generated based on it.

        classification_model (str): The name of the classification
        sagemaker endpoint being used to generate explanations.

        masked_language_model (str): The name of the masked language
        model sagemaker endpoint that will be used to unmask important
        tokens.

        desired_class (str): The name of the desired class. The input
        text will be "steered" towards this class during the
        counterfactual generation process.

        undesired_class (str): The name of the undesired class. This
        tells the function which tokens to mask.

        unmasking_token_types (List(str)): A list of strings where each
        string is a tag from SpaCy's part-of-speech tagging. Tags in
        this list will be marked as acceptable to mask and replace.

        depth (int): In special cases, we want to re-run the process
        on an input text that has already gone through the
        counterfactual generation process. In this case, we re-run the
        process, but track depth to avoid excessive recursion.

    Returns:
        Union[str, List[str]]. A list of strings or an error message.

    """

    input_text = event["input_text"]
    classification_model = event["classification_model"]
    masked_language_model = event["masked_language_model"]
    desired_class = event["desired_class"]
    undesired_class = event["undesired_class"]
    unmasking_token_types = event.get(
        "unmasking_token_types",
        ["ADJ", "ADV", "AUX", "CCONJ", "INTJ", "PART", "PUNCT"],
        )
    depth = event.get("depth", 0)
    
    # We remove apostrophes to simplify tokenization and masking
    input_text = re.sub("'", "", input_text)

    # Generate explanation using the classification model
    request = {"data": input_text, "explain": True}
    endpoint_response = invoke_endpoint(request, endpoint_name=classification_model)
    
    label, score = get_predicted_label(endpoint_response)

    if label == desired_class and score>0.5:
        return {
            "result": input_text,
            "endpoint_response": endpoint_response,
            "message": (
                "The model classified your input text as being the desired class,"
                " so there is no need to provide a counterfactual."
            ),
        }


    tokens = []
    for token_n in endpoint_response["explanation"]:
        tokens.append(token_n[0])

    tokens_text = []
    tokens_pos = []
    for token_n in tokens:
        doc = nlp(token_n)
        part_of_speech = []
        tokens = []
        for token in doc:
            part_of_speech.append(token.pos_)
            tokens.append(token.text)
        
        tokens_text.append(tokens[0] if len(tokens)>0 else "")
        tokens_pos.append(part_of_speech[0] if len(part_of_speech)>0 else "X")
    
    # Find the most significant token in the string so we can override
    # disallowed part-of-speech masking
    max_undesired_shapley = 0
    for token_index in range(len(endpoint_response["explanation"])):
        analyzed_tokens_n = endpoint_response["explanation"][token_index]
        max_undesired_shapley = max(max_undesired_shapley, analyzed_tokens_n[1][undesired_class])

    for token_index in range(len(endpoint_response["explanation"])):
        analyzed_tokens_n = endpoint_response["explanation"][token_index]
        token_pos_n = tokens_pos[token_index]
        analyzed_tokens_n.append(token_pos_n)

        # To maintain the meaning, we only allow the masking of certain
        # types of tokens. In special cases where a token is very
        # important in the undesired class prediction, we allow its
        # replacement regardless of its part-of-speech
        if token_pos_n in unmasking_token_types:
            analyzed_tokens_n.append(True)
        elif analyzed_tokens_n[1][undesired_class] > max_undesired_shapley*0.8:
            analyzed_tokens_n.append(True)
        else:
            analyzed_tokens_n.append(False)
        
        endpoint_response["explanation"][token_index] = analyzed_tokens_n

    # Mask tokens that contribute most to the currently predicted class
    masked_string, mask_indices = mask_explanation(
        endpoint_response
        )

    if len(mask_indices) == 0:
        return {
            "result": input_text,
            "endpoint_response": endpoint_response,
            "message": (
                "The function failed to mask any input tokens, so no"
                " counterfactuals can be provided."
            ),
        }

    # Apply mlm to unmask the newly masked tokens
    unmasked_strings = unmask_string(
        masked_string=masked_string,
        mask_indices = mask_indices,
        desired_class = desired_class,
        classification_model=classification_model,
        masked_language_model=masked_language_model,
        top_k=max(3, int(6/len(mask_indices)))
        )
    
    # Only send back strong (score>0.5 in desired class)
    # counterfactuals
    labels, scores = [], []
    for new_string_n in unmasked_strings:
        request = {"data": new_string_n}
        endpoint_response_n = invoke_endpoint(request, endpoint_name=classification_model)
        label_n, score_n = get_predicted_label(endpoint_response_n)
        labels.append(label_n)
        scores.append(score_n)

    counterfactuals_to_keep = []
    for n in range(len(unmasked_strings)):
        counterfactual = unmasked_strings[n]
        label = labels[n]
        score = scores[n]

        if label == desired_class and score > 0.5:
            counterfactuals_to_keep.append(counterfactual)

    # If there are no strong counterfactuals, we send the closest
    # counterfactual back to the handler for another attempt. We start
    # by finding the strongest in desired class string, if no strings
    # are in the desired class, we settle for the lowest scoring not-in
    # -class string.
    if len(counterfactuals_to_keep) == 0 and depth<3:
        slightly_positive = []
        slightly_positive_scores = []
        for n in range(len(unmasked_strings)):
            label = labels[n]
            if label == desired_class:
                slightly_positive.append(slightly_positive)
                slightly_positive_scores.append(scores[n])
        if len(slightly_positive)>0:
            string_to_send = slightly_positive[
                    slightly_positive_scores.index(
                        max(slightly_positive_scores)
                        )
                    ]
        else:
            string_to_send = unmasked_strings[
                    scores.index(
                        min(scores)
                        )
                    ]

        return handler(
            event={
                "input_text": string_to_send,
                "classification_model": classification_model,
                "masked_language_model": masked_language_model,
                "desired_class": desired_class,
                "undesired_class": undesired_class,
                "unmasking_token_types": unmasking_token_types,
                "depth": (depth+1)
            },
            context=context,
            )
    # If after three attempts at generating counterfactuals, return
    # whatever we have    
    elif len(counterfactuals_to_keep) == 0 and depth == 3:
        return {
            "result": unmasked_strings,
            "endpoint_response": endpoint_response,
            "message": (
                "Strong counterfactuals could not be generated."
                " Returning current counterfactuals."
            )
        }

    return {
        "result": counterfactuals_to_keep,
        "endpoint_response": endpoint_response,
        "message": "Counterfactuals successfully generated"
        }