import boto3
import json
from typing import List
from statistics import stdev

runtime_sagemaker_client = boto3.client(service_name='sagemaker-runtime')

def invoke_endpoint(request, endpoint_name):
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
        explanation,
        ):
    """
    This function parses the response from an endpoint that generates
    explanations using the SHAP library. The Shapley values for each
    token are stored, and the predicted class of the token
    """

    prediction = explanation["prediction"]
    response_dict = {}
    for label_n in prediction:
        response_dict[label_n["label"]] = label_n["score"]

    predicted_class = max(response_dict, key=response_dict.get)

    print(predicted_class)

    prediction_weights = []
    prediction_tokens = []
    for token_n, vals_n in explanation["explanation"]:
        prediction_weights.append(vals_n[predicted_class])
        prediction_tokens.append(token_n)

    # We run a loop to adjust the cutoff point for when to mask a
    # token. We do not accept more than 4 masks, but require one mask.
    # If we are above or below the target number of masks, we adjust
    # the threshold.
    standard_deviation = stdev(prediction_weights)
    std_adjustment = 0.8
    mask_indices = [0]*len(prediction_tokens)

    # We use to counter to avoid infinite loops in the event of an error
    counter = 0
    while counter < 20:
        
        mask_indices = []
        for index, weight in enumerate(prediction_weights):
            if weight > standard_deviation*std_adjustment:
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
    

    print()
    return prediction_tokens, mask_indices


def unmask_string(
        masked_string: List[str],
        mask_indices: List[int],
        desired_class: str,
        classification_model: str,
        mlm_model: str,
        top_k: int=2,
        all_accepted_unmasks: List = None
        ):
    
    # If at the first level of recursion, initialize the list that
    # we will append to and eventually return
    if not all_accepted_unmasks:
        all_accepted_unmasks = []

    # Replace [MASK] with UNK for all but the leftmost mask. We will
    # slowly replace each 
    for mask_n in mask_indices[1:]:
        masked_string[mask_n] = "UNK "

    print(f"in function mask indices: {mask_indices}")

    leftmost_mask = mask_indices[0]
    masked_string[leftmost_mask] = "[MASK] "

    
    # Unmask leftmost mask
    unmask_request = {
        "inputs": "".join(masked_string)
    }
    unmask_response = invoke_endpoint(
        request=unmask_request,
        endpoint_name=mlm_model,
    )

    print(unmask_response, type(unmask_response))
    
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
            max_class_index = unmasked_string_class_prediction.index(max(unmasked_string_class_prediction))
            accepted_unmasks.append(unmasked_strings[max_class_index])

            unmasked_string_class_prediction.pop(max_class_index)
            unmasked_strings.pop(max_class_index)

    if len(mask_indices) > 1:
        print("len of mask indices greater than 1: ", mask_indices)
        for accepted_unmasked_str_n in accepted_unmasks:
            all_accepted_unmasks = unmask_string(
                accepted_unmasked_str_n,
                mask_indices[1:],
                desired_class,
                classification_model,
                mlm_model,
                top_k,
                )
            return all_accepted_unmasks
            
    else:
        for unmasked_str_n in accepted_unmasks:
            all_accepted_unmasks.append("".join(unmasked_str_n))
        return all_accepted_unmasks


def handler(event, context):

    input_text = event["input_text"]

    # The classification model must be able to provide explanations
    classification_model = event["classification_model"]

    # The mlm should be finetuned on the desired class for best results
    mlm_model = event["mlm_model"]

    desired_class = event["desired_class"]

    # Generate explanation
    request = {"data": input_text, "explain": True}
    explanation = invoke_endpoint(request, endpoint_name=classification_model)

    # Mask tokens that contribute most to the currently predicted class
    masked_string, mask_indices = mask_explanation(
        explanation
        )

    # Apply mlm to unmask the newly masked tokens
    unmasked_strings = unmask_string(
        masked_string=masked_string,
        mask_indices = mask_indices,
        desired_class = desired_class,
        classification_model=classification_model,
        mlm_model=mlm_model,
        )
    
    return unmasked_strings