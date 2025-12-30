"""
Helper script to generate config.js for the static website.
This is called during CDK deployment to inject runtime configuration.
"""
import json


def generate_config_js(api_endpoint: str, classification_models: dict, endpoint_names: dict) -> str:
    """
    Generate config.js content with deployment configuration.

    Args:
        api_endpoint: API Gateway endpoint URL
        classification_models: Model configuration dict
        endpoint_names: Mapping of environment variable names to endpoint names

    Returns:
        JavaScript content as string
    """
    config = {
        "API_ENDPOINT": api_endpoint,
        "CLASSIFICATION_MODELS": classification_models,
        "ENDPOINT_NAMES": endpoint_names
    }

    js_content = f"""// Configuration injected during CDK deployment
window.CONFIG = {json.dumps(config, indent=2)};
"""
    return js_content
