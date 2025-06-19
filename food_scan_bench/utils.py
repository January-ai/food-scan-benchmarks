import base64
from pathlib import Path

from .prompts import PROMPT_VARIANTS

MODEL_COSTS = {
    "january/food-vision-v1": {
        "input": 0.0,  # Not used, cost is per-image
        "output": 0.0,
        "display_name": "January AI",
    },
    "gpt-4.1": {
        "input": 2.00,
        "output": 8.00,
        "display_name": "gpt-4.1",
    },
    "gpt-4o": {
        "input": 2.50,
        "output": 10.00,
        "display_name": "gpt-4o",
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
        "display_name": "gpt-4o-mini",
    },
    "gemini/gemini-2.5-flash-preview-05-20": {
        "input": 0.15,
        "output": 0.60,
        "display_name": "gemini-2.5-flash",
    },
    "gemini/gemini-2.5-pro-preview-06-05": {
        "input": 1.25,
        "output": 10.00,
        "display_name": "gemini-2.5-pro",
    },
}


def img2b64(path: Path) -> str:
    """
    Converts an image file to a base64 encoded string for API calls.
    
    Args:
        path: Path to the image file
        
    Returns:
        Base64 encoded string with data URI prefix
    """
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/jpeg;base64,{encoded}"


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the cost for a model based on token usage.
    
    Args:
        model_name: Name of the model
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Total cost in USD
    """
    if model_name not in MODEL_COSTS:
        return 0.0

    costs = MODEL_COSTS[model_name]
    input_cost = (input_tokens / 1_000_000) * costs["input"]
    output_cost = (output_tokens / 1_000_000) * costs["output"]
    return round(input_cost + output_cost, 6)


def get_display_name(model_name: str) -> str:
    """
    Return the user-friendly model label (falls back to raw id).
    
    Args:
        model_name: Model identifier
        
    Returns:
        Display name for the model
    """
    return MODEL_COSTS.get(model_name, {}).get("display_name", model_name)


def pretty_label(full_model_name: str) -> str:
    """
    Generate a pretty label for a model including prompt variant suffix.
    
    Args:
        full_model_name: Full model name potentially including variant suffix
        
    Returns:
        Pretty formatted label
    """
    if full_model_name == "january/food-vision-v1":
        return get_display_name(full_model_name)

    for variant, meta in PROMPT_VARIANTS.items():
        postfix = f"_{variant}"
        if full_model_name.endswith(postfix):
            base = full_model_name[: -len(postfix)]
            return f"{get_display_name(base)}_{meta['suffix']}"

    if "_" in full_model_name:
        base, variant = full_model_name.rsplit("_", 1)
        suffix = PROMPT_VARIANTS.get(variant, {}).get(
            "suffix", variant[0].lower()
        )
        return f"{get_display_name(base)}_{suffix}"

    return get_display_name(full_model_name)