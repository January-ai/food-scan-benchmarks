def get_prompts(b64_img: str) -> list[dict]:
    """
    Creates the messages payload for a vision model.
    """
    system = {
        "role": "system",
        "content": "You are an expert nutritionist. Analyze the food image and provide a detailed breakdown of its ingredients and nutritional content in the requested JSON format.",
    }

    user = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Identify the food in the image and provide its name, a list of ingredients with their quantities and macros, and the total macros for the meal.",
            },
            {
                "type": "image_url",
                "image_url": {"url": b64_img},
            },
        ],
    }
    return [system, user]