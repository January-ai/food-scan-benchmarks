import json
from pathlib import Path
from typing import Optional, Tuple

import litellm
from litellm.exceptions import APIError

from ..prompts import PROMPT_VARIANTS
from ..schema import FoodAnalysis
from ..utils import calculate_cost, img2b64


class LiteModel:
    """A robust wrapper around any LiteLLM-supported vision model with prompt engineering options."""

    def __init__(self, model_name: str, prompt_variant="detailed", **litellm_kwargs):
        """
        Initialize the LiteModel.
        
        Args:
            model_name: Name of the model to use (e.g., "gpt-4o")
            prompt_variant: Prompt variant to use (detailed, step_by_step, conservative, confident)
            **litellm_kwargs: Additional kwargs to pass to litellm
        """
        self.model_name = model_name
        self.prompt_variant = prompt_variant
        self.kwargs = litellm_kwargs

        cfg = PROMPT_VARIANTS.get(prompt_variant, PROMPT_VARIANTS["detailed"])
        self.system_prompt = cfg["prompt"]
        self.prompt_suffix = cfg["suffix"]

    async def analyse(self, img_path: Path) -> Tuple[Optional[dict], Optional[str]]:
        """
        Analyzes an image and returns a structured dict with cost info, or None and an error message on failure.

        Args:
            img_path: Path to the image file

        Returns:
            Tuple[Optional[dict], Optional[str]]: A tuple of (result, error_message).
                                                  On success, result is a dict and error_message is None.
                                                  On failure, result is None and error_message is a string.
        """
        b64_img = img2b64(img_path)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this food image and provide a detailed nutritional breakdown. Include: meal name, all ingredients with quantities/units, and complete macro information (calories, carbs, protein, fat) for each ingredient and the total meal.",
                    },
                    {"type": "image_url", "image_url": {"url": b64_img}},
                ],
            },
        ]

        try:
            resp = await litellm.acompletion(
                model=self.model_name,
                messages=messages,
                response_format=FoodAnalysis,
                temperature=0.0,
                **self.kwargs,
            )
            raw = resp.choices[0].message.content.strip()
            data = json.loads(raw)

            usage = resp.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            cost = calculate_cost(self.model_name, input_tokens, output_tokens)

            result = FoodAnalysis(**data).model_dump()
            result["cost_usd"] = cost
            result["prompt_variant"] = self.prompt_variant

            return result, None

        except APIError as e:
            error_msg = f"API Error: {e}"
            print(f"{error_msg} for {img_path.name}")
            return None, error_msg
        except Exception as e:
            error_msg = f"Unexpected Error: {e}"
            print(f"{error_msg} for {img_path.name}")
            return None, error_msg

