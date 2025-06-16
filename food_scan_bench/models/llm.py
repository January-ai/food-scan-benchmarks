
from __future__ import annotations

from pathlib import Path
from typing import Any

import litellm
from litellm.exceptions import APIError, ContentPolicyViolationError

from ..prompts import get_prompts
from ..schema import FoodAnalysis
from ..utils import img2b64


class LiteModel:
    def __init__(self, model_name: str, **litellm_kwargs):
        self.model_name = model_name
        self.kwargs = litellm_kwargs

    def analyse(self, img_path: Path) -> dict | None:
        """Returns a python dict with the analysis, or None on failure."""
        b64_img = img2b64(img_path)
        
        try:
            resp = litellm.completion(
                model=self.model_name,
                messages=get_prompts(b64_img),
                response_format=FoodAnalysis, 
                **self.kwargs,
            )
            
            content: Any = resp.choices[0].message.content
            if isinstance(content, FoodAnalysis):
                return content.model_dump()
            elif hasattr(content, 'model_dump'):
                return content.model_dump()
            elif isinstance(content, dict):
                return content
            else:
                return None 
        
        except (APIError, ContentPolicyViolationError) as e:
            print(f"LiteLLM API Error for {img_path.name}: {e}")
            return None
        except Exception as e:
            
            print(f"An unexpected error occurred for {img_path.name}: {e}")
            return None