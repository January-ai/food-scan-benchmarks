import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

from ..schema import FoodAnalysis, Ingredient, TotalMacros
from ..utils import img2b64


class JanuaryAIModel:
    """A wrapper for the proprietary January AI food vision API."""

    COST_PER_IMAGE = 0.01

    def __init__(self):
        """Initialize the January AI model with API credentials from environment."""
        self.endpoint = os.getenv("JANUARY_API_ENDPOINT")
        self.uuid = os.getenv("JANUARY_API_UUID")
        self.token = os.getenv("JANUARY_API_TOKEN")

        if not all([self.endpoint, self.uuid, self.token]):
            raise ValueError(
                "January AI API credentials not found in environment variables."
            )

        self.headers = {
            "Content-Type": "application/json",
            "UUID": self.uuid,
            "x-jan-e2e-tests-token": self.token,
        }
        self.client = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.client = httpx.AsyncClient()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()
            self.client = None

    async def _ensure_client(self):
        """Ensure client is initialized."""
        if self.client is None:
            self.client = httpx.AsyncClient()

    async def close(self):
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None

    def _calculate_ingredient_macros(self, ingredient: dict) -> Dict[str, float]:
        """Re-implements legacy macro logic for a single ingredient."""
        (cal, fat, carbs, prot, mass, fiber) = (0, 0, 0, 0, 0, 0)
        if "servings" in ingredient and ingredient["servings"]:
            s = ingredient["servings"][0]
            q, sel, scale = s["quantity"], s["selected_quantity"], s["scaling_factor"]
            w = s.get("weight_grams", 0)
            cal = ingredient["energy"] * sel * scale / q
            fat = ingredient["fat"] * sel * scale / q
            carbs = ingredient["carbs"] * sel * scale / q
            prot = ingredient["protein"] * sel * scale / q
            fib = ingredient.get("fiber", 0) or 0
            fiber = fib * sel * scale / q
            mass = w * sel / q
        return dict(
            calories=cal, fat=fat, carbs=carbs, protein=prot, fiber=fiber, mass=mass
        )

    def _parse_server_response(self, js: Dict) -> FoodAnalysis:
        """Parses the raw JSON from the API into the standard FoodAnalysis schema."""
        ing_objs: List[Ingredient] = []
        for ing in js.get("ingredients", []):
            m = self._calculate_ingredient_macros(ing)
            ing_objs.append(
                Ingredient(
                    name=ing.get("name", "unknown ingredient"),
                    quantity=ing.get("servings", [{}])[0].get("selected_quantity", 0),
                    unit=ing.get("servings", [{}])[0].get("unit", "g"),
                    calories=m["calories"],
                    carbs=m["carbs"],
                    protein=m["protein"],
                    fat=m["fat"],
                )
            )

        tot = TotalMacros(
            calories=sum(i.calories for i in ing_objs),
            carbs=sum(i.carbs for i in ing_objs),
            protein=sum(i.protein for i in ing_objs),
            fat=sum(i.fat for i in ing_objs),
        )

        return FoodAnalysis(
            meal_name=js.get("foodName", "unknown meal"),
            ingredients=ing_objs,
            total_macros=tot,
        )

    async def analyse(self, img_path: Path) -> Tuple[Optional[dict], Optional[str]]:
        """
        Analyzes an image using the January AI API.
        
        Args:
            img_path: Path to the image file
            
        Returns:
            A structured dict, or None and an error message on failure.
        """
        await self._ensure_client()
        payload = {"photoUrl": img2b64(img_path)}
        try:
            if self.client is None:
                raise RuntimeError("HTTP client not initialized")
            if self.endpoint is None:
                raise RuntimeError("API endpoint not configured")
            r = await self.client.post(
                self.endpoint, headers=self.headers, json=payload, timeout=30.0
            )
            r.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            js = r.json()
            parsed_response = self._parse_server_response(js)

            result = parsed_response.model_dump()
            result["cost_usd"] = self.COST_PER_IMAGE
            result["prompt_variant"] = "default"

            return result, None

        except httpx.HTTPStatusError as e:
            error_msg = f"API Error: {e.response.status_code} - {e.response.text}"
            print(f"{error_msg} for {img_path.name}")
            return None, error_msg
        except Exception as e:
            error_msg = f"Unexpected Error: {e}"
            print(f"{error_msg} for {img_path.name}")
            return None, error_msg