from typing import List

from pydantic import BaseModel, Field


class Ingredient(BaseModel):
    """Schema for individual food ingredients."""

    name: str = Field(description="Name of the ingredient")
    quantity: float = Field(description="Numerical quantity of the ingredient")
    unit: str = Field(description="Unit of measurement")
    calories: float = Field(description="Estimated calories for this ingredient")
    carbs: float = Field(
        description="Estimated grams of carbohydrates for this ingredient"
    )
    protein: float = Field(description="Estimated grams of protein for this ingredient")
    fat: float = Field(description="Estimated grams of fat for this ingredient")


class TotalMacros(BaseModel):
    """Schema for total macronutrient information."""

    calories: float = Field(description="Total estimated calories for the entire meal")
    carbs: float = Field(
        description="Total estimated grams of carbohydrates for the entire meal"
    )
    protein: float = Field(
        description="Total estimated grams of protein for the entire meal"
    )
    fat: float = Field(description="Total estimated grams of fat for the entire meal")


class FoodAnalysis(BaseModel):
    """Complete food analysis schema including meal name, ingredients, and total macros."""

    meal_name: str = Field(description="A descriptive name for the meal")
    ingredients: List[Ingredient] = Field(
        description="A list of all identified ingredients and their nutritional information"
    )
    total_macros: TotalMacros = Field(
        description="The sum of macros for all ingredients"
    )
