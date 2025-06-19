PROMPT_VARIANTS = {
    "detailed": {
        "suffix": "d",
        "prompt": "You are an expert nutritionist with 20 years of experience. Analyze this food image very carefully and provide the most accurate breakdown possible. Consider portion sizes, cooking methods, and hidden ingredients.",
    },
    "step_by_step": {
        "suffix": "s",
        "prompt": "You are an expert nutritionist. Please analyze this image step by step: 1) First identify all visible food items, 2) Estimate portion sizes, 3) Calculate nutritional content for each item, 4) Sum the totals. Be precise and systematic.",
    },
    "conservative": {
        "suffix": "c",
        "prompt": "You are a conservative nutritionist who prefers to underestimate rather than overestimate. Analyze this food image and provide a realistic, slightly conservative nutritional breakdown.",
    },
    "confident": {
        "suffix": "f",
        "prompt": "You are a highly confident nutritionist. Analyze this food image and provide your best estimate of the nutritional content. Trust your expertise.",
    },
}
