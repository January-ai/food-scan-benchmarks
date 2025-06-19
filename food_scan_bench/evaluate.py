import asyncio
import time
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from tqdm.auto import tqdm

from .dataset import FoodScanDataset
from .metrics import Metrics
from .models import JanuaryAIModel, LiteModel
from .prompts import PROMPT_VARIANTS


async def _process_sample(
    idx: int,
    ds: FoodScanDataset,
    llm: Union[LiteModel, JanuaryAIModel],
    model_name: str,
    use_embeddings: bool = True,
) -> dict:
    """
    Process a single sample through a model and compute metrics.

    Args:
        idx: Sample index in dataset
        ds: FoodScanDataset instance
        llm: Model instance to use
        model_name: Name identifier for results
        use_embeddings: Whether to use embedding-based metrics

    Returns:
        Dictionary containing all metrics and results
    """
    start_time = time.time()
    sample = ds[idx]
    pred, error_msg = await llm.analyse(sample["image_path"])
    end_time = time.time()
    gt = sample["gt"]

    item = {
        "image_id": sample["image_id"],
        "model": model_name,
        "response_time_seconds": round(end_time - start_time, 2),
    }

    if pred is None:
        item.update(
            {
                "meal_name_similarity": 0.0,
                "semantic_precision_ing": 0.0,
                "semantic_match": 0.0,
                "semantic_f1_ing": 0.0,
                "semantic_match_embeddings": 0.0,
                "ingredient_count_acc": 0.0,
                "wmape_mac": None,
                "error": error_msg or "failed",
                "cost_usd": 0.0,
                "calories_pct_error": None,
                "carbs_pct_error": None,
                "protein_pct_error": None,
                "fat_pct_error": None,
                "match_details": None,
            }
        )
    else:
        gt_ingredients = gt["ingredients"]
        pred_ingredients = pred.get("ingredients", [])
        gt_macros = gt["macros"]
        pred_macros = pred.get("total_macros", {})

        meal_name_sim = await Metrics.meal_name_similarity(
            gt["meal_name"], pred.get("meal_name", "")
        )

        semantic_match_embeddings, match_details = 0.0, None
        semantic_f1 = 0.0
        semantic_precision = 0.0
        if use_embeddings:
            try:
                (
                    semantic_match_embeddings,
                    match_details,
                ) = await Metrics.semantic_ingredient_match_embeddings(
                    gt_ingredients, pred_ingredients
                )
                semantic_f1 = await Metrics.semantic_f1_score(
                    gt_ingredients, pred_ingredients
                )
                semantic_precision = await Metrics.semantic_precision_score(
                    gt_ingredients, pred_ingredients
                )
            except Exception as e:
                print(
                    f"Error in embedding similarity for image {sample['image_id']}: {e}"
                )
                semantic_match_embeddings = Metrics.semantic_ingredient_match(
                    gt_ingredients, pred_ingredients
                )
                semantic_f1 = 0.0
                semantic_precision = 0.0

        pct_errors = Metrics.macro_percentage_error(gt_macros, pred_macros)

        item.update(
            {
                "meal_name": pred.get("meal_name", ""),
                "gt_meal_name": gt["meal_name"],
                "meal_name_similarity": meal_name_sim,
                "semantic_precision_ing": semantic_precision,
                "semantic_match": Metrics.semantic_ingredient_match(
                    gt_ingredients, pred_ingredients
                ),
                "semantic_f1_ing": semantic_f1,
                "semantic_match_embeddings": semantic_match_embeddings,
                "ingredient_count_acc": Metrics.ingredient_count_accuracy(
                    gt_ingredients, pred_ingredients
                ),
                "wmape_mac": Metrics.macro_wMAPE(gt_macros, pred_macros),
                "error": None,
                "cost_usd": pred.get("cost_usd", 0.0),
                "calories_pct_error": pct_errors.get("calories"),
                "carbs_pct_error": pct_errors.get("carbs"),
                "protein_pct_error": pct_errors.get("protein"),
                "fat_pct_error": pct_errors.get("fat"),
            }
        )

    return item


async def run_evaluation(
    models: Union[str, List[str]],
    cache_dir: Path,
    max_items: Optional[int] = None,
    max_concurrent: int = 5,
    use_embeddings: bool = True,
) -> pd.DataFrame:
    """
    Run evaluation with multiple models, including custom ones.

    Args:
        models: Single model name or list of model names
        cache_dir: Directory for caching dataset
        max_items: Maximum number of items to evaluate (None for all)
        max_concurrent: Maximum concurrent API requests
        use_embeddings: Whether to use embedding-based metrics

    Returns:
        DataFrame containing all evaluation results
    """
    models_to_run = [models] if isinstance(models, str) else models
    all_prompt_variants = list(PROMPT_VARIANTS.keys())

    ds = FoodScanDataset(cache_dir)
    n = min(max_items, len(ds)) if max_items else len(ds)
    tasks_to_run = []
    for model_name in models_to_run:
        if model_name == "january/food-vision-v1":
            tasks_to_run.append({"model_name": model_name, "prompt_variant": "default"})
        else:
            for variant in all_prompt_variants:
                tasks_to_run.append(
                    {"model_name": model_name, "prompt_variant": variant}
                )

    sem = asyncio.Semaphore(max_concurrent)

    january_model = None
    if any(t["model_name"] == "january/food-vision-v1" for t in tasks_to_run):
        january_model = JanuaryAIModel()

    async def _worker(task_info: dict, idx: int):
        async with sem:
            model_name = task_info["model_name"]
            prompt_variant = task_info["prompt_variant"]

            if model_name == "january/food-vision-v1":
                llm = january_model
                model_name_for_results = model_name
            else:
                llm = LiteModel(model_name, prompt_variant=prompt_variant)
                model_name_for_results = f"{model_name}_{prompt_variant}"

            if llm is not None:
                return await _process_sample(
                    idx, ds, llm, model_name_for_results, use_embeddings
                )
            return None

    all_jobs = [(i, task) for i in range(n) for task in tasks_to_run]

    pbar = tqdm(total=len(all_jobs), desc="Processing images", dynamic_ncols=True)

    async def _job_runner(job):
        result = await _worker(job[1], job[0])
        pbar.update(1)
        return result

    try:
        results = await asyncio.gather(*[_job_runner(job) for job in all_jobs])
        pbar.close()

        return pd.DataFrame(results)
    finally:
        # Clean up HTTP clients
        if january_model:
            await january_model.close()

        # Clean up OpenAI client
        from .metrics import Metrics

        await Metrics.close_openai_client()
