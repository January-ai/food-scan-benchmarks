from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from .dataset.foodscan_dataset import FoodScanDataset
from .metrics import ingredients_f1, macro_mae
from .models.llm import LiteModel
from .utils import ensure_dir


def run(model_name: str,
        out_dir: Path = Path("results"),
        cache_dir: Path = Path(".cache/food_scan_bench").expanduser(),
        max_items: int | None = None):

    ds  = FoodScanDataset(cache_dir)
    llm = LiteModel(model_name)

    ensure_dir(out_dir)
    results = []

    loop = tqdm(range(len(ds)), desc=f"Evaluating {model_name}")
    for idx in loop:
        if max_items and idx >= max_items:
            break

        sample  = ds[idx]
        pred    = llm.analyse(sample["image_path"])
        gt      = sample["gt"]

        if pred is None:
            print(f"Skipping {sample['image_id']} due to analysis failure.")
            results.append({
                "image_id": sample["image_id"],
                "f1_ing": 0.0,
                "mae_mac": None,
                "error": "Analysis failed"
            })
            continue

        f1  = ingredients_f1(gt["ingredients"], pred.get("ingredients", []))
        mae = macro_mae(gt["macros"], pred.get("total_macros", {}))

        results.append({
            "image_id": sample["image_id"],
            "f1_ing": f1,
            "mae_mac": mae,
        })

        loop.set_postfix(f1=f"{f1:0.2f}", mae=f"{mae:0.1f}")

    path = out_dir / f"{model_name.replace('/','_')}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in results))
    return results