from __future__ import annotations

import numpy as np


def ingredients_f1(gt: list[dict], pred: list[dict]) -> float:
    """
    F1 on ingredient names (case-insensitive).
    """
    g = {x["name"].lower() for x in gt}
    p = {x["name"].lower() for x in pred}
    tp = len(g & p)
    if tp == 0:
        return 0.0
    precision = tp / len(p)
    recall    = tp / len(g)
    return 2 * precision * recall / (precision + recall)


def macro_mae(gt_mac: dict, pred_mac: dict) -> float:
    """
    Mean-absolute-error over the four macros.
    """
    keys = ["calories", "carbs", "protein", "fat"]
    return float(np.mean([abs(gt_mac[k] - pred_mac.get(k, 0)) for k in keys]))