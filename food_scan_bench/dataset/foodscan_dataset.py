from __future__ import annotations

import ast
import tarfile
from pathlib import Path

import boto3
from botocore.config import Config
import pandas as pd

_S3 = "january-food-image-dataset-public"
_KEY = "photolog/food-scan-benchmark-dataset.tar.gz"


class FoodScanDataset:
    def __init__(self, root: Path):
        """
        Parameters
        ----------
        root : Path
            Local directory where the archive will be cached /
            extracted, e.g. ~/.cache/food_scan_bench
        """
        self.root = root.expanduser()
        self.img_dir = self.root / "photolog" / "food-scan-benchmark-dataset" / "fsb_images"
        self.csv_path = self.root / "photolog" / "food-scan-benchmark-dataset" / "food_scan_bench_v1.csv"

        if not self.csv_path.exists():
            self._download_and_extract()

        self.df = pd.read_csv(self.csv_path)

    def _download_and_extract(self):
        print("Dataset missing → downloading from public S3 …")
        ensure = self.root
        ensure.mkdir(parents=True, exist_ok=True)

        local_archive = self.root / "fsb.tar.gz"
        s3 = boto3.client("s3", config=Config(signature_version="unsigned"))
        with open(local_archive, "wb") as f:
            s3.download_fileobj(_S3, _KEY, f)

        with tarfile.open(local_archive) as tar:
            tar.extractall(path=self.root)
        local_archive.unlink()

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row.image_filename

        try:
            ingredients = ast.literal_eval(row.ingredients_list)
        except (ValueError, SyntaxError):
            raise ValueError(f"Invalid ingredients_list format in row {idx}: {row.ingredients_list}")
        return {
            "image_id":   row.image_id,
            "image_path": img_path,
            "gt": {
                "meal_name": row.meal_name,
                "ingredients": ingredients,
                "macros": {
                    "calories": row.total_calories,
                    "carbs":    row.total_carbs,
                    "protein":  row.total_protein,
                    "fat":      row.total_fat,
                },
            },
        }