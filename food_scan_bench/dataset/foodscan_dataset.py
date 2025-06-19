import ast
import tarfile
from pathlib import Path

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.client import Config


class FoodScanDataset:
    """Handles downloading, caching, and loading the food dataset."""

    _S3_BUCKET = "january-food-image-dataset-public"
    _S3_KEY = "food-scan-benchmark-dataset.tar.gz"

    def __init__(self, root: Path):
        """
        Initialize the dataset.

        Args:
            root: Root directory for dataset storage
        """
        self.root = root.expanduser()
        self.img_dir = self.root / "food-scan-benchmark-dataset" / "fsb_images"
        self.csv_path = (
            self.root / "food-scan-benchmark-dataset" / "food_scan_bench_v1.csv"
        )

        if not self.csv_path.exists():
            self._download_and_extract()

        self.df = pd.read_csv(self.csv_path)

    def _download_and_extract(self):
        """Download and extract dataset from S3 if not present."""
        print(f"Dataset not found in {self.root}. Downloading from S3...")
        self.root.mkdir(parents=True, exist_ok=True)
        local_archive = self.root / "fsb.tar.gz"

        s3 = boto3.client(
            "s3",
            config=Config(signature_version=UNSIGNED),
        )
        with open(local_archive, "wb") as f:
            s3.download_fileobj(self._S3_BUCKET, self._S3_KEY, f)

        print("Download complete. Extracting...")
        with tarfile.open(local_archive) as tar:
            tar.extractall(path=self.root)
        local_archive.unlink()
        print("Extraction complete.")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int):
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing image_id, image_path, and ground truth data
        """
        row = self.df.iloc[idx]
        img_path = self.img_dir / row.image_filename

        try:
            ingredients = ast.literal_eval(row.ingredients_list)
        except (ValueError, SyntaxError):
            ingredients = []

        return {
            "image_id": row.image_id,
            "image_path": img_path,
            "gt": {
                "meal_name": row.meal_name,
                "ingredients": ingredients,
                "macros": {
                    "calories": row.total_calories,
                    "carbs": row.total_carbs,
                    "protein": row.total_protein,
                    "fat": row.total_fat,
                },
            },
        }
