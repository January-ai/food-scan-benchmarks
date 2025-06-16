import argparse
import json
from pathlib import Path

import pandas as pd


def analyze(results_dir: Path):
    all_results = []
    for result_file in results_dir.glob("*.jsonl"):
        model_name = result_file.stem
        with open(result_file, "r") as f:
            for line in f:
                data = json.loads(line)
                data["model"] = model_name
                all_results.append(data)

    if not all_results:
        print(f"No result files found in {results_dir}")
        return

    df = pd.DataFrame(all_results)

    df['mae_mac'] = pd.to_numeric(df['mae_mac'], errors='coerce')

    summary = df.groupby('model').agg(
        mean_f1=('f1_ing', 'mean'),
        std_f1=('f1_ing', 'std'),
        mean_mae=('mae_mac', 'mean'),
        std_mae=('mae_mac', 'std'),
        failure_rate=('mae_mac', lambda x: x.isna().mean())
    ).reset_index()

    print("--- Benchmark Summary ---")
    print(summary.to_string(index=False, float_format="%.3f"))
    
    model_to_check = summary['model'][0]
    print(f"\n--- Worst F1 Scores for {model_to_check} ---")
    worst_f1 = df[df['model'] == model_to_check].sort_values('f1_ing', ascending=True).head(5)
    print(worst_f1[['image_id', 'f1_ing']].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze benchmark results.")
    parser.add_argument("--dir", default="results", type=Path, help="Directory containing result .jsonl files")
    args = parser.parse_args()
    analyze(args.dir)