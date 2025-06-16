import argparse
from pathlib import Path

from food_scan_bench.evaluate import run

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    help="e.g. 'gpt-4o'")
parser.add_argument("--out", default="results", type=Path)
parser.add_argument("--cache", default="~/.cache/food_scan_bench", type=Path)
parser.add_argument("--n", type=int, help="debug subset size")
args = parser.parse_args()

run(args.model, out_dir=args.out, cache_dir=args.cache, max_items=args.n)