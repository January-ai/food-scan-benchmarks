# Food-Scan Benchmarks

Evaluate vision-language models on real-world food recognition and nutritional analysis tasks.

---

## 1. Overview

Food-Scan Benchmarks run a curated set of food images through several vision APIs and score the output on three axes:

- Meal name semantic similarity
- Ingredient precision / recall
- Macro-nutrient (kcal, carbs, protein, fat) accuracy

The three scores are combined into a single **overall score (0-100)** so models can be compared at a glance.

![Overall score](assets/overall.png)

## 2. Quick start

Requirements

- Python 3.12+
- `uv` (or `pipx install uv`)

```bash
# 1 . clone
git clone https://github.com/January-ai/food-scan-benchmarks.git
cd food-scan-benchmarks

# 2 . install
uv sync

# 3 . set credentials
cp .env.example .env  # then edit with your API keys

# 4 . run benchmark
python -m food_scan_bench.run_benchmark --models january/food-vision-v1 gpt-4o
```

The first run downloads the dataset from S3 and caches it locally.

## 3. CLI usage

```bash
python -m food_scan_bench.run_benchmark [OPTIONS]

Options
  --models TEXT...          Models to evaluate (default: all)
  --max-items INTEGER       Number of images to process (default: 20)
  --visualize / --no-visualize  Create Plotly HTML dashboards (on by default)
  --baseline-model TEXT     Model used for win/loss comparison plots
  --export-report           Write detailed CSV report to disk
  --help                    Show full help
```

Example: cost-effectiveness analysis for small, cheap models

```bash
python -m food_scan_bench.run_benchmark \
    --models gpt-4o-mini gemini/gemini-2.5-flash-preview-05-20 \
    --max-items 50
```

## 4. Supported models

| Provider   | Identifier                                        |
| ---------- | ------------------------------------------------- |
| January AI | `january/food-vision-v1`                          |
| OpenAI     | `gpt-4o`, `gpt-4o-mini`                           |
| Google     | `gemini/gemini-2.5-flash-preview-05-20`           |
|            | `gemini/gemini-2.5-pro-preview-06-05`             |
| Other      | Any LiteLLM-compatible model that supports images |

Add your own model in a few lines—see below.

## 5. Project layout

```text
food_scan_bench/
├── run_benchmark.py     # CLI entry-point
├── models/              # API wrappers (January, LiteLLM, …)
├── evaluate.py          # Async evaluation pipeline
├── metrics.py           # Scoring functions
├── analyze_results.py   # Plots & stats
└── dataset/             # Dataset download / cache helpers
```

## 6. Adding a new model

1. Create a wrapper in `food_scan_bench/models/` that implements `analyse(self, image: PIL.Image) -> FoodAnalysis`.
2. Register the model id in `food_scan_bench/run_benchmark.py`.
3. `FoodAnalysis` must include the meal name, ingredients list, and macro nutrients.

## 7. Contributing & license

Bug reports, feature requests and PRs are welcome! Please:

1. Fork → feature branch → write tests.
2. Ensure `pre-commit run --all-files` pass (see `.pre-commit-config.yaml`).
3. Open a pull request.

Licensed under the MIT License (see [LICENSE](LICENSE)).

---

© January AI. For commercial or research use of the January AI Food Vision model, please contact [January AI](https://january.ai).
