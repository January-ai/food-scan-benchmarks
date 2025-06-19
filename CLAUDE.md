# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a food image recognition benchmark system that evaluates various multimodal models (including January AI's proprietary food vision API and popular LLMs like GPT-4, Gemini) on their ability to analyze food images and extract nutritional information.

## Core Architecture

- **Main benchmark runner**: `food_scan_bench/run_benchmark.py` - Entry point for running evaluations
- **Model wrappers**: Two main model types in `food_scan_bench/models/`:
  - `january.py` - January AI's proprietary food vision API wrapper
  - `llm.py` - LiteLLM wrapper for various vision models (GPT-4, Gemini, etc.)
- **Dataset handling**: `food_scan_bench/dataset/foodscan_dataset.py` - Downloads and manages food image dataset from S3
- **Evaluation pipeline**: `food_scan_bench/evaluate.py` - Orchestrates model evaluation with async processing
- **Metrics calculation**: `food_scan_bench/metrics.py` - Computes similarity scores and performance metrics
- **Results analysis**: `food_scan_bench/analyze_results.py` - Generates visualizations and summary statistics

## Key Dependencies

- Uses `uv` for dependency management (pyproject.toml)
- Async processing with `httpx` and `asyncio` for concurrent model evaluations
- `litellm` for unified API access to various LLM providers
- `pandas` for data manipulation and CSV output
- `plotly` for interactive visualizations
- `boto3` for S3 dataset downloads

## Common Commands

### Running the benchmark
```bash
# Run with default models and settings
python -m food_scan_bench.run_benchmark

# Run specific models with custom settings
python -m food_scan_bench.run_benchmark --models gpt-4o january/food-vision-v1 --max-items 50 --visualize

# Run without visualizations
python -m food_scan_bench.run_benchmark --no-visualize

# Export detailed reports
python -m food_scan_bench.run_benchmark --export-report
```

### Environment Setup
- Requires Python 3.12+
- Install dependencies: `uv sync`
- Set up environment variables for API access:
  - `JANUARY_API_ENDPOINT`, `JANUARY_API_UUID`, `JANUARY_API_TOKEN` for January AI
  - Standard LLM API keys for other models (OpenAI, Google, etc.)

## Data Flow

1. Dataset is automatically downloaded from S3 on first run
2. Images are processed concurrently through selected models
3. Results are cached and saved to `results/` directory
4. Metrics are computed using embedding-based similarity matching
5. Visualizations and summary statistics are generated

## Output Structure

- Raw results: `results/benchmark_results.csv`
- Summary statistics: `results/summary_statistics.txt`
- Interactive dashboards: `results/performance_dashboard.html`
- Comparative analysis: `results/win_loss_analysis.html`

## Model Integration

To add new models, create a wrapper class implementing the `analyse(img_path: Path)` async method that returns `(Optional[dict], Optional[str])` tuple. The dict should conform to the `FoodAnalysis` schema defined in `schema.py`.
