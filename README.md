# Food Scan Benchmarks

A comprehensive toolkit for evaluating food image analysis models across standardized datasets and metrics.

## Features

- **Multi-model support**: Evaluate various vision models through LiteLLM integration
- **Standardized datasets**: Download and use curated food image datasets
- **Comprehensive metrics**: Accuracy, top-k accuracy, and extensible custom metrics
- **Batch processing**: Efficient evaluation across multiple models and datasets
- **Visualization tools**: Generate comparison plots and analysis tables
- **Easy configuration**: JSON-based configuration for reproducible benchmarks

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/food-scan-benchmarks.git
cd food-scan-benchmarks

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

## Quick Start

### 1. Basic Model Evaluation

```python
from pathlib import Path
from food_scan_benchmarks.models import LiteLLMWrapper
from food_scan_benchmarks.evaluate import evaluate_model

# Initialize a model
model = LiteLLMWrapper(
    name="gpt-4-vision",
    model_name="gpt-4-vision-preview",
    api_key="your-openai-api-key"
)

# Evaluate on a dataset
results = evaluate_model(
    model=model,
    dataset_path=Path("data/food-101-subset"),
    output_path=Path("results/gpt-4-vision")
)

print(f"Accuracy: {results['accuracy']:.3f}")
```

### 2. Configuration-based Benchmark

Create a configuration file (`benchmark_config.json`):

```json
{
  "datasets": [
    {
      "name": "food-101-subset",
      "bucket": "food-benchmark-datasets"
    }
  ],
  "models": [
    {
      "name": "gpt-4-vision",
      "type": "litellm",
      "model_name": "gpt-4-vision-preview",
      "api_key": "your-openai-key"
    },
    {
      "name": "claude-3-vision",
      "type": "litellm", 
      "model_name": "claude-3-sonnet-20240229",
      "api_key": "your-anthropic-key"
    }
  ]
}
```

Run the benchmark:

```bash
# Download datasets and run evaluation
python scripts/run_benchmark.py --config benchmark_config.json --output-dir results

# Generate analysis plots
python scripts/analyze_results.py --results results/benchmark_results.json --output-dir analysis
```

### 3. Jupyter Notebook Tutorial

Check out the interactive tutorial:

```bash
jupyter notebook notebooks/tutorial.ipynb
```

## Project Structure

```
food-scan-benchmarks/
├── README.md
├── LICENSE
├── pyproject.toml          # uv/pip configuration
├── .gitignore
├── .python-version         # Python version specification
├── food-scan-benchmarks/   # Main package
│   ├── __init__.py
│   ├── download.py         # S3 dataset downloading
│   ├── evaluate.py         # Core evaluation logic
│   ├── metrics.py          # Evaluation metrics
│   └── models.py           # Model wrappers (LiteLLM)
├── scripts/
│   ├── run_benchmark.py    # CLI entry point
│   └── analyze_results.py  # Generate plots/tables
├── notebooks/
│   └── tutorial.ipynb      # Getting started guide
├── results/                # Evaluation results
│   └── .gitkeep
└── docs/
    └── README.md           # Detailed documentation
```

## Supported Models

Through LiteLLM integration, the toolkit supports:

- **OpenAI**: GPT-4 Vision, GPT-4o
- **Anthropic**: Claude 3 (Opus, Sonnet, Haiku)
- **Google**: Gemini Pro Vision
- **And many more**: Any model supported by LiteLLM

## Evaluation Metrics

- **Accuracy**: Standard classification accuracy
- **Top-k Accuracy**: Accuracy considering top-k predictions
- **Custom Metrics**: Extensible framework for additional metrics

## Dataset Support

- S3-based dataset downloading
- Configurable dataset sources
- Support for various food image datasets
- Extensible for custom datasets

## Advanced Usage

### Adding Custom Models

```python
from food_scan_benchmarks.models import ModelWrapper

class CustomModelWrapper(ModelWrapper):
    def predict(self, image_path: Path, prompt: str) -> str:
        # Your custom model implementation
        pass
    
    def predict_batch(self, image_paths: List[Path], prompts: List[str]) -> List[str]:
        # Batch prediction implementation
        pass
```

### Adding Custom Metrics

```python
from food_scan_benchmarks.metrics import calculate_metrics

def custom_metric(predictions: List[str], ground_truth: List[str]) -> float:
    # Your custom metric implementation
    pass

# Register in calculate_metrics function
```

## CLI Reference

### run_benchmark.py

```bash
python scripts/run_benchmark.py [OPTIONS]

Options:
  --config PATH          Configuration file path [required]
  --output-dir PATH      Output directory [default: results]
  --verbose             Enable verbose logging
  --download-only       Only download datasets, skip evaluation
```

### analyze_results.py  

```bash
python scripts/analyze_results.py [OPTIONS]

Options:
  --results PATH        Benchmark results JSON file [required]
  --output-dir PATH     Analysis output directory [default: analysis]
  --verbose            Enable verbose logging
```

## Requirements

- Python 3.8+
- LiteLLM for model access
- Boto3 for S3 dataset downloading
- Matplotlib/Seaborn for visualization
- Pandas for data analysis

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{food_scan_benchmarks,
  title={Food Scan Benchmarks: A Toolkit for Evaluating Food Image Analysis Models},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/food-scan-benchmarks}
}
```
