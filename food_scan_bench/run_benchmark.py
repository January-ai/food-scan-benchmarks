import argparse
import asyncio
import warnings
from pathlib import Path

from dotenv import load_dotenv

from .analyze_results import BenchmarkAnalyzer
from .evaluate import run_evaluation

# Default configuration
DEFAULT_CONFIG = {
    "models": [
        "january/food-vision-v1",
        "gpt-4o-mini",
        "gpt-4o",
        "gemini/gemini-2.5-flash-preview-05-20",
        "gemini/gemini-2.5-pro-preview-06-05",
    ],
    "max_items": 20,
    "cache_dir": Path(".cache/food_scan_bench"),
    "max_concurrent_requests": 50,
    "use_embeddings_for_matching": True,
    "report_filename": "benchmark_results.csv",
}


async def main(args):
    """Main benchmark execution function."""
    # Suppress warnings
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", message=".*Unclosed client session.*")
    warnings.filterwarnings("ignore", message=".*Unclosed connector.*")

    # Load environment variables
    load_dotenv()

    # Ensure results directory exists
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Configure models to run
    models = args.models if args.models else DEFAULT_CONFIG["models"]

    print(f"Running Food Scan Benchmark with {len(models)} models...")
    print(f"Processing up to {args.max_items} images...")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Max concurrent requests: {args.max_concurrent}")
    print(f"Results will be saved to: {results_dir}")
    print()

    # Run evaluation
    results_df = await run_evaluation(
        models=models,
        cache_dir=args.cache_dir,
        max_items=args.max_items,
        max_concurrent=args.max_concurrent,
        use_embeddings=args.use_embeddings,
    )

    # Save raw results
    results_file = results_dir / args.report_filename
    results_df.to_csv(results_file, index=False)
    print(f"Raw results saved to: {results_file}")

    # Analyze results
    analyzer = BenchmarkAnalyzer(results_df)

    # Save summary statistics to file instead of printing
    summary_file = results_dir / "summary_statistics.txt"
    with open(summary_file, "w") as f:
        from contextlib import redirect_stdout

        with redirect_stdout(f):
            analyzer.summary_statistics()
    print(f"Summary statistics saved to: {summary_file}")

    # Create visualizations
    if args.visualize:
        dashboard_file = results_dir / "performance_dashboard.html"
        analyzer.create_performance_dashboard(save_path=dashboard_file)
        print(f"Performance dashboard saved to: {dashboard_file}")

        plot_file = results_dir / "overall_score_plot.html"
        analyzer.plot_overall_score(save_path=plot_file)
        print(f"Overall score plot saved to: {plot_file}")

        if args.baseline_model:
            winloss_file = results_dir / f"win_loss_analysis_{args.baseline_model}.html"
            analyzer.create_win_loss_analysis(
                baseline_model_name=args.baseline_model, save_path=winloss_file
            )
        else:
            winloss_file = results_dir / "win_loss_analysis.html"
            analyzer.create_win_loss_analysis(save_path=winloss_file)
        print(f"Win-loss analysis saved to: {winloss_file}")

    # Export detailed report
    if args.export_report:
        detailed_file = results_dir / f"detailed_{args.report_filename}"
        analyzer.export_detailed_report(str(detailed_file))
        print(f"Detailed report saved to: {detailed_file}")

    print(f"\nBenchmark completed! All results saved to {results_dir}/")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Food Scan Benchmark for multimodal food analysis models."
    )

    parser.add_argument(
        "--models",
        nargs="+",
        help="List of models to evaluate (default: all supported models)",
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=DEFAULT_CONFIG["max_items"],
        help=f"Maximum number of images to process (default: {DEFAULT_CONFIG['max_items']})",
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CONFIG["cache_dir"],
        help=f"Directory for dataset caching (default: {DEFAULT_CONFIG['cache_dir']})",
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_CONFIG["max_concurrent_requests"],
        help=f"Maximum concurrent API requests (default: {DEFAULT_CONFIG['max_concurrent_requests']})",
    )

    parser.add_argument(
        "--no-embeddings",
        dest="use_embeddings",
        action="store_false",
        help="Disable embedding-based similarity metrics (uses string matching instead)",
    )

    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="Skip creating visualization dashboards",
    )

    parser.add_argument(
        "--baseline-model",
        type=str,
        help="Model to use as baseline for win-loss analysis",
    )

    parser.add_argument(
        "--export-report",
        action="store_true",
        help="Export detailed results to CSV files",
    )

    parser.add_argument(
        "--report-filename",
        type=str,
        default=DEFAULT_CONFIG["report_filename"],
        help=f"Filename for exported report (default: {DEFAULT_CONFIG['report_filename']})",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
