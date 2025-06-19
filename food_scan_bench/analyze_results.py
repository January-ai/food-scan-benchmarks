"""Comprehensive analysis and visualization of benchmark results."""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .prompts import PROMPT_VARIANTS
from .utils import get_display_name, pretty_label


class BenchmarkAnalyzer:
    """Comprehensive analysis and visualization of benchmark results."""

    def __init__(self, results_df):
        """
        Initialize the analyzer with benchmark results.

        Args:
            results_df: DataFrame containing benchmark results
        """
        self.df = results_df.copy()

        def get_base_model(model_name):
            if model_name == "january/food-vision-v1":
                return model_name

            for variant in PROMPT_VARIANTS:
                if model_name.endswith(f"_{variant}"):
                    return model_name.rsplit(f"_{variant}", 1)[0]

            return model_name

        self.df["base_model"] = self.df["model"].apply(get_base_model)
        self.df["pretty_model"] = self.df["model"].apply(pretty_label)

        if not self.df.empty:
            self.successful_df = self.df[self.df["error"].isna()].copy()

            if not self.successful_df.empty:
                self.successful_df = self._add_overall_score(self.successful_df)

                best_indices = self.successful_df.groupby(["image_id", "base_model"])[
                    "overall_score"
                ].idxmax()
                self.best_of_df = self.successful_df.loc[best_indices].copy()
                self.best_of_df["model"] = self.best_of_df["base_model"].apply(
                    lambda x: (
                        f"{get_display_name(x)} (Best)"
                        if x != "january/food-vision-v1"
                        else get_display_name(x)
                    )
                )

                numeric_cols = [
                    "meal_name_similarity",
                    "semantic_match_embeddings",
                    "semantic_precision_ing",
                    "semantic_f1_ing",
                    "ingredient_count_acc",
                    "wmape_mac",
                    "cost_usd",
                    "response_time_seconds",
                    "calories_pct_error",
                    "carbs_pct_error",
                    "protein_pct_error",
                    "fat_pct_error",
                ]
                self.average_of_df = (
                    self.successful_df.groupby(["image_id", "base_model"])[numeric_cols]
                    .mean()
                    .reset_index()
                )
                self.average_of_df["model"] = self.average_of_df["base_model"].apply(
                    lambda x: (
                        f"{get_display_name(x)} (Avg)"
                        if x != "january/food-vision-v1"
                        else get_display_name(x)
                    )
                )

                self.average_of_df = self._add_overall_score(self.average_of_df)

                self.plot_df = pd.concat(
                    [self.best_of_df, self.average_of_df], ignore_index=True
                ).drop_duplicates(subset=["model", "image_id"], keep="first")

            else:
                self.best_of_df = pd.DataFrame()
                self.average_of_df = pd.DataFrame()
                self.plot_df = pd.DataFrame()

        else:
            self.successful_df = pd.DataFrame()
            self.best_of_df = pd.DataFrame()
            self.average_of_df = pd.DataFrame()
            self.plot_df = pd.DataFrame()

    def _add_overall_score(self, df):
        """
        Calculates a overall score using a weighted geometric mean.
        This method is robust to outliers and uses a "knock-out" criterion,
        where a score of 0 in any key metric results in an overall score of 0.
        """
        if df.empty:
            return df

        COST_CEILING = 1
        TIME_CEILING = 60

        weights = {
            "name_similarity": 0.15,
            "ing_accuracy": 0.40,
            "macro_accuracy": 0.25,
            "cost": 0.10,
            "speed": 0.10,
        }
        assert np.isclose(sum(weights.values()), 1.0), "Weights must sum to 1.0"

        norm_name_sim = df["meal_name_similarity"].clip(0, 1)
        norm_acc_ing = df["semantic_f1_ing"].clip(0, 1)
        norm_acc_macro = (1 - (df["wmape_mac"] / 100)).clip(0, 1)

        clipped_cost = df["cost_usd"].clip(upper=COST_CEILING)
        norm_cost = 1 - (clipped_cost / COST_CEILING)

        clipped_time = df["response_time_seconds"].clip(upper=TIME_CEILING)
        norm_speed = 1 - (clipped_time / TIME_CEILING)

        df["overall_score"] = 100 * (
            (norm_name_sim ** weights["name_similarity"])
            * (norm_acc_ing ** weights["ing_accuracy"])
            * (norm_acc_macro ** weights["macro_accuracy"])
            * (norm_cost ** weights["cost"])
            * (norm_speed ** weights["speed"])
        )

        df["norm_name_similarity"] = norm_name_sim
        df["norm_ing_accuracy"] = norm_acc_ing
        df["norm_macro_accuracy"] = norm_acc_macro
        df["norm_cost"] = norm_cost
        df["norm_speed"] = norm_speed

        return df

    def summary_statistics(self):
        """Generate comprehensive summary statistics for individual variants and aggregated models."""
        print("=== BENCHMARK SUMMARY (PER VARIANT) ===\n")
        if self.df.empty:
            print("No results to analyze.")
            return

        for model in sorted(self.df["model"].unique()):
            display_name = pretty_label(model)
            model_df = self.df[self.df["model"] == model]
            successful = model_df[model_df["error"].isna()]
            success_rate = (
                (len(successful) / len(model_df)) * 100 if len(model_df) > 0 else 0
            )

            print(f"--- {display_name} ---")
            print(
                f"  Success Rate: {success_rate:.1f}% ({len(successful)}/{len(model_df)})"
            )

            if not successful.empty:
                print(
                    f"  Avg Semantic Match (Embeddings): {successful['semantic_match_embeddings'].mean():.3f}"
                )
                print(
                    f"  Avg Semantic Precision (Ingredients): {successful['semantic_precision_ing'].mean():.3f}"
                )
                print(f"  Avg wMAPE (Macros): {successful['wmape_mac'].mean():.1f}%")
                print(f"  Avg Cost per Image: ${successful['cost_usd'].mean():.4f}")
                print(
                    f"  Avg Response Time: {successful['response_time_seconds'].mean():.1f}s\n"
                )

        print("\n=== AGGREGATED SUMMARY (BEST OF N PROMPTS) ===\n")
        if self.best_of_df.empty:
            print("No successful results to analyze for aggregation.")
        else:
            agg_best_df = (
                self.best_of_df.groupby("model")
                .agg(
                    semantic_match_embeddings=("semantic_match_embeddings", "mean"),
                    semantic_precision_ing=("semantic_precision_ing", "mean"),
                    wmape_mac=("wmape_mac", "mean"),
                    cost_usd=("cost_usd", "mean"),
                    response_time_seconds=("response_time_seconds", "mean"),
                    overall_score=("overall_score", "mean"),
                    sample_count=("image_id", "count"),
                )
                .reset_index()
            )

            for _, row in agg_best_df.iterrows():
                print(f"--- {row['model']} (from {row['sample_count']} samples) ---")
                print(
                    f"  Avg Semantic Match (Embeddings): {row['semantic_match_embeddings']:.3f}"
                )
                print(
                    f"  Avg Semantic Precision (Ingredients): {row['semantic_precision_ing']:.3f}"
                )
                print(f"  Avg wMAPE (Macros): {row['wmape_mac']:.1f}%")
                print(f"  Avg Cost per Image: ${row['cost_usd']:.4f}")
                print(f"  Avg Response Time: {row['response_time_seconds']:.1f}s\n")
                print(f"  overall Score: {row['overall_score']:.2f} / 100\n")

        print("\n=== AGGREGATED SUMMARY (AVERAGE) ===\n")
        if self.average_of_df.empty:
            print("No successful results to analyze for aggregation.")
            self.analyze_errors()
            return

        agg_df = (
            self.average_of_df.groupby("model")
            .agg(
                semantic_match_embeddings=("semantic_match_embeddings", "mean"),
                semantic_precision_ing=("semantic_precision_ing", "mean"),
                wmape_mac=("wmape_mac", "mean"),
                cost_usd=("cost_usd", "mean"),
                response_time_seconds=("response_time_seconds", "mean"),
                overall_score=("overall_score", "mean"),
                sample_count=("image_id", "count"),
            )
            .reset_index()
        )

        for _, row in agg_df.iterrows():
            print(f"--- {row['model']} (from {row['sample_count']} samples) ---")
            print(
                f"  Avg Semantic Match (Embeddings): {row['semantic_match_embeddings']:.3f}"
            )
            print(
                f"  Avg Semantic Precision (Ingredients): {row['semantic_precision_ing']:.3f}"
            )
            print(f"  Avg wMAPE (Macros): {row['wmape_mac']:.1f}%")
            print(f"  Avg Cost per Image: ${row['cost_usd']:.4f}")
            print(f"  Avg Response Time: {row['response_time_seconds']:.1f}s\n")
            print(f"  overall Score: {row['overall_score']:.2f} / 100\n")

        self.analyze_errors()

    def analyze_errors(self):
        """Analyze and summarize the specific errors encountered."""
        print("--- ERROR ANALYSIS ---")
        error_df = self.df[self.df["error"].notna()]
        if error_df.empty:
            print("No errors encountered. All API calls were successful.\n")
            return

        print("Error counts by model:")
        error_summary = error_df.groupby("model")["error"].count().sort_index()
        print(error_summary.to_string())

        print("\nMost common error messages:")
        common_errors = (
            error_df["error"].str.split(":").str[0].value_counts().nlargest(5)
        )
        print(common_errors.to_string())
        print()

    def create_performance_dashboard(self, save_path=None):
        """Create a comprehensive performance comparison dashboard with improved styling."""
        if self.plot_df.empty:
            print("No successful predictions to plot.")
            return

        models = sorted(self.plot_df["model"].unique())

        short_labels = {m: m for m in models}
        idx_map = {m: i + 1 for i, m in enumerate(short_labels)}

        colors = px.colors.qualitative.Plotly

        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=(
                "Meal Name Similarity",
                "Macro Nutritional wMAPE (%)",
                "Response Time Distribution",
                "Recall (Ingredients)",
                "Precision (Ingredients)",
                "F1 Score (Ingredients)",
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.08,
        )

        for i, model in enumerate(models):
            m_idx = str(idx_map[model])
            color = colors[i % len(colors)]
            d_ok = self.plot_df[self.plot_df["model"] == model]
            if d_ok.empty:
                continue

            box_style = dict(
                marker_color=color,
                marker_line_color="rgba(0,0,0,0.3)",
                marker_line_width=1,
                line_color=color,
                line_width=2,
            )
            fig.add_trace(
                go.Box(
                    y=d_ok["meal_name_similarity"],
                    name=m_idx,
                    legendgroup=m_idx,
                    showlegend=False,
                    **box_style,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Box(
                    y=d_ok["wmape_mac"],
                    name=m_idx,
                    legendgroup=m_idx,
                    showlegend=False,
                    **box_style,
                ),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Box(
                    y=d_ok["response_time_seconds"],
                    name=m_idx,
                    legendgroup=m_idx,
                    showlegend=False,
                    **box_style,
                ),
                row=1,
                col=3,
            )
            fig.add_trace(
                go.Box(
                    y=d_ok["semantic_match_embeddings"],
                    name=m_idx,
                    legendgroup=m_idx,
                    showlegend=False,
                    **box_style,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Box(
                    y=d_ok["semantic_precision_ing"],
                    name=m_idx,
                    legendgroup=m_idx,
                    showlegend=False,
                    **box_style,
                ),
                row=2,
                col=2,
            )
            fig.add_trace(
                go.Box(
                    y=d_ok["semantic_f1_ing"],
                    name=m_idx,
                    legendgroup=m_idx,
                    showlegend=False,
                    **box_style,
                ),
                row=2,
                col=3,
            )

        for i, model in enumerate(models):
            m_idx = str(idx_map[model])
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=colors[i % len(colors)],
                        symbol="circle",
                        line=dict(width=2, color="rgba(0,0,0,0.3)"),
                    ),
                    legendgroup=m_idx,
                    showlegend=True,
                    name=f"{m_idx}: {short_labels[model]}",
                )
            )

        axis_style = dict(
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.4)",
            zerolinewidth=1,
            tickfont=dict(size=11, color="#2f2f2f"),
        )
        for row in [1, 2]:
            for col in [1, 2, 3]:
                fig.update_xaxes(**axis_style, row=row, col=col, overwrite=True)
                fig.update_yaxes(**axis_style, row=row, col=col, overwrite=True)
        fig.update_yaxes(title_text="Cosine Similarity", row=1, col=1)
        fig.update_yaxes(title_text="Weighted MAPE (%)", row=1, col=2)
        fig.update_yaxes(title_text="Response Time (sec)", row=1, col=3)
        fig.update_yaxes(title_text="Recall", row=2, col=1)
        fig.update_yaxes(title_text="Precision", row=2, col=2)
        fig.update_yaxes(title_text="F1 Score", row=2, col=3)
        fig.update_layout(
            height=800,
            width=1400,
            title=dict(
                text="<b>Model Performance Dashboard</b>",
                x=0.5,
                xanchor="center",
                font=dict(size=24, color="#1f1f1f", family="Arial Black"),
            ),
            showlegend=True,
            legend=dict(
                title="<b>Models</b>",
                title_font=dict(size=14, color="#1f1f1f"),
                font=dict(size=11),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(128,128,128,0.5)",
                borderwidth=1,
                x=1.02,
                y=1,
                xanchor="left",
                yanchor="top",
            ),
            plot_bgcolor="rgba(248,249,250,0.8)",
            paper_bgcolor="white",
            font=dict(family="Arial, sans-serif", size=11, color="#2f2f2f"),
            margin=dict(l=80, r=200, t=120, b=80),
            hovermode="closest",
        )
        config = dict(
            displayModeBar=True,
            displaylogo=False,
            modeBarButtonsToRemove=["pan2d", "lasso2d"],
            toImageButtonOptions=dict(
                format="png",
                filename="model_performance_dashboard",
                height=800,
                width=1400,
                scale=2,
            ),
        )
        if save_path:
            fig.write_html(str(save_path), config=config)
        else:
            fig.show(config=config)
            fig.write_html("performance_dashboard.html", config=config)

    def export_detailed_report(self, filename="benchmark_report.csv"):
        """Export a detailed report with raw data and summary to CSV files."""
        print(f"Exporting detailed report to {filename}...")
        self.df.to_csv(filename, index=False)

        if not self.plot_df.empty:
            summary = (
                self.plot_df.groupby("model")
                .agg(
                    semantic_match_embeddings_mean=(
                        "semantic_match_embeddings",
                        "mean",
                    ),
                    semantic_precision_ing_mean=("semantic_precision_ing", "mean"),
                    wmape_mac_mean=("wmape_mac", "mean"),
                    cost_usd_total=("cost_usd", "sum"),
                    response_time_seconds_mean=("response_time_seconds", "mean"),
                )
                .reset_index()
            )
            summary_filename = filename.replace(".csv", "_summary.csv")
            summary.to_csv(summary_filename, index=False)
        print("Export complete.")

    def create_win_loss_analysis(
        self, baseline_model_name: Optional[str] = None, save_path=None
    ):
        """Win-Tie-Loss analysis comparing average performance of a baseline model
        against the Average and Best performance of competitor models."""
        print(
            "\n=== WIN-TIE-LOSS ANALYSIS (Comparing Best-of and Average-of Performance) ==="
        )
        if self.average_of_df.empty or self.best_of_df.empty:
            print("No successful results to analyze.")
            return

        base_models = sorted(self.average_of_df["base_model"].unique())

        if len(base_models) < 2:
            print("Need at least two models to perform a win-loss comparison.")
            return

        if baseline_model_name:
            if baseline_model_name not in base_models:
                print(
                    f"Error: Baseline model '{baseline_model_name}' not found in results."
                )
                print(f"Available models are: {base_models}")
                return
            baseline_model = baseline_model_name
        else:
            baseline_model = (
                "january/food-vision-v1"
                if "january/food-vision-v1" in base_models
                else base_models[0]
            )
            print(
                f"INFO: No baseline model specified. Using average performance of '{get_display_name(baseline_model)}' as the default."
            )

        baseline_data = self.average_of_df[
            self.average_of_df["base_model"] == baseline_model
        ].copy()

        competitor_dfs = {
            "Avg": self.average_of_df[
                self.average_of_df["base_model"] != baseline_model
            ],
            "Best": self.best_of_df[self.best_of_df["base_model"] != baseline_model],
        }

        macro_cols = [
            "calories_pct_error",
            "carbs_pct_error",
            "protein_pct_error",
            "fat_pct_error",
        ]
        win_loss_data = {}

        for perf_type, df in competitor_dfs.items():
            base_model_series = df["base_model"]
            unique_models_list = list(set(base_model_series.tolist()))
            for competitor_base_model in unique_models_list:
                competitor_model_name = (
                    f"{get_display_name(competitor_base_model)} ({perf_type})"
                )
                win_loss_data[competitor_model_name] = {
                    m: {"win": 0, "tie": 0, "loss": 0} for m in macro_cols
                }

                competitor_data = df[df["base_model"] == competitor_base_model]

                # Ensure we have DataFrame objects for merge
                if not isinstance(competitor_data, pd.DataFrame):
                    competitor_data = pd.DataFrame(competitor_data)

                comparison_df = pd.merge(
                    baseline_data,
                    competitor_data,
                    on="image_id",
                    suffixes=("_base", "_comp"),
                )
                if comparison_df.empty:
                    continue

                for macro in macro_cols:
                    base_error = comparison_df[f"{macro}_base"]
                    comp_error = comparison_df[f"{macro}_comp"]
                    win_loss_data[competitor_model_name][macro]["win"] = (
                        base_error < comp_error
                    ).sum()
                    win_loss_data[competitor_model_name][macro]["tie"] = (
                        base_error == comp_error
                    ).sum()
                    win_loss_data[competitor_model_name][macro]["loss"] = (
                        base_error > comp_error
                    ).sum()

        if not win_loss_data:
            print("No common images found to compare models.")
            return

        competitor_models = sorted(win_loss_data.keys())
        colors = {"win": "#16A085", "tie": "#95A5A6", "loss": "#E74C3C"}
        macro_titles = ["Calories", "Carbohydrates", "Protein", "Fat"]
        fig = make_subplots(
            rows=len(competitor_models),
            cols=4,
            subplot_titles=macro_titles,
            shared_xaxes=True,
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
        )

        for i, competitor in enumerate(competitor_models):
            row_idx = i + 1

            for j, macro in enumerate(macro_cols):
                col_idx = j + 1
                data = win_loss_data[competitor][macro]
                total = sum(data.values())
                if total == 0:
                    continue
                win_pct, tie_pct, loss_pct = (
                    data["win"] / total * 100,
                    data["tie"] / total * 100,
                    data["loss"] / total * 100,
                )
                fig.add_trace(
                    go.Bar(
                        name="Win",
                        x=[win_pct],
                        y=[competitor],
                        orientation="h",
                        marker=dict(color=colors["win"]),
                        showlegend=False,
                        text=f"{win_pct:.0f}%" if win_pct > 5 else "",
                        textposition="inside",
                        customdata=[data["win"]],
                        hovertemplate="<b>Win</b><br>Count: %{customdata}<br>Percentage: %{x:.1f}%<br><extra></extra>",
                    ),
                    row=row_idx,
                    col=col_idx,
                )
                fig.add_trace(
                    go.Bar(
                        name="Tie",
                        x=[tie_pct],
                        y=[competitor],
                        orientation="h",
                        marker=dict(color=colors["tie"]),
                        showlegend=False,
                        text=f"{tie_pct:.0f}%" if tie_pct > 5 else "",
                        textposition="inside",
                        customdata=[data["tie"]],
                        hovertemplate="<b>Tie</b><br>Count: %{customdata}<br>Percentage: %{x:.1f}%<br><extra></extra>",
                    ),
                    row=row_idx,
                    col=col_idx,
                )
                fig.add_trace(
                    go.Bar(
                        name="Loss",
                        x=[loss_pct],
                        y=[competitor],
                        orientation="h",
                        marker=dict(color=colors["loss"]),
                        showlegend=False,
                        text=f"{loss_pct:.0f}%" if loss_pct > 5 else "",
                        textposition="inside",
                        customdata=[data["loss"]],
                        hovertemplate="<b>Loss</b><br>Count: %{customdata}<br>Percentage: %{x:.1f}%<br><extra></extra>",
                    ),
                    row=row_idx,
                    col=col_idx,
                )
                fig.update_yaxes(
                    row=row_idx, col=col_idx, showticklabels=False, title_text=""
                )

            pretty_base = get_display_name(baseline_model)
            pretty_comp = competitor
            axis_num = i * 4 + 1
            y_anchor_ref = f"y{'' if axis_num == 1 else axis_num}"

            fig.add_annotation(
                x=-0.05,
                y=competitor,
                xref="paper",
                yref=y_anchor_ref,
                text=f"<b>{pretty_base}</b>",
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                font=dict(size=12),
            )
            fig.add_annotation(
                x=1.05,
                y=competitor,
                xref="paper",
                yref=y_anchor_ref,
                text=f"<b>{pretty_comp}</b>",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=12),
            )

        fig.update_layout(
            barmode="stack",
            title={
                "text": f"<b>{get_display_name(baseline_model)} vs. Others</b>",
                "x": 0.5,
                "xanchor": "center",
            },
            height=max(200, 60 * len(competitor_models) + 80),
            width=1000,
            showlegend=False,
            plot_bgcolor="rgba(250,251,252,0.8)",
            paper_bgcolor="white",
            margin=dict(l=200, r=200, t=100, b=20),
        )
        fig.update_xaxes(range=[0, 100], ticksuffix="%", showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        print(
            f"\nComparing {get_display_name(baseline_model)} (Average performance) against:"
        )
        for competitor in competitor_models:
            total_comparisons = sum(
                sum(win_loss_data[competitor][m].values()) for m in macro_cols
            )
            if total_comparisons > 0:
                baseline_wins = sum(
                    win_loss_data[competitor][m]["win"] for m in macro_cols
                )
                baseline_losses = sum(
                    win_loss_data[competitor][m]["loss"] for m in macro_cols
                )
                print(
                    f"  - vs {competitor}: {baseline_wins} wins, {baseline_losses} losses across {total_comparisons // 4} images"
                )
        if save_path:
            fig.write_html(str(save_path))
        else:
            fig.show()

    def plot_overall_score(self, save_path=None):
        """Create a bar chart comparing the overall score of each model."""
        if self.plot_df.empty or "overall_score" not in self.plot_df.columns:
            print("No overall score data to plot.")
            return

        summary_df = self.plot_df.groupby("model")["overall_score"].mean().reset_index()
        summary_df = summary_df.sort_values("overall_score", ascending=False)

        fig = px.bar(
            summary_df,
            x="model",
            y="overall_score",
            title="<b>Overall Score</b>",
            labels={"model": "Model", "overall_score": "Composite Score (0-100)"},
            text="overall_score",
            color="model",
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")

        max_score = summary_df["overall_score"].max() if not summary_df.empty else 100

        fig.update_layout(
            uniformtext_minsize=8,
            uniformtext_mode="hide",
            xaxis_title=None,
            xaxis_tickangle=-45,
            showlegend=False,
            title_x=0.5,
            title_font=dict(size=20, family="Arial Black"),
            font=dict(family="Arial, sans-serif", size=12),
            yaxis_range=[0, max_score * 1.15],
        )
        if save_path:
            fig.write_html(str(save_path))
        else:
            fig.show()
