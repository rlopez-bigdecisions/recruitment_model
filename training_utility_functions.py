import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpecFromSubplotSpec
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report


import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    explained_variance_score
)

def evaluate_regression_model(
    model,
    X_train, y_train,
    X_val,   y_val,
    X_test=None, y_test=None,
    X_predict=None
):
    """
    Train a regression model and compute a battery of regression metrics
    on train / validation / test / prediction datasets.

    Parameters
    ----------
    model : scikit-learn compatible regressor
    X_*   : feature matrices
    y_*   : target vectors
    X_predict : feature matrix used only to obtain predictions
    """
    def score_dataset(X, y, y_pred):
        """Return a dict with regression metrics for a single dataset."""
        scores = {}
        if y is not None:
            # Standard regression metrics
            scores["mae"]   = mean_absolute_error(y, y_pred)
            scores["mse"]   = mean_squared_error(y, y_pred)
            scores["rmse"] = np.sqrt(scores["mse"])
            # scores["rmse"]  = mean_squared_error(y, y_pred, squared=False)
            scores["medae"] = median_absolute_error(y, y_pred)
            scores["r2"]    = r2_score(y, y_pred)
            scores["explained_variance"] = explained_variance_score(y, y_pred)
            # Extra information that can be useful
            scores["n_samples"] = len(y)
        else:
            # When y is None (pure prediction set) we skip the metrics
            scores = {k: None for k in [
                "mae", "mse", "rmse", "medae", "r2",
                "explained_variance", "n_samples"
            ]}
        return scores

    # ----- 1. Fit the model --------------------------------------------------
    print(f"Fitting model {model}...")
    model.fit(X_train, y_train)

    # ----- 2. Prepare container ---------------------------------------------
    datasets = ["train", "val", "test", "predict"]
    Xs       = [X_train, X_val,  X_test,  X_predict]
    ys       = [y_train, y_val,  y_test,  None]

    results = {
        "model": model,
        "datasets": {}
    }

    # ----- 3. Predict & score each split ------------------------------------
    for ds_name, X, y in zip(datasets, Xs, ys):
        print(f"\tPredicting and scoring on dataset '{ds_name}'...")
        if X is None:  # Skip if the split was not provided
            results["datasets"][ds_name] = None
            continue

        y_pred = model.predict(X)

        results["datasets"][ds_name] = {
            "predictions": y_pred,
            "metrics":     score_dataset(X, y, y_pred)
        }

    return results



def scores_df(scores):
    columns = ["model", "threshold", "dataset", "f1_score", "recall", "precision", "accuracy", "count_1s_predict", "ratio_1s_predict", "type1_error", "type2_error"]
    scores_df = pd.DataFrame(columns=columns)
    for model, model_data in scores.items():
        for dataset, dataset_scores in model_data["datasets"].items():
            for threshold, threshold_scores in dataset_scores["metrics"].items():
                scores_df = pd.concat([scores_df,
                                        pd.DataFrame({
                                            "model": model,
                                            "threshold": threshold,
                                            "dataset": dataset,
                                            "f1_score": threshold_scores["f1_score"],
                                            "recall": threshold_scores["recall"],
                                            "precision": threshold_scores["precision"],
                                            "accuracy": threshold_scores["accuracy"],
                                            "count_1s_predict": model_data["datasets"]["predict"]["metrics"][threshold]["count_1s"],
                                            "ratio_1s_predict": model_data["datasets"]["predict"]["metrics"][threshold]["ratio_1s"],
                                            "type1_error": threshold_scores["type1_error"],
                                            "type2_error": threshold_scores["type2_error"]
                                        }, index=[0])])
    scores_df = scores_df.reset_index(drop=True)
    scores_df = scores_df.sort_values(by=["dataset", "f1_score"], ascending=False)
    return scores_df


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def barplot_regression_models_scores(
    scores,
    plot_datasets=("train", "val"),
    plot_metrics=("rmse", "mae", "r2", "explained_variance")
):
    """
    Visualize regression metrics for several models and datasets in a grid of bar plots.

    Parameters
    ----------
    scores : dict
        Output of `evaluate_regression_model`, shaped like:
        {
          "modelA": {
            "datasets": {
              "train":   {"metrics": {"rmse": 0.52, "mae": ...}},
              "val":     {"metrics": {...}},
              ...
            }
          },
          "modelB": { ... },
          ...
        }

    plot_datasets : tuple[str]
        Datasets to include in the plots, e.g. ("train", "val", "test").

    plot_metrics : tuple[str]
        Regression metrics to plot as rows.
    """
    # Style per dataset (colors / hatch patterns, pick what you like)
    DATASET_STYLE = {
        "train":   {"color": "C0", "hatch": ""},
        "val":     {"color": "C1", "hatch": "//"},
        "test":    {"color": "C2", "hatch": "xx"},
        "predict": {"color": "C3", "hatch": ".."},
    }

    # %-style metrics – express them on a 0-100 scale
    PERCENT_METRICS = {"r2"}          # you can add e.g. "mape" if you compute it

    # --- Prepare figure ------------------------------------------------------
    n_rows = len(plot_metrics)
    n_cols = 1
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    if n_rows == 1:                   # ensure iterable
        axs = np.array([axs])

    model_names = list(scores.keys())
    x = np.arange(len(model_names))   # model positions on the x-axis
    width = 0.8 / len(plot_datasets)  # bar width so all datasets fit side-by-side

    percent_formatter = FuncFormatter(lambda v, _: f"{v:.0f}%")

    # --- Main loop over metrics (rows) ---------------------------------------
    for r, metric in enumerate(plot_metrics):
        ax = axs[r]

        for i, ds_name in enumerate(plot_datasets):
            # Gather metric values per model (None -> np.nan so bar can be empty)
            vals = []
            for m in model_names:
                ds = scores[m]["datasets"].get(ds_name)
                if ds is None or "metrics" not in ds:
                    val = np.nan
                else:
                    val = ds["metrics"].get(metric, np.nan)
                vals.append(val)

            # Percent metrics -> scale ×100
            vals_plot = np.array(vals, dtype=float)
            if metric in PERCENT_METRICS:
                vals_plot = vals_plot * 100

            # Offset each dataset’s bar group
            offset = (i - (len(plot_datasets) - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                vals_plot,
                width,
                label=ds_name.capitalize(),
                color=DATASET_STYLE.get(ds_name, {}).get("color"),
                hatch=DATASET_STYLE.get(ds_name, {}).get("hatch", ""),
                edgecolor="black",
                alpha=0.85,
            )

            # Label each bar with its numeric value
            for rect, value in zip(bars, vals_plot):
                if np.isnan(value):
                    continue
                label = f"{value:.2f}" if metric not in PERCENT_METRICS else f"{value:.1f}"
                height = rect.get_height()
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    height,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # Axes cosmetics
        ax.set_title(metric.upper(), fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=30, ha="right")
        if metric in PERCENT_METRICS:
            ax.yaxis.set_major_formatter(percent_formatter)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    # --- Global legend -------------------------------------------------------
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(plot_datasets), frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])   # leave room for legend
    plt.show()


def plotbar_final_model_scores(
    scores,
    selected_model,
    selected_threshold,
    plot_metrics=(
        ("accuracy", "precision", "recall"),
        ("f1_score", "type1_error", "type2_error")
    )
):
    """
    Creates a 2D grid of bar charts (by default 2 rows × 3 columns = 6 subplots)
    for the given 'selected_model' and 'selected_threshold'. Each cell in
    'plot_metrics' is a metric name (e.g. "accuracy", "precision", etc.).

    Data structure:
      scores = {
        "modelA": {
          "train": {
            "predictions": {...},
            "metrics": {
              "0.30": {"accuracy": <val>, "precision": <val>, "recall": <val>, ...},
              "0.50": {...}
            }
          },
          "val": {...},
          "test": {...}
        },
        "modelB": {...}
      }

    For each metric in 'plot_metrics':
      - We create a bar chart of 3 bars (train, val, test).
      - We read the value from scores[selected_model][dataset]["metrics"]
        [selected_threshold][metric]. If missing, we assume 0.
      - We convert the value to percentage (value * 100).
      - The Y-axis is fixed at 0..100% (with one decimal place).
      - We place a numeric label (e.g. "27.8%") above each bar.
      - The X-axis is labeled ["Train", "Validation", "Test"].
      - There is no legend in this figure.

    """

    def percent_formatter_one_decimal(value, _):
        """Format number as one-decimal percent, e.g. 27.8%"""
        return f"{value:.1f}%"

    # We'll display three bars for these three datasets
    dataset_order = ["train", "val", "test"]
    dataset_labels = ["Train", "Validation", "Test"]  # for the X-axis

    # Create a figure with enough space for a 2×3 grid
    rows = len(plot_metrics)
    cols = len(plot_metrics[0])
    fig, axs = plt.subplots(rows, cols, figsize=(18, 10))

    # In case 'plot_metrics' is strictly 2D, we can do axs[r][c]. For 2×3, that's fine.
    for r in range(rows):
        for c in range(cols):
            metric_name = plot_metrics[r][c]
            ax = axs[r][c]

            # Collect metric values for train/val/test
            values = []
            for ds in dataset_order:
                ds_data = scores.get(selected_model, {}).get("datasets", {}).get(ds, {})
                thr_metrics = ds_data.get("metrics", {}).get(selected_threshold, {})
                val = thr_metrics.get(metric_name, 0.0)
                val_percent = val * 100.0
                values.append(val_percent)

            # Make a bar chart with 3 bars
            x_positions = np.arange(len(dataset_order))  # [0, 1, 2]
            bars = ax.bar(x_positions, values, width=0.6)

            # Place textual labels (e.g. "27.8%") above each bar
            # Using an offset of about 2% of the maximum bar height ensures
            # labels won't collide with the top boundary when y=100.
            max_val = max(values) if values else 0
            offset = max_val * 0.02 if max_val else 1.0

            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + offset,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=10
                )

            # Title is the metric name
            ax.set_title(metric_name)
            # Configure X-axis
            ax.set_xticks(x_positions)
            ax.set_xticklabels(dataset_labels)
            # Y-axis in percentage format, 0..100
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter_one_decimal))

    fig.tight_layout()
    plt.show()
