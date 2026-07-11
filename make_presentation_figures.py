from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results"

MODEL_ORDER = [
    "DummyClassifier",
    "LogisticRegression",
    "MLP",
    "LightGBM",
    "XGBoost",
    "CatBoost",
]

COLORS = {
    "DummyClassifier": "#9CA3AF",
    "LogisticRegression": "#4C78A8",
    "MLP": "#F58518",
    "LightGBM": "#54A24B",
    "XGBoost": "#E45756",
    "CatBoost": "#7B61FF",
}


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()


def load_metrics() -> pd.DataFrame:
    df = pd.read_csv(RESULT_DIR / "tabular_model_comparison.csv")
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    return df.sort_values("model")


def make_metric_summary(metrics: pd.DataFrame) -> None:
    plot_df = metrics[metrics["model"] != "DummyClassifier"].copy()
    plot_df = plot_df.sort_values("f1_score", ascending=True)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.2), sharey=True)
    fig.suptitle("Presentation Summary: Key Metrics by Model", fontsize=16, fontweight="bold")

    specs = [
        ("f1_score", "F1-score", "Balanced detection"),
        ("recall", "Recall", "Risk coverage"),
        ("average_precision", "Average Precision", "Positive ranking"),
    ]

    y = np.arange(len(plot_df))
    for ax, (col, title, subtitle) in zip(axes, specs):
        colors = [COLORS[m] for m in plot_df["model"].astype(str)]
        ax.barh(y, plot_df[col], color=colors, alpha=0.88)
        ax.set_title(f"{title}\n{subtitle}", fontsize=11)
        ax.set_xlim(0, max(0.75, float(plot_df[col].max()) + 0.08))
        ax.grid(axis="x", alpha=0.18)
        for i, value in enumerate(plot_df[col]):
            ax.text(value + 0.01, i, f"{value:.3f}", va="center", fontsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(plot_df["model"].astype(str), fontsize=10)
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False, labelleft=False)

    fig.text(
        0.01,
        0.01,
        "Note: DummyClassifier is excluded from this presentation chart because its F1/Recall are 0 under threshold 0.5.",
        fontsize=9,
        color="#555555",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    savefig(RESULT_DIR / "presentation_metric_summary.png")


def make_selected_curves(metrics: pd.DataFrame) -> None:
    preds = pd.read_csv(RESULT_DIR / "tabular_test_predictions.csv")
    selected = ["LogisticRegression", "MLP", "CatBoost"]
    positive_ratio = float(preds[preds["model"] == selected[0]]["delay_risk"].mean())

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    fig.suptitle("Selected ROC / PR Curves for Presentation", fontsize=16, fontweight="bold")

    for model in selected:
        sub = preds[preds["model"] == model]
        y_true = sub["delay_risk"].astype(int).to_numpy()
        y_prob = sub["y_prob"].astype(float).to_numpy()
        row = metrics[metrics["model"].astype(str) == model].iloc[0]

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)

        axes[0].plot(
            fpr,
            tpr,
            linewidth=2.2,
            color=COLORS[model],
            label=f"{model} (AUC={row['roc_auc']:.3f})",
        )
        axes[1].plot(
            recall,
            precision,
            linewidth=2.2,
            color=COLORS[model],
            label=f"{model} (AP={row['average_precision']:.3f})",
        )

    axes[0].plot([0, 1], [0, 1], "--", color="#9CA3AF", linewidth=1.5, label="Random")
    axes[0].set_title("ROC curve: ranking ability")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="lower right", fontsize=9)

    axes[1].axhline(
        positive_ratio,
        linestyle="--",
        color="#9CA3AF",
        linewidth=1.5,
        label=f"Positive ratio ({positive_ratio:.3f})",
    )
    axes[1].set_title("PR curve: imbalanced-class view")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper right", fontsize=9)

    fig.text(
        0.01,
        0.01,
        "Selected models only: LogisticRegression baseline, MLP neural baseline, CatBoost best F1/ROC-AUC model.",
        fontsize=9,
        color="#555555",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    savefig(RESULT_DIR / "presentation_selected_roc_pr.png")


def make_confusion_matrix(metrics: pd.DataFrame) -> None:
    row = metrics[metrics["model"].astype(str) == "CatBoost"].iloc[0]
    cm = np.array([[int(row["tn"]), int(row["fp"])], [int(row["fn"]), int(row["tp"])]])
    labels = np.array([["TN", "FP"], ["FN", "TP"]])

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("CatBoost Confusion Matrix\n(test set, threshold=0.58)", fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])

    threshold = cm.max() / 2
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > threshold else "#111827"
            ax.text(j, i, f"{labels[i, j]}\n{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=14, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count", rotation=270, labelpad=14)
    fig.text(
        0.5,
        0.01,
        "FN is the missed-risk group; FP is the over-alert group. Use this with Recall/F1 interpretation.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    savefig(RESULT_DIR / "presentation_catboost_confusion_matrix.png")


def make_threshold_sensitivity() -> None:
    sens = pd.read_csv(RESULT_DIR / "delay_threshold_sensitivity.csv")
    selected = ["MLP", "LightGBM", "XGBoost", "CatBoost"]
    plot_df = sens[sens["model"].isin(selected)].copy()
    label_map = {
        "delay_gt_15_q80": "q80: target_gap > 15",
        "delay_gt_24_q90": "q90: target_gap > 24",
    }
    pivot = (
        plot_df.pivot(index="model", columns="label_name", values="f1_score")
        .reindex(selected)
        .rename(columns=label_map)
    )

    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    x = np.arange(len(pivot.index))
    width = 0.34
    ax.bar(x - width / 2, pivot["q80: target_gap > 15"], width, label="q80: target_gap > 15", color="#2F7776")
    ax.bar(x + width / 2, pivot["q90: target_gap > 24"], width, label="q90: target_gap > 24", color="#E39D43")

    for i, model in enumerate(pivot.index):
        ax.text(i - width / 2, pivot.loc[model, "q80: target_gap > 15"] + 0.008, f"{pivot.loc[model, 'q80: target_gap > 15']:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, pivot.loc[model, "q90: target_gap > 24"] + 0.008, f"{pivot.loc[model, 'q90: target_gap > 24']:.3f}", ha="center", fontsize=9)

    ax.set_title("Delay Threshold Sensitivity\nq80 early-detection vs q90 high-risk label", fontsize=14, fontweight="bold")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 0.58)
    ax.set_xticks(x, pivot.index)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.01,
        0.01,
        "q80 positive ratio=18.65%; q90 positive ratio=9.90%. q90 is stricter but has fewer positive samples.",
        fontsize=9,
        color="#555555",
    )
    plt.tight_layout(rect=[0, 0.04, 0.84, 1])
    savefig(RESULT_DIR / "presentation_threshold_sensitivity.png")


def main() -> None:
    metrics = load_metrics()
    make_metric_summary(metrics)
    make_selected_curves(metrics)
    make_confusion_matrix(metrics)
    make_threshold_sensitivity()


if __name__ == "__main__":
    main()
