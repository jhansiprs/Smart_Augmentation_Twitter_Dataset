# ============================================================
# Reproducibility
# ============================================================

import os
import random
import numpy as np
import torch

SEEDS = [7, 42, 99, 123, 2024]

def set_reproducibility(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)

    from transformers import set_seed
    set_seed(seed)


# ============================================================
# Imports
# ============================================================

import sys
import platform
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, recall_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmentation.smart_augmentation_hybrid_new import augment_dataset
from models.albert_model import ALBERTModel
from training.train_single_model import train_model
from evaluation.evaluate_model import evaluate_model
from evaluation.visualization import plot_training_history
from evaluation.confusion_matrix import plot_confusion_matrix
from evaluation.storage import save_metrics


# ============================================================
# ROC Curve
# ============================================================

def plot_roc_curve(y_true, probs, save_path):
    probs = np.array(probs)
    classes = np.unique(y_true)
    y_true_bin = label_binarize(y_true, classes=classes)

    plt.figure()

    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
        auc = roc_auc_score(y_true_bin[:, i], probs[:, i])
        plt.plot(fpr, tpr, label=f"Class {classes[i]} AUC={auc:.3f}")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig(save_path)
    plt.close()


# ============================================================
# Recall Table
# ============================================================

def create_recall_table(results_df):
    recall_cols = [c for c in results_df.columns if "recall_class" in c]
    rows = []

    for model in results_df["model"].unique():
        model_df = results_df[results_df["model"] == model]
        row = {"model": model}

        for col in recall_cols:
            row[col] = model_df[col].mean()

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Main Experiment
# ============================================================

def run_augmentation_experiment():

    print("Running Experiment: Smart Augmentation")

    base_dir = os.path.dirname(os.path.dirname(__file__))

    train_path = os.path.join(base_dir, 'data/splits/train.csv')
    augmented_train_path = os.path.join(base_dir, 'data/splits/train_smart_augmentation_hybrid.csv')
    val_path = os.path.join(base_dir, 'data/splits/val.csv')
    test_path = os.path.join(base_dir, 'data/splits/test.csv')

    results_dir = os.path.join(base_dir, 'results_augmented-ALBERTModel')
    os.makedirs(results_dir, exist_ok=True)

    # --------------------------------------------------------
    # Augment Dataset
    # --------------------------------------------------------

    if not os.path.exists(augmented_train_path):
        print("Creating augmented dataset...")
        augment_dataset(train_path, augmented_train_path)

    train_df = pd.read_csv(augmented_train_path)
    val_df = pd.read_csv(val_path)

    all_results = []

    # ========================================================
    # Seed Loop
    # ========================================================

    for seed in SEEDS:

        print(f"\n===== SEED {seed} =====")

        set_reproducibility(seed)

        print("Python:", platform.python_version())
        print("Torch:", torch.__version__)

        model = ALBERTModel()
        model_name = "albert"

        # ------------------ Train ------------------
        history, trained_model = train_model(model, train_df, val_df)

        if seed == SEEDS[0]:
            plot_training_history(
                history,
                save_path=os.path.join(results_dir, f"{model_name}_training.png")
            )

        # ------------------ Evaluate ------------------
        metrics, preds, true_vals, probs = evaluate_model(
            trained_model,
            test_path
        )

        # ------------------ Metrics ------------------
        metrics["f1_macro"] = f1_score(true_vals, preds, average="macro")
        metrics["f1_micro"] = f1_score(true_vals, preds, average="micro")
        metrics["f1_weighted"] = f1_score(true_vals, preds, average="weighted")

        metrics["recall_macro"] = recall_score(true_vals, preds, average="macro")
        metrics["recall_micro"] = recall_score(true_vals, preds, average="micro")

        classes = np.unique(true_vals)

        recall_per_class = recall_score(true_vals, preds, average=None)
        for i, cls in enumerate(classes):
            metrics[f"recall_class_{cls}"] = recall_per_class[i]

        # ------------------ ROC AUC ------------------
        y_true_bin = label_binarize(true_vals, classes=classes)

        metrics["roc_auc_macro"] = roc_auc_score(
            y_true_bin,
            probs,
            average="macro",
            multi_class="ovr"
        )

        # ------------------ Plots ------------------
        try:
            plot_roc_curve(
                true_vals,
                probs,
                os.path.join(results_dir, f"{model_name}_seed{seed}_roc.png")
            )
        except Exception as e:
            print("ROC ERROR:", e)

        try:
            plot_confusion_matrix(
                true_vals,
                preds,
                save_path=os.path.join(results_dir, f"{model_name}_seed{seed}_cm.png")
            )
        except Exception as e:
            print("CM ERROR:", e)

        # ------------------ Metadata ------------------
        metrics["seed"] = seed
        metrics["model"] = model_name

        save_metrics(
            f"exp_aug_{model_name}_seed{seed}",
            metrics,
            save_dir=results_dir
        )

        all_results.append(metrics)

    # =====================================================
    # Save Combined Results
    # =====================================================

    results_df = pd.DataFrame(all_results)

    results_path = os.path.join(results_dir, "all_results-Smart-ALBERTModel.csv")
    results_df.to_csv(results_path, index=False)

    print("\nSaved:", results_path)

    # =====================================================
    # Recall Table
    # =====================================================

    recall_df = create_recall_table(results_df)

    recall_path = os.path.join(results_dir, "recall_table-Smart-ALBERTModel.csv")
    recall_df.to_csv(recall_path, index=False)

    print("Saved:", recall_path)

    return results_df


# ============================================================

if __name__ == "__main__":
    run_augmentation_experiment()