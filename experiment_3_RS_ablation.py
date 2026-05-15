# ============================================================
# Reproducibility
# ============================================================

import os
import random
import numpy as np
import torch

SEEDS = [7, 42, 99,123, 2024]


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

from sklearn.metrics import (
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

from sklearn.preprocessing import label_binarize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bert_model import BERTModel
from models.roberta_model import RoBERTaModel
from models.distilbert_model import DistilBERTModel
from models.albert_model import ALBERTModel

from augmentation.smart_augmentation_hybrid_new import augment_dataset

from training.train_single_model import train_model
from evaluation.evaluate_model import evaluate_model
from evaluation.visualization import plot_training_history
from evaluation.confusion_matrix import plot_confusion_matrix
from evaluation.storage import save_metrics


# ============================================================
# ROC Plot
# ============================================================

def plot_roc_curve(y_true, probs, save_path):

    probs = np.array(probs)

    classes = np.unique(y_true)

    y_true_bin = label_binarize(y_true, classes=classes)

    plt.figure()

    for i in range(len(classes)):

        fpr, tpr, _ = roc_curve(
            y_true_bin[:, i],
            probs[:, i]
        )

        auc = roc_auc_score(
            y_true_bin[:, i],
            probs[:, i]
        )

        plt.plot(fpr, tpr, label=f"Class {classes[i]} AUC={auc:.3f}")

    plt.plot([0,1],[0,1],'k--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig(save_path)
    plt.close()


# ============================================================
# Recall Summary Table
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

def run_experiment():

    print("\nRunning Experiment 3: Ablation Study with Multi-Seed Evaluation for RS\n")

    base_dir = os.path.dirname(os.path.dirname(__file__))

    train_path = os.path.join(base_dir, 'data/splits/train.csv')
    val_path   = os.path.join(base_dir, 'data/splits/val.csv')
    test_path  = os.path.join(base_dir, 'data/splits/test.csv')

    results_dir = os.path.join(base_dir, 'results_exp3_ablation_multiseed_for_RS')
    os.makedirs(results_dir, exist_ok=True)

    val_df = pd.read_csv(val_path)

    # =====================================================
    # Augmentation Configurations (Experiment 5)
    # =====================================================

    configs = {

        'RS_Only': {
            'random_swap': 1.0
        }
    }

    model_classes = [
        BERTModel,
        RoBERTaModel,
        DistilBERTModel,
        ALBERTModel
    ]

    all_results = []

    # =====================================================
    # Config Loop (Ablation)
    # =====================================================

    for config_name, weights in configs.items():

        print("\n==========================================")
        print("Running Ablation Config:", config_name)
        print("==========================================")

        aug_path = os.path.join(
            base_dir,
            f'data/splits/train_aug_{config_name}.csv'
        )

        if not os.path.exists(aug_path):

            print("Generating augmented dataset...")

            augment_dataset(
                train_path,
                aug_path,
                technique_weights=weights
            )

        train_df = pd.read_csv(aug_path)

        config_dir = os.path.join(results_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)

        # =====================================================
        # Seed Loop
        # =====================================================

        for seed in SEEDS:

            print("\nRunning SEED:", seed)

            set_reproducibility(seed)

            print("Python:", platform.python_version())
            print("Torch:", torch.__version__)

            if torch.cuda.is_available():
                print("GPU:", torch.cuda.get_device_name(0))

            # =====================================================
            # Model Loop
            # =====================================================

            for model_class in model_classes:

                model_wrapper = model_class()

                model_name = model_wrapper.__class__.__name__.replace("Model","").lower()

                print(f"\nTraining {model_name.upper()} | Config={config_name} | Seed={seed}")

                # ------------------------------------------------
                # Train
                # ------------------------------------------------

                history, trained_model = train_model(
                    model_wrapper,
                    train_df,
                    val_df
                )

                if seed == SEEDS[0]:

                    plot_training_history(
                        history,
                        save_path=os.path.join(
                            config_dir,
                            f"{model_name}_training_history.png"
                        )
                    )

                # ------------------------------------------------
                # Evaluate
                # ------------------------------------------------

                metrics, preds, true_vals, probs = evaluate_model(
                    trained_model,
                    test_path
                )

                # ------------------------------------------------
                # F1 Scores
                # ------------------------------------------------

                metrics["f1_macro"] = f1_score(true_vals, preds, average="macro")
                metrics["f1_micro"] = f1_score(true_vals, preds, average="micro")
                metrics["f1_weighted"] = f1_score(true_vals, preds, average="weighted")

                # ------------------------------------------------
                # Recall
                # ------------------------------------------------

                metrics["recall_macro"] = recall_score(true_vals, preds, average="macro")
                metrics["recall_micro"] = recall_score(true_vals, preds, average="micro")

                classes = np.unique(true_vals)

                recall_per_class = recall_score(
                    true_vals,
                    preds,
                    average=None
                )

                for i, cls in enumerate(classes):
                    metrics[f"recall_class_{cls}"] = recall_per_class[i]

                # ------------------------------------------------
                # ROC AUC
                # ------------------------------------------------

                y_true_bin = label_binarize(true_vals, classes=classes)

                auc = roc_auc_score(
                    y_true_bin,
                    probs,
                    average="macro",
                    multi_class="ovr"
                )

                metrics["roc_auc_macro"] = auc

                # ------------------------------------------------
                # ROC Plot
                # ------------------------------------------------

                plot_roc_curve(
                    true_vals,
                    probs,
                    os.path.join(
                        config_dir,
                        f"{model_name}_seed{seed}_roc.png"
                    )
                )

                # ------------------------------------------------
                # Confusion Matrix
                # ------------------------------------------------

                plot_confusion_matrix(
                    true_vals,
                    preds,
                    save_path=os.path.join(
                        config_dir,
                        f"{model_name}_seed{seed}_cm.png"
                    )
                )

                # ------------------------------------------------
                # Metadata
                # ------------------------------------------------

                metrics["seed"] = seed
                metrics["model"] = model_name
                metrics["config"] = config_name

                save_metrics(
                    f"{model_name}_{config_name}_seed{seed}",
                    metrics,
                    save_dir=config_dir
                )

                all_results.append(metrics)

    # =====================================================
    # Save All Results
    # =====================================================

    results_df = pd.DataFrame(all_results)

    csv_path = os.path.join(results_dir, "all_results.csv")

    results_df.to_csv(csv_path, index=False)

    print("\nResults saved:", csv_path)

    # =====================================================
    # Final Statistics
    # =====================================================

    print("\n===== FINAL RESULTS (Mean ± Std) =====")

    metrics_to_print = [
        "accuracy",
        "f1_weighted",
        "f1_macro",
        "recall_macro",
        "roc_auc_macro"
    ]

    for model in results_df["model"].unique():

        print(f"\n{model.upper()}")

        model_df = results_df[results_df["model"] == model]

        for metric in metrics_to_print:

            mean = model_df[metric].mean()
            std  = model_df[metric].std()

            print(f"{metric:15} : {mean:.4f} ± {std:.4f}")

    # =====================================================
    # Recall Table
    # =====================================================

    recall_table = create_recall_table(results_df)

    recall_path = os.path.join(results_dir, "per_class_recall_table.csv")

    recall_table.to_csv(recall_path, index=False)

    print("\nPer-class recall table saved:", recall_path)


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    run_experiment()