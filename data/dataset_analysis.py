import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# ==========================================================
# MTLD Function
# ==========================================================
def mtld(text, threshold=0.72):

    words = text.split()
    factors = 0
    types = set()
    token_count = 0

    for word in words:
        token_count += 1
        types.add(word)
        ttr = len(types) / token_count

        if ttr <= threshold:
            factors += 1
            types = set()
            token_count = 0

    if token_count != 0:
        factors += (1 - ttr) / (1 - threshold)

    if factors == 0:
        return 0

    return len(words) / factors


# ==========================================================
# Main Analysis
# ==========================================================
def analyze_dataset(file_path):

    df = pd.read_csv(file_path)

    print("\n===== DATASET STATISTICS =====")
    print(f"Total Reviews: {len(df)}")

    # ------------------------------------------------------
    # Class Distribution
    # ------------------------------------------------------
    class_counts = df['Sentiment'].value_counts()
    class_percent = class_counts / len(df) * 100

    print("\nClass Distribution:")
    print(class_counts)

    print("\nClass Percentages:")
    print(class_percent.round(2))

    # Imbalance Ratio
    imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"\nImbalance Ratio (Majority/Minority): {imbalance_ratio:.3f}")

    # ------------------------------------------------------
    # Duplicate Analysis
    # ------------------------------------------------------
    duplicates = df[df.duplicated(subset=['Cleaned_Sentence'], keep=False)]
    num_duplicates = len(duplicates)
    unique_sentences = df['Cleaned_Sentence'].nunique()

    print("\nDuplicate Analysis:")
    print(f"Total Duplicates (including originals): {num_duplicates}")
    print(f"Unique Sentences: {unique_sentences}")
    print(f"Duplicate Percentage: {num_duplicates / len(df) * 100:.2f}%")

    # ------------------------------------------------------
    # Length Distribution
    # ------------------------------------------------------
    df['Length'] = df['Cleaned_Sentence'].apply(lambda x: len(str(x).split()))

    print("\nReview Length Statistics:")
    print(df['Length'].describe())

    # ------------------------------------------------------
    # Lexical Richness
    # ------------------------------------------------------
    full_text = " ".join(df["Cleaned_Sentence"].astype(str))
    words = full_text.split()
    vocab = set(words)

    ttr = len(vocab) / len(words)
    mtld_score = mtld(full_text)

    print("\nLexical Richness:")
    print(f"Vocabulary Size: {len(vocab)}")
    print(f"Type-Token Ratio (TTR): {ttr:.4f}")
    print(f"MTLD: {mtld_score:.3f}")

    # ------------------------------------------------------
    # Visualization
    # ------------------------------------------------------
    plt.figure(figsize=(6,4))
    class_counts.plot(kind='bar')
    plt.title("Class Distribution")
    plt.ylabel("Count")
    plt.tight_layout()

    output_plot = os.path.join(os.path.dirname(file_path), "Class_Distribution.png")
    plt.savefig(output_plot)
    plt.close()

    print(f"\nSaved class distribution plot to {output_plot}")

    return df


if __name__ == "__main__":

    data_path = os.path.join(os.path.dirname(__file__), 'cleaned_dataset.csv')

    if os.path.exists(data_path):
        analyze_dataset(data_path)
    else:
        print("Cleaned dataset not found. Run preprocessing.py first.")