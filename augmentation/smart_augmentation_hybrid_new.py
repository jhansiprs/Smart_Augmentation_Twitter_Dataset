import pandas as pd
import random
import os
from tqdm import tqdm

# ==========================================================
# Augmentation techniques
# ==========================================================

from augmentation.synonym_replacement import synonym_replacement
from augmentation.random_insertion import random_insertion
from augmentation.random_deletion import random_deletion
from augmentation.bert_paraphrasing import bert_paraphrasing
from augmentation.duplicate_checker import is_duplicate

# ==========================================================
# Semantic similarity model
# ==========================================================

from sentence_transformers import SentenceTransformer, util

similarity_model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================================
# Semantic similarity filter
# ==========================================================

def semantic_filter(original, augmented):

    emb1 = similarity_model.encode(original, convert_to_tensor=True)
    emb2 = similarity_model.encode(augmented, convert_to_tensor=True)

    sim = util.cos_sim(emb1, emb2).item()

    if sim < 0.65:
        return False

    if sim > 0.97:
        return False

    return True


# ==========================================================
# Apply augmentation
# ==========================================================

def apply_augmentation(sentence, technique):

    try:

        if technique == "synonym_replacement":
            return synonym_replacement(sentence, n=1)

        elif technique == "random_insertion":
            return random_insertion(sentence, n=1)

        elif technique == "random_deletion":
            return random_deletion(sentence, p=0.05)

        elif technique == "bert_paraphrasing":
            return bert_paraphrasing(sentence)

    except Exception:
        return None

    return None


# ==========================================================
# Class-aware hybrid technique selection
# ==========================================================

def choose_technique(sentiment_label):

    # Negative class: preserve sentiment polarity
    if sentiment_label == "positive":

        technique_weights = {
            "bert_paraphrasing": 0.45,
            "random_deletion": 0.05,
            "random_insertion": 0.25,
            "synonym_replacement": 0.25
        }


    # Neutral class: increase contextual diversity
    elif sentiment_label == "neutral":

        technique_weights = {
            "bert_paraphrasing": 0.6,
            "random_insertion": 0.10,
            "random_deletion": 0.05,
            "synonym_replacement": 0.25
        }


    techniques = list(technique_weights.keys())
    weights = list(technique_weights.values())

    return random.choices(techniques, weights=weights, k=1)[0]


# ==========================================================
# Augment one class
# ==========================================================

def augment_class(df_class, target_count, existing_sentences, sentiment_label):

    current_count = len(df_class)
    needed = target_count - current_count

    if needed <= 0:
        return []

    print(f"\nAugmenting {sentiment_label} -> Need {needed} samples")

    new_samples = []

    source_sentences = df_class["Cleaned_Sentence"].dropna().tolist()

    pbar = tqdm(total=needed)

    attempts = 0
    max_attempts = needed * 30

    while len(new_samples) < needed and attempts < max_attempts:

        attempts += 1

        sentence = random.choice(source_sentences)

        length = len(sentence.split())

        if length < 5 or length > 20:
            continue

        technique = choose_technique(sentiment_label)

        aug_sentence = apply_augmentation(sentence, technique)

        if not aug_sentence:
            continue

        if aug_sentence.strip() == sentence.strip():
            continue

        aug_len = len(aug_sentence.split())

        if aug_len < 5 or aug_len > 25:
            continue

        if not semantic_filter(sentence, aug_sentence):
            continue

        if is_duplicate(aug_sentence, existing_sentences):
            continue

        new_samples.append({
            "Sentence": sentence,
            "Cleaned_Sentence": aug_sentence,
            "Sentiment": sentiment_label,
            "Is_Augmented": True,
            "Technique": technique
        })

        existing_sentences.add(aug_sentence)

        pbar.update(1)

    pbar.close()

    return new_samples


# ==========================================================
# Main augmentation pipeline
# ==========================================================

def augment_dataset(input_path, output_path):

    df = pd.read_csv(input_path)

    df["Is_Augmented"] = False
    df["Technique"] = "original"

    class_counts = df["Sentiment"].value_counts()

    print("\nOriginal Distribution:")
    print(class_counts)

    target_samples_per_class = class_counts.max()

    print("\nTarget samples per class:", target_samples_per_class)

    pos_df = df[df["Sentiment"] == "positive"]
    neu_df = df[df["Sentiment"] == "neutral"]
    neg_df = df[df["Sentiment"] == "negative"]

    existing_sentences = set(df["Cleaned_Sentence"].dropna().values)

    augmented_samples = []

    # Only augment minority classes
    augmented_samples.extend(
        augment_class(neu_df, target_samples_per_class, existing_sentences, "neutral")
    )

    augmented_samples.extend(
        augment_class(pos_df, target_samples_per_class, existing_sentences, "positive")
    )

    if augmented_samples:

        aug_df = pd.DataFrame(augmented_samples)
        final_df = pd.concat([df, aug_df], ignore_index=True)

    else:

        final_df = df.copy()

    # ==============================
    # 🔍 Diversity check (ADD HERE)
    # ==============================
    df_aug = final_df[final_df["Is_Augmented"] == True]

    if len(df_aug) > 0:
        diversity_ratio = df_aug["Cleaned_Sentence"].nunique() / len(df_aug)
        print("\nAugmented Data Diversity Ratio:", round(diversity_ratio, 4))
    
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nTotal Augmented Samples:", len(augmented_samples))

    print("\nFinal Distribution:")
    print(final_df["Sentiment"].value_counts())

    if augmented_samples:

        print("\nTechnique Usage:")
        print(pd.DataFrame(augmented_samples)["Technique"].value_counts())

    final_df.to_csv(output_path, index=False)

    print("\nSaved augmented dataset to:", output_path)


# ==========================================================
# Run script
# ==========================================================

if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.dirname(__file__))

    train_path = os.path.join(base_dir, "data/splits/train.csv")
    output_path = os.path.join(base_dir, "data/splits/train_augmented_hybrid.csv")

    if os.path.exists(train_path):

        augment_dataset(train_path, output_path)

    else:

        print("Train split not found:", train_path)