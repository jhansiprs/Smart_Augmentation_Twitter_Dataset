import pandas as pd
from sklearn.model_selection import train_test_split
import os


def print_ratio(df, name):
    counts = df['Sentiment'].value_counts()
    ratio = counts.max() / counts.min()
    print(f"{name} Imbalance Ratio: {ratio:.2f}")


def split_dataset(input_path, output_dir, test_size=0.2, val_size=0.2, random_state=42):

    df = pd.read_csv(input_path)

    # Train+Val vs Test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['Sentiment'],
        random_state=random_state
    )

    # Train vs Val
    val_relative_size = val_size / (1 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        stratify=train_val_df['Sentiment'],
        random_state=random_state
    )

    print("\n===== Split Sizes =====")
    print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    print("\n===== Class Distribution =====")

    print("\nTrain:")
    print(train_df['Sentiment'].value_counts())
    print_ratio(train_df, "Train")

    print("\nValidation:")
    print(val_df['Sentiment'].value_counts())
    print_ratio(val_df, "Validation")

    print("\nTest:")
    print(test_df['Sentiment'].value_counts())
    print_ratio(test_df, "Test")

    # Save
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print(f"\nSaved splits to {output_dir}")


if __name__ == "__main__":

    data_path = os.path.join(os.path.dirname(__file__), 'cleaned_dataset.csv')
    output_dir = os.path.join(os.path.dirname(__file__), 'splits')

    if os.path.exists(data_path):
        split_dataset(data_path, output_dir)
    else:
        print("Run preprocessing first.")