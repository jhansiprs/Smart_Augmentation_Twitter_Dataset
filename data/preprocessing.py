import pandas as pd
import re
import os
import emoji

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove mentions
    text = re.sub(r'@\w+', '', text)

    # Remove hashtags but keep the word
    text = re.sub(r'#', '', text)

    # Convert emojis to words
    text = emoji.demojize(text, delimiters=(" ", " "))

    # Remove special characters
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text.lower()


def load_and_preprocess_data(file_path):

    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format")

    # Standardize column names
    if 'Review' in df.columns:
        df.rename(columns={'Review': 'Sentence'}, inplace=True)
    if 'text' in df.columns:
        df.rename(columns={'text': 'Sentence'}, inplace=True)

    if 'Class' in df.columns:
        df.rename(columns={'Class': 'Sentiment'}, inplace=True)
    if 'Label' in df.columns:
        df.rename(columns={'Label': 'Sentiment'}, inplace=True)
    if 'airline_sentiment' in df.columns:
        df.rename(columns={'airline_sentiment': 'Sentiment'}, inplace=True)

    # Drop missing values
    df.dropna(subset=['Sentence', 'Sentiment'], inplace=True)

    df['Sentiment'] = df['Sentiment'].astype(str).str.lower().str.strip()

    # Clean text
    df['Cleaned_Sentence'] = df['Sentence'].apply(clean_text)

    # Remove empty rows
    df = df[df['Cleaned_Sentence'].str.len() > 0]

    # Remove duplicates
    df.drop_duplicates(subset=['Cleaned_Sentence'], inplace=True)

    print("\nFinal Cleaned Dataset Size:", len(df))
    print("\nClass Distribution:")
    print(df["Sentiment"].value_counts())

    return df


if __name__ == "__main__":

    data_path = os.path.join(os.path.dirname(__file__), 'Tweets.csv')

    if os.path.exists(data_path):
        df = load_and_preprocess_data(data_path)

        output_path = os.path.join(os.path.dirname(__file__), 'cleaned_dataset.csv')
        df.to_csv(output_path, index=False)

        print(f"\nSaved cleaned data to {output_path}")
    else:
        print("Data file not found.")