import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def plot_model_comparison(results, metric='accuracy', save_path=None):
    """
    Plot comparison of multiple models.
    results: dict {model_name: metric_value}
    """
    df = pd.DataFrame(list(results.items()), columns=['Model', metric.capitalize()])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y=metric.capitalize(), data=df)
    plt.title(f'Model Comparison - {metric.capitalize()}')
    plt.ylim(0, 1.0)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_training_history(history, save_path=None):
    """
    Plot training loss and validation accuracy.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_acc'], 'g-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_multimetric_comparison(df, metrics=['accuracy', 'f1_weighted', 'recall_weighted'], save_path=None):
    """
    Plot comparison of multiple metrics across experiments.
    df: DataFrame containing metrics
    """
    # Melt the dataframe for seaborn
    df_melted = df.melt(id_vars=['experiment'], value_vars=metrics, var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='experiment', y='Score', hue='Metric', data=df_melted)
    plt.title('Performance Comparison across Experiments')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_class_recall_comparison(df, class_names=['negative', 'neutral', 'positive'], save_path=None):
    """
    Plot recall for each class across experiments.
    """
    # Construct column names based on how they are stored (e.g., recall_negative, recall_neutral)
    # Assuming metrics are flattened as 'recall_negative', etc.
    
    # Construct column names based on how they are stored (e.g., negative_recall, neutral_recall)
    # The metrics are stored as 'negative_recall', 'neutral_recall', 'positive_recall'
    
    metric_cols = [f'{c}_recall' for c in class_names]
    
    # Check if columns exist
    valid_cols = [c for c in metric_cols if c in df.columns]
    
    if not valid_cols:
        print(f"No class recall columns found in DataFrame. Expected: {metric_cols}. Found: {df.columns.tolist()}")
        return

    df_melted = df.melt(id_vars=['experiment'], value_vars=valid_cols, var_name='Class', value_name='Recall')
    # Clean up class names for legend
    df_melted['Class'] = df_melted['Class'].str.replace('_recall', '').str.capitalize()
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='experiment', y='Recall', hue='Class', data=df_melted)
    plt.title('Per-Class Recall Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()
