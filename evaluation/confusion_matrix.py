import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot and save confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    labels = ['Negative', 'Neutral', 'Positive']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()
    plt.close()
