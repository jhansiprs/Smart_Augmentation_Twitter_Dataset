from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # Per-class metrics
    precision_cls, recall_cls, f1_cls, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1, 2])
    
    metrics = {
        'accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'negative_recall': recall_cls[0],
        'neutral_recall': recall_cls[1],
        'positive_recall': recall_cls[2],
        'negative_f1': f1_cls[0],
        'neutral_f1': f1_cls[1],
        'positive_f1': f1_cls[2]
    }
    
    return metrics

def print_metrics(metrics):
    print("\nPerformance Metrics:")
    print("-" * 20)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Weighted F1: {metrics['f1_weighted']:.4f}")
    print(f"Negative Recall: {metrics['negative_recall']:.4f}")
    print(f"Neutral Recall: {metrics['neutral_recall']:.4f}")
    print(f"Positive Recall: {metrics['positive_recall']:.4f}")
