import torch
from torch.utils.data import DataLoader
from training.train_single_model import SentimentDataset
from training.training_config import TrainingConfig
from evaluation.metrics import calculate_metrics, print_metrics
from evaluation.confusion_matrix import plot_confusion_matrix
import pandas as pd
import os
import numpy as np

def evaluate_model(model_wrapper, test_path, model_path=None):
    """
    Evaluate a model on the test set.
    """
    config = TrainingConfig
    device = torch.device(config.DEVICE)
    
    # Load model
    if model_path:
        model_wrapper.model = model_wrapper.model.from_pretrained(model_path)
        model_wrapper.tokenizer = model_wrapper.tokenizer.from_pretrained(model_path)
        
    model = model_wrapper.get_model()
    tokenizer = model_wrapper.get_tokenizer()
    model.to(device)
    model.eval()
    
    # Load data
    test_df = pd.read_csv(test_path)
    test_dataset = SentimentDataset(
        test_df['Cleaned_Sentence'].to_numpy(),
        test_df['Sentiment'].to_numpy(),
        tokenizer,
        config.MAX_SEQ_LEN
    )
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    
    predictions = []
    real_values = []
    probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            probs = torch.softmax(outputs.logits, dim=1)
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            real_values.extend(labels.cpu().tolist())
            probabilities.extend(probs.cpu().tolist())
            
    metrics = calculate_metrics(real_values, predictions)
    print_metrics(metrics)
    
    return metrics, predictions, real_values, probabilities
