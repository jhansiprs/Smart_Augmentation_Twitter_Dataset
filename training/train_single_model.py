import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, recall_score
import numpy as np
import os
from tqdm import tqdm
from training.training_config import TrainingConfig

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Map labels to integers
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.label_map[label], dtype=torch.long)
        }

def train_model(model_wrapper, train_df, val_df, config=TrainingConfig):
    model = model_wrapper.get_model()
    tokenizer = model_wrapper.get_tokenizer()
    device = torch.device(config.DEVICE)
    
    model.to(device)
    
    train_dataset = SentimentDataset(
        train_df['Cleaned_Sentence'].to_numpy(),
        train_df['Sentiment'].to_numpy(),
        tokenizer,
        config.MAX_SEQ_LEN
    )
    
    val_dataset = SentimentDataset(
        val_df['Cleaned_Sentence'].to_numpy(),
        val_df['Sentiment'].to_numpy(),
        tokenizer,
        config.MAX_SEQ_LEN
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    best_accuracy = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(config.EPOCHS):
        print(f'Epoch {epoch + 1}/{config.EPOCHS}')
        print('-' * 10)
        
        # Training
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_losses.append(loss.item())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIPPING)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        print(f"Train Loss: {avg_train_loss}")
        
        # Validation
        model.eval()
        val_losses = []
        predictions = []
        real_values = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                val_losses.append(loss.item())
                
                _, preds = torch.max(outputs.logits, dim=1)
                predictions.extend(preds.cpu().tolist())
                real_values.extend(labels.cpu().tolist())
                
        avg_val_loss = np.mean(val_losses)
        val_acc = accuracy_score(real_values, predictions)
        val_f1 = f1_score(real_values, predictions, average='weighted')
        
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Val Loss: {avg_val_loss}")
        print(f"Val Accuracy: {val_acc}")
        print(f"Val F1: {val_f1}")
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            # Save best model
            save_path = os.path.join(config.MODEL_SAVE_DIR, f"{model_wrapper.__class__.__name__}_best")
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Saved best model to {save_path}")
            
    return history, model_wrapper
