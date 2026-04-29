from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

class DistilBERTModel:
    def __init__(self, num_labels=3):
        self.model_name = 'distilbert-base-uncased'
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=num_labels
        )
        
    def get_tokenizer(self):
        return self.tokenizer
        
    def get_model(self):
        return self.model
