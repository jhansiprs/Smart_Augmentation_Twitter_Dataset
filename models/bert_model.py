from transformers import BertForSequenceClassification, BertTokenizer

class BERTModel:
    def __init__(self, num_labels=3):
        self.model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=num_labels
        )
        
    def get_tokenizer(self):
        return self.tokenizer
        
    def get_model(self):
        return self.model
