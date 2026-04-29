from transformers import AlbertForSequenceClassification, AlbertTokenizer

class ALBERTModel:
    def __init__(self, num_labels=3):
        self.model_name = 'albert-base-v2'
        self.tokenizer = AlbertTokenizer.from_pretrained(self.model_name)
        self.model = AlbertForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=num_labels
        )
        
    def get_tokenizer(self):
        return self.tokenizer
        
    def get_model(self):
        return self.model
