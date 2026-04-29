from transformers import RobertaForSequenceClassification, RobertaTokenizer

class RoBERTaModel:
    def __init__(self, num_labels=3):
        self.model_name = 'roberta-base'
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=num_labels
        )
        
    def get_tokenizer(self):
        return self.tokenizer
        
    def get_model(self):
        return self.model
