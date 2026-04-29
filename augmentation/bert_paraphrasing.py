from transformers import BertTokenizer, BertForMaskedLM
import torch
import random
from nltk.corpus import stopwords

# Load model and tokenizer once
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

stop_words = set(stopwords.words('english'))

def bert_paraphrasing(sentence, mask_prob=0.15, top_k=5):
    """
    Use BERT MLM to generate paraphrases.
    """
    words = sentence.split()
    n_mask = max(1, int(len(words) * mask_prob))
    
    # Create masked sentence
    masked_words = words.copy()
    candidate_indices = [i for i, w in enumerate(words) if w.lower() not in stop_words and w.isalnum()]
    
    if not candidate_indices:
        return sentence
        
    mask_indices = random.sample(candidate_indices, min(len(candidate_indices), n_mask))
    
    for idx in mask_indices:
        masked_words[idx] = '[MASK]'
        
    masked_sentence = ' '.join(masked_words)
    
    # Predict
    inputs = tokenizer(masked_sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
        
    # Replace masks
    new_words = words.copy()
    for idx in mask_indices:
        token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(words[idx]))
        if not token_id:
             continue
        
        # Get top predictions
        mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0][mask_indices.index(idx)]
        predicted_token_ids = predictions[0, mask_token_index].topk(top_k).indices.tolist()
        
        # Choose a random prediction that is not the original word
        candidates = []
        for pid in predicted_token_ids:
            token = tokenizer.decode([pid]).strip()
            if token.lower() != words[idx].lower() and token.isalnum():
                candidates.append(token)
                
        if candidates:
            new_words[idx] = random.choice(candidates)
            
    return ' '.join(new_words)
