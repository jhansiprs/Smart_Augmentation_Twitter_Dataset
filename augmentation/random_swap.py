import random

def random_swap(sentence, n=1):
    """
    Randomly swap positions of n word pairs.
    """
    words = sentence.split()
    new_words = words.copy()
    
    for _ in range(n):
        new_words = swap_word(new_words)
        
    return ' '.join(new_words)

def swap_word(new_words):
    if len(new_words) < 2:
        return new_words
        
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
            
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    return new_words
