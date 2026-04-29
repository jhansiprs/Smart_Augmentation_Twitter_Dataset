from Levenshtein import distance as levenshtein_distance

def is_duplicate(sentence, existing_sentences, threshold=0):
    """
    Check if sentence is a duplicate of any existing sentence.
    Threshold 0 means exact match.
    """
    if sentence in existing_sentences:
        return True
    return False

def is_near_duplicate(sentence, existing_sentences, distance_threshold=3):
    """
    Check if sentence is a near duplicate using Levenshtein distance.
    This is computationally expensive for large datasets.
    """
    for existing in existing_sentences:
        if levenshtein_distance(sentence, existing) < distance_threshold:
            return True
    return False
