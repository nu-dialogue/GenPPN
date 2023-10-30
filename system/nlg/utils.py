import re

def remove_ws_before_punctuation(text):
    tokens = text.split()
    new_tokens = []
    for token in tokens:
        m = re.match(r'\W|n\'', token) # " ."  " ,"  " ?"  " n't"  " 're"
        if m is None or not new_tokens:
            new_tokens.append(token)
        else:
            new_tokens[-1] += token
    return " ".join(new_tokens)