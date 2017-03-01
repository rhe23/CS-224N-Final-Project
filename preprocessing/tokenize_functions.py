import re

def append_start_end_tokens(tokens):
    #append a start and end token to a list of tokens
    tokens.insert(0, "<start>")
    tokens.insert(len(tokens), "<end>")
    return tokens

def tokenizer(token_list, punc = True):
    return [tokenize(punc, s) for s in toke_list]

def tokenize(s, preserve_punc = True):
    #tokenize a string to a list of tokens
    if preserve_punc ==False:
        return re.findall(r'[a-zA-Z]+[\W]*[a-zA-Z]*', s)

    return re.findall(r'[a-zA-Z]+[\W]*[a-zA-Z]*|(?<=\s)*\W+(?=\s+)', s)




