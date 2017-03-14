import re

def append_start_end_tokens(tokens):
    #append a start and end token to a list of tokens
    tokens.insert(0, "<start>")
    tokens.insert(len(tokens), "<end>")
    return tokens

def tokenizer(token_list, punc = True):
    return [tokenize(punc, s) for s in token_list]

def tokenize(s, preserve_punc = True):
    #tokenize a string to a list of tokens
    if preserve_punc ==False:
        return re.findall(r'[a-zA-Z]+|[a-zA-Z]+[\W]*[a-zA-Z]*', s)

    l= filter(lambda x: x.strip(), re.findall(r"[-\w]+'[-\w]|[-\w]+|(?<=\s)*\W+(?=\s+)|\W+(?=[a-zA-Z0-9])|(?<=\w)+\W", s))
    return [x.lstrip().rstrip() for x in l ]

def tokenize_2(s, preserve_punc = True):
    if preserve_punc == False:
        token_regex = r"[a-zA-Z]+|[a-zA-Z]+[\W]*[a-zA-Z]*"
        tokens = re.findall(token_regex, s)
    else:
        token_regex = r"[-\w]+'[-\w]|[-\w]+|(?<=\s)*\W+(?=\s+)|\W+(?=[a-zA-Z0-9])|(?<=\w)+\W"
        tokens = filter(lambda x: x.strip(), re.findall(token_regex, s))
    tokens = [x.strip() for x in tokens]
    # append start and end tokens 
    tokens.append("<start>")
    tokens.append("<end>")
    return tokens
