#main processing file for vocab/word embeddings

from embeddings_train import Vocab_Builder

path = ""

vocab_builder = Vocab_Builder

with open(path, 'r') as f:
    for line in f:
        vocab_builder.add(line)

vocab_builder.update_vocab()
