#main processing file for vocab/word embeddings
import os
from embeddings_train import Vocab_Builder, make_embeddings

def make_corpus(path):
    corpus = []
    with open(path, 'r') as f:
        for line in f:
            corpus.append(line)

    return corpus


os.chdir('../..')
filename = "data/tifu_201509_titles"


vocab_builder = Vocab_Builder()
corpus = []
with open(filename, 'r') as f:
    for line in f:
        vocab_builder.add(line)
        corpus.append(line)

vocab_builder.update_vocab()
glove_embedder = make_embeddings(corpus=corpus, vocab=vocab_builder.get_vocab())
glove_embedder.make_cooccurance_mat()
co_occur = glove_embedder.get_coocurance_mat()
glove_embedder.train(iters=500, v_dim=50, a=0.75, x_max=100)
W = glove_embedder.get_weights()