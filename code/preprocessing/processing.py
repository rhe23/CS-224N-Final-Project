#main processing file for vocab/word embeddings
import os, sys
from embeddings_train import Vocab_Builder, make_embeddings

def make_corpus(path):
    corpus = []
    with open(path, 'r') as f:
        for line in f:
            corpus.append(line)
    return corpus

def make_vocab(input_path, output_path):
    vocab_builder = Vocab_Builder()
    with open(input_path, 'r') as f:
        for line in f:
            vocab_builder.add(line)
    vocab_builder.export(output_path)
    return vocab_builder.get_vocab()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print "Usage: python processing.py <input-file> <weights-output-file> <vocab-output-file> <num-iters>"
        exit(1)

    input_file = sys.argv[1]
    weights_output_file = sys.argv[2]
    vocab_output_file = sys.argv[3]
    num_iters = int(sys.argv[4])

    corpus = make_corpus(input_file)
    vocab = make_vocab(input_file, vocab_output_file)
    #embeddings:
    glove_trainer = make_embeddings(corpus = corpus, vocab = vocab)
    glove_trainer.make_cooccurrence_mat()
    glove_trainer.train(iters = num_iters, v_dim = 25, a = 0.75, x_max = 100)
    glove_trainer.save_weights(weights_output_file)