#main processing file for vocab/word embeddings
import os, sys
from embeddings_train import Vocab_Builder, make_embeddings
import time

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

def print_time_elapsed(start_time, func_name):
    time_elapsed = time.clock() - start
    print "{} completed: Time elapsed - {}s".format(func_name, time_elapsed)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print "Usage: python processing.py <input-file> <weights-output-file> <vocab-output-file> <num-iters>"
        exit(1)

    input_file = sys.argv[1]
    weights_output_file = sys.argv[2]
    vocab_output_file = sys.argv[3]
    num_iters = int(sys.argv[4])

    start = time.clock()
    corpus = make_corpus(input_file)
    print_time_elapsed(start, "make_corpus")
    start = time.clock()
    vocab = make_vocab(input_file, vocab_output_file)
    print_time_elapsed(start, "make_vocab")
    #embeddings:
    glove_trainer = make_embeddings(corpus = corpus, vocab = vocab)
    start = time.clock()
    glove_trainer.make_cooccurrence_mat()
    print_time_elapsed(start, "make_embeddings")
    glove_trainer.train(iters = num_iters, v_dim = 200, a = 0.75, x_max = 100, save_intermediate_path=weights_output_file)
    print_time_elapsed(start, "Training")
    glove_trainer.save_weights(weights_output_file)
