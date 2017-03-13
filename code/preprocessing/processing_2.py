#main processing file for vocab/word embeddings
import sys
from embeddings_train_2 import GloveTrainer
import time
import argparse

def make_corpus(path):
    corpus = []
    with open(path, 'r') as f:
        for line in f:
            corpus.append(line)
    return corpus

def print_time_elapsed(start_time, func_name):
    time_elapsed = time.clock() - start
    print "{} completed: Time elapsed - {}s".format(func_name, time_elapsed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains GloVe embeddings on a corpus")
    parser.add_argument('input_file', help="input file path")
    parser.add_argument('weights_file', help="weights output file path")
    parser.add_argument('vocab_file', help="vocab output file path")
    parser.add_argument('-s', '--intermediate_path', help="optional file path for saving intermediate results")
    parser.add_argument('-i', '--iterations', default=100, type=int, help="num iterations to run for")
    parser.add_argument('-b', '--batch_size', default=1, type=int, help="size of minibatches (-1 for entire batch)")
    parser.add_argument('-m', '--opt_method', default="adagrad", help="optimization method: either vanilla \
        gradient descent ('grad') or adagrad ('adagrad')")
    parser.add_argument('-c', '--num_cores', default=1, type=int, help="number of cores to use")
    parser.add_argument('-a', '--alpha', default=0.75, type=float, help="alpha for weighting function")
    parser.add_argument('-x', '--x_max', default=100, type=float, help="x_max for weighting function")
    parser.add_argument('-v', '--v_dim', default=200, type=int, help="number of dimensions for embeddings")
    parser.add_argument('-l', '--learning_rate', default=0.05, type=float, help="initial learning rate")
    args = parser.parse_args()

    start = time.clock()
    corpus = make_corpus(args.input_file)
    print_time_elapsed(start, "Corpus creation")

    start = time.clock()
    trainer = GloveTrainer(corpus)
    trainer.make_vocab_and_cooccurrence_mat()
    print_time_elapsed(start, "Vocab and cooccurrence matrix creation")

    trainer.export_vocab(args.vocab_file)

    start = time.clock()
    trainer.train(iters=args.iterations, v_dim=args.v_dim, alpha=args.alpha, x_max=args.x_max, \
        batch_size=args.batch_size, num_cores=args.num_cores, learning_rate=args.learning_rate, \
        save_intermediate_path=args.intermediate_path, opt_method=args.opt_method)
    print_time_elapsed(start, "Training")

    trainer.export_weights(args.weights_file)