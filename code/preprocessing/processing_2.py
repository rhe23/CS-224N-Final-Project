#main processing file for vocab/word embeddings
import sys
from embeddings_train_2 import GloveTrainer as CPUGloveTrainer
from embeddings_train_gpu import GloveTrainer as GPUGloveTrainer
import time
import argparse
from operator import itemgetter

def make_corpus(path):
    corpus = []
    with open(path, 'r') as f:
        for line in f:
            corpus.append(line)
    return corpus

def print_time_elapsed(start_time, func_name):
    time_elapsed = time.clock() - start
    print "{} completed: Time elapsed - {}s".format(func_name, time_elapsed)

def train_glove_embeddings(args):
    start = time.clock()
    corpus = make_corpus(args.input_file)
    print_time_elapsed(start, "Corpus creation")

    start = time.clock()
    if args.use_gpu:
        trainer = GPUGloveTrainer(corpus)
    else:
        trainer = CPUGloveTrainer(corpus)
    trainer.make_vocab_and_cooccurrence_mat()
    print_time_elapsed(start, "Vocab and cooccurrence matrix creation")

    trainer.export_vocab(args.vocab_file)

    start = time.clock()
    trainer.train(iters=args.iterations, v_dim=args.v_dim, alpha=args.alpha, x_max=args.x_max, \
        batch_size=args.batch_size, learning_rate=args.learning_rate, \
                  save_intermediate_path=args.intermediate_path, opt_method=args.opt_method, saved_weights_path=args.saved_weights_path)
    print_time_elapsed(start, "Training")

    trainer.export_weights(args.weights_file)

def eval_plot(args):
    raise NotImplementedError()

def eval_sim(args):
    MIN_SIM_WORDS = 1  # minimum number of similar words to display
    MAX_SIM_WORDS = 100  # maximum number of similar words to display

    print "Loading trained data..."
    vocab = np.load(args.trained_vocab_file)
    embeddings = np.load(args.trained_weights_file)
    
    base_word = ""
    num_sim_words = 0

    while True:
        base_word = input("Enter token to find similar tokens to (-1 to exit): ").strip()
        if base_word == "-1":
            return
        if base_word not in vocab:
            print "Given token is not in vocab! Try again."

        while True:
            num_sim_words = input("Enter number of similar tokens to find: ").strip()
            try:
                num_sim_words = int(num_sim_words)
                if num_sim_words < MIN_SIM_WORDS or num_sim_words > MAX_SIM_WORDS:
                    print "Please enter a positive integer between {} and {}".format(MIN_SIM_WORDS, MAX_SIM_WORDS)
                else:
                    break
            except ValueError:
                print "Please enter a positive integer between {} and {}".format(MIN_SIM_WORDS, MAX_SIM_WORDS)

        print "Computing..."
        base_embeddings = embeddings[vocab[base_word]]
        word_to_cosine_sim = {}
        for word in vocab:
            if word == base_word:
                continue
            cosine_sim = np.dot(base_embeddings, embeddings[vocab[base_word]])
            word_to_cosine_sim[word] = cosine_sim

        most_similar = sorted(word_to_cosine_sim.iteritems(), key=itemgetter(1), reverse=True)[:num_sim_words]

        print "Tokens most similar to {}:".format(base_word)
        for word in most_similar:
            print "Token: {}, cosine similarity: {}".format(word, word_to_cosine_sim[word])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or perform diagnostics on GloVe embeddings")
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('train', help='Train GloVe embeddings on a corpus')
    command_parser.add_argument('input_file', help="input file path")
    command_parser.add_argument('weights_file', help="weights output file path")
    command_parser.add_argument('vocab_file', help="vocab output file path")
    command_parser.add_argument('-s', '--intermediate_path', help="optional file path for saving intermediate results")
    command_parser.add_argument('-p', '--saved_weights_path', help="file path for using saved weights")
    command_parser.add_argument('-i', '--iterations', default=100, type=int, help="num iterations to run for")
    command_parser.add_argument('-b', '--batch_size', default=1, type=int, help="size of minibatches (-1 for entire batch)")
    command_parser.add_argument('-m', '--opt_method', default="adagrad", help="optimization method: either vanilla \
        gradient descent ('grad') or adagrad ('adagrad')")
    command_parser.add_argument('-a', '--alpha', default=0.75, type=float, help="alpha for weighting function")
    command_parser.add_argument('-x', '--x_max', default=100, type=float, help="x_max for weighting function")
    command_parser.add_argument('-v', '--v_dim', default=200, type=int, help="number of dimensions for embeddings")
    command_parser.add_argument('-l', '--learning_rate', default=0.05, type=float, help="initial learning rate")
    command_parser.add_argument('-g', '--use_gpu', action='store_true', help="use the GPU to train")
    command_parser.set_defaults(func=train_glove_embeddings)

    command_parser = subparsers.add_parser('eval_plot', help='Plot a subset of the trained embeddings next to each other')
    command_parser.add_argument('trained_weights_file', help="trained weights file path")
    command_parser.add_argument('trained_vocab_file', help="trained vocab file path")
    command_parser.set_defaults(func=eval_plot)

    command_parser = subparsers.add_parser('eval_sim', help='Find the most similar tokens to a given token')
    command_parser.add_argument('trained_weights_file', help="trained weights file path")
    command_parser.add_argument('trained_vocab_file', help="trained vocab file path")
    command_parser.set_defaults(func=eval_sim)

    args = parser.parse_args()
    if args.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        args.func(args)
    

