#main processing file for vocab/word embeddings
import os, sys
from embeddings_train import Vocab_Builder, make_embeddings

def make_corpus(path):
    corpus = []
    with open(path, 'r') as f:
        for line in f:
            corpus.append(line)

    return corpus

if __name__ == "__main__":
	if len(sys.argv != 4):
		print "Usage: python processing.py <input-file> <weights-output-file> <num-iters>"
	
	input_file = sys.argv[1]
	weights_output_file = sys.argv[2]
	num_iters = int(sys.argv[3])

	vocab_builder = Vocab_Builder()
	corpus = []
	with open(input_file, 'r') as f:
	    for line in f:
	        vocab_builder.add(line)
	        corpus.append(line)

	#vocab_builder.update_vocab()
	vocab_builder.export("data/")

	glove_embedder = make_embeddings(corpus=corpus, vocab=vocab_builder.get_vocab())
	glove_embedder.make_cooccurance_mat()
	co_occur = glove_embedder.get_coocurance_mat()
	glove_embedder.train(iters=num_iters, v_dim=50, a=0.75, x_max=100)
	W = glove_embedder.get_weights()
	glove_embedder.save_weights(weights_output_file)
