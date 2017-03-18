import numpy as np
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: python condense_weights.py <weights-file-base-path> <output-weights-path>"
        exit(1)

    base_path = sys.argv[1]
    W = np.vstack((np.load(base_path + "W_1.pkl"), np.load(base_path + "W_2.pkl")))
    W_tilde = np.vstack((np.load(base_path + "W_tilde_1.pkl"), np.load(base_path + "W_tilde_2.pkl")))

    null_embeddings_row = np.zeros(W.shape[1])  # zero emebeddings
    output_path = sys.argv[2]
    final_weights = np.vstack((W + W_tilde, null_embeddings_row))
    np.save(output_path, final_weights)
