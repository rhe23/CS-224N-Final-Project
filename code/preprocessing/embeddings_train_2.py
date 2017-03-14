from tokenize_functions import tokenize_2
import cPickle
import random
import numpy as np
import time
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import scipy.sparse
import sys
from operator import itemgetter

class GloveTrainer:
    #Class for training glove vectors
    def __init__(self,  corpus = []):
        self.corpus = corpus  # corpus is assumed to be a list of individual sentences
        self.cooccurrence_mat = None  # in sparse dok (dictionary-of-keys) format
        self.nonzeros = None  # nonzero keys/indices of the cooccurrence matrix
        self.vocab = None
        self.V = 0  # size of vocab

        self.W = None
        self.W_tilde = None
        self.gradW = None
        self.b = None
        self.b_tilde = None
        self.gradb = None

        self.f_x = None  # weighting function

        # Cumulative squared gradient sums for AdaGrad
        self.G_W = None
        self.G_W_tilde = None
        self.G_b = None
        self.G_b_tilde = None

    def get_vocab(self):
        assert self.vocab, "Haven't built vocab yet!"
        return self.vocab

    def export_vocab(self, path):
        with open(path, 'wb') as file:
            cPickle.dump(self.vocab, file, protocol=2)

    def get_cooccurrence_mat(self):
        assert self.cooccurrence_mat, "Haven't built cooccurrence matrix yet!"
        return self.cooccurrence_mat

    def get_weights(self):
        assert self.W, "Haven't trained embeddings yet!"
        return self.W, self.W_tilde

    def export_weights(self, path):
        with open(path + 'W_1.pkl', 'wb') as file:
            np.save(file, self.W[:self.V / 2])
        with open(path + 'W_2.pkl', 'wb') as file:
            np.save(file, self.W[self.V / 2:])
        with open(path + 'W_tilde_1.pkl', 'wb') as file:
            np.save(file, self.W_tilde[:self.V / 2])
        with open(path + 'W_tilde_2.pkl', 'wb') as file:
            np.save(file, self.W_tilde[self.V / 2:])

    def weight_func(x_ij, a, x_max):
        #x_ij is the occcurencc count, alpha is the scaling param, x_max is the cutoff
        if x_ij < x_max:
            return (x_ij / float(x_max)) ** a
        return 1

    def get_batches(data_size, batch_size):
        #returns a list of indices for each batch of data during optimization
        data_indices = np.arange(data_size)
        for i in xrange(0, data_size, batch_size):
            yield data_indices[i:i+batch_size]

    def make_vocab_and_cooccurrence_mat(self):
        model = CountVectorizer(ngram_range=(1, 1), tokenizer=tokenize_2)
        X = model.fit_transform(self.corpus)

        self.vocab = model.vocabulary_
        self.V = len(self.vocab)

        M = X.T.dot(X)  # this count includes a word cooccurring with itself at the same position
                        # M is V x V, where V is the size of the vocab
        word_freq = np.asarray(X.sum(axis=0)).reshape(-1)  # 1 x V matrix representing frequencies of each word
        M.setdiag(M.diagonal() - word_freq)  # to remove the effect of words cooccurring with themselves

        nonzeros_i, nonzeros_j, vals = scipy.sparse.find(M)
        nonzeros_i = nonzeros_i.tolist()
        nonzeros_j = nonzeros_j.tolist()
        vals = vals.tolist()
        self.nonzeros = zip(nonzeros_i, nonzeros_j)

        self.cooccurrence_mat = defaultdict(lambda :dict())
        for k in range(len(self.nonzeros)):
            self.cooccurrence_mat[nonzeros_i[k]][nonzeros_j[k]] = vals[k]

    # train on minibatches
    # opt_method can be adagrad or grad (vanilla gradient descent)
    def minimize_batch(self, batch_size, eta, opt_method):
        #using a batch_gradient descent method
        cost = 0.0
        eps = 1e-8  # for use in adagrad
        np.random.shuffle(self.nonzeros)
        for lower_index in range(0, len(self.nonzeros), batch_size):
            upper_index = min(lower_index + batch_size, len(self.nonzeros))  # index of first element after the end of this batch
            batch = [self.nonzeros[k] for k in range(lower_index, upper_index)]
            batch_i = [index[0] for index in batch]
            batch_j = [index[1] for index in batch]
            
            # calculate intermediate values
            cost_inner = np.array([self.W[i].dot(self.W_tilde[j]) for i, j in batch]) + self.b[batch_i] + \
                self.b_tilde[batch_j] - np.log(np.array([self.cooccurrence_mat[i][j] for i, j in batch]))
            weighted_cost_inner = np.array([self.f_x[i][j] for i, j in batch]) * cost_inner

            # calculate the gradients of each parameter
            self.gradW[batch_i] = (self.W_tilde[batch_j].T * weighted_cost_inner).T
            self.gradW_tilde[batch_j] = (self.W[batch_i].T * weighted_cost_inner).T
            self.gradb[batch_i] = self.gradb_tilde[batch_j] = weighted_cost_inner

            if opt_method == "adagrad":
                # calculate the adaptive learning rates for adagrad
                lr_W = eta / np.sqrt(self.G_W[batch_i] + eps)
                lr_W_tilde = eta / np.sqrt(self.G_W_tilde[batch_j] + eps)
                lr_b = eta / np.sqrt(self.G_b[batch_i] + eps)
                lr_b_tilde = eta / np.sqrt(self.G_b_tilde[batch_j] + eps)

                # perform the main parameter updates
                self.W[batch_i] -= (self.gradW[batch_i].T * lr_W).T
                self.W_tilde[batch_j] -= (self.gradW_tilde[batch_j].T * lr_W_tilde).T
                self.b[batch_i] -= self.gradb[batch_i] * lr_b
                self.b_tilde[batch_j] -= self.gradb_tilde[batch_j] * lr_b_tilde

                 # update sum of square gradients for use in future iterations
                self.G_W[batch_i] += np.sum(np.square(self.gradW[batch_i]), axis=1)
                self.G_W_tilde[batch_j] += np.sum(np.square(self.gradW_tilde[batch_j]), axis=1)
                self.G_b[batch_i] += self.gradb[batch_i] ** 2
                self.G_b_tilde[batch_j] += self.gradb_tilde[batch_j] ** 2
            else:
                # perform the main parameter updates
                self.W[batch_i] -= eta * self.gradW[batch_i]
                self.W_tilde[batch_j] -= eta * self.gradW_tilde[batch_j]
                self.b[batch_i] -= eta * self.gradb[batch_i]
                self.b_tilde[batch_j] -= eta * self.gradb_tilde[batch_j]            

            cost += np.sum(weighted_cost_inner * cost_inner)
        return cost

    # train on one sample at a time
    # opt_method can be adagrad or grad (vanilla gradient descent)
    def minimize(self, eta, opt_method):
        cost = 0
        eps = 1e-8  # for use in AdaGrad
        np.random.shuffle(self.nonzeros)
        for i, j in self.nonzeros:
            cost_inner = self.W[i].dot(self.W_tilde[j]) + self.b[i] + self.b_tilde[j] - np.log(self.cooccurrence_mat[i][j])
            weighted_cost_inner = self.f_x[i][j] * cost_inner

            self.gradW[i] = self.W_tilde[j] * weighted_cost_inner
            self.gradW_tilde[j] = self.W[i] * weighted_cost_inner
            self.gradb[i] = weighted_cost_inner
            self.gradb_tilde[j] = weighted_cost_inner

            if opt_method == "adagrad":
                lr_W = eta / np.sqrt(self.G_W[i] + eps)
                lr_W_tilde = eta / np.sqrt(self.G_W_tilde[j] + eps)
                lr_b = eta / np.sqrt(self.G_b[i] + eps)
                lr_b_tilde = eta / np.sqrt(self.G_b_tilde[j] + eps)

            else:
                lr_W = lr_W_tilde = lr_b = lr_b_tilde = eta

            self.W[i] -= self.gradW[i] * lr_W
            self.W_tilde[j] -= self.gradW_tilde[j] * lr_W_tilde
            self.b[i] -= self.gradb[i] * lr_b
            self.b_tilde[j] -= self.gradb_tilde[j] * lr_b_tilde

            if opt_method == "adagrad":
                self.G_W[i] += np.sum(np.square(self.gradW[i]))
                self.G_W_tilde[j] += np.sum(np.square(self.gradW_tilde[i]))
                self.G_b[i] += self.gradb[i] ** 2
                self.G_b_tilde[j] += self.gradb_tilde[j] ** 2

            cost += weighted_cost_inner * cost_inner
        return cost

    def train(self, iters, v_dim, alpha, x_max, batch_size, learning_rate, save_intermediate_path, opt_method):
        data_type=np.float64
        self.W = np.random.rand(self.V, v_dim)
        self.W_tilde = np.random.rand(self.V, v_dim)
        self.b = np.zeros((self.V,), dtype=data_type)
        self.b_tilde = np.zeros((self.V,), dtype=data_type)

        self.gradW = np.ones((self.V, v_dim), dtype=data_type)
        self.gradW_tilde = np.ones((self.V, v_dim), dtype=data_type)
        self.gradb = np.ones((self.V,), dtype=data_type)
        self.gradb_tilde = np.ones((self.V,), dtype=data_type)

        if opt_method == "adagrad":
            self.G_W = np.ones((self.V,), dtype=data_type)
            self.G_W_tilde = np.ones((self.V,), dtype=data_type)
            self.G_b = np.ones((self.V,), dtype=data_type)
            self.G_b_tilde = np.ones((self.V,), dtype=data_type)

        # These values don't have to be recomputed each iteration
        self.f_x = defaultdict(lambda : dict())
        for i, j in self.nonzeros:
            if self.cooccurrence_mat[i][j] < x_max:
                self.f_x[i][j] = (self.cooccurrence_mat[i][j] / float(x_max)) ** alpha
            else:
                self.f_x[i][j] = 1.0
                
        for i in range(iters):
            start = time.clock()
            if batch_size == 1:
                cost = self.minimize(learning_rate, opt_method)
            elif batch_size == -1:
                cost = self.minimize_batch(sys.maxint, learning_rate, opt_method)
            else:
                cost = self.minimize_batch(batch_size, learning_rate, opt_method)
            end = time.clock()
            print "Cost for Iteration " + str(i) + " : " + str(cost)
            print "Time elapsed: {}s".format(end - start)
            if save_intermediate_path:
                if i % 10 == 0:
                    self.export_weights(save_intermediate_path + "_iter{}".format(i))

    """some diagnostic tools for the trained embeddings. One which captures most similar words, and the other which visualizes vectors """

    def get_similar(self, word, n_similar):
        switched = {ind:val for val, ind in self.vocab.iteritems()}
        #returns the n words that are the most similar to the input word using cosine similarity
        try:
            word_ind = self.vocab[word]
        except KeyError:
            return "Sorry this word does not exist."

        word_magnitude = np.linalg.norm(self.W[word_ind])

        scores = self.W.dot(self.W[word_ind])/(np.linalg.norm(self.W, axis=1)*word_magnitude)
        #kind of a retarded way to get the top n argmax but numpy doesn't have a build in method for it:

        score_dict = dict(zip(scores, switched.keys()))

        return [(switched[score_dict[i]],i) for i in sorted(score_dict.keys(), reverse=True)[0:n_similar+1]] #the first index will always be 1 because of the word itself

    def plot_vectors(self, words = []):
        #see https://www.quora.com/How-do-I-visualise-word2vec-word-vectors
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        m = TSNE(n_components=2, random_state=0)
        m.fit_transform(self.W)
        #reduces the word vectors into a lower dimensional space and then plot them

        #get word indices for words:
        try:
            word_inds = [self.vocab[w] for w in words]
        except KeyError:
            return "Sorry one of your words does not exist."
        m = m[word_inds,:]
        plt.scatter(m[:, 0], m[:, 1])
        for label, x, y in zip(self.vocab, m[:, 0], m[:, 1]):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

        plt.show()
