from tokenize_functions import tokenizer, tokenize, append_start_end_tokens
import os, csv, random, cPickle, copy, itertools
import numpy as np
from collections import defaultdict, Counter
import time

class Vocab_Builder:
    def __init__(self):
        self.vocab = {}

    def add(self, sentence):
        tokenized = tokenize(sentence.lower())
        tokenized = append_start_end_tokens(tokenized)
        for word in tokenized:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)

    def get_vocab(self):
        return self.vocab

    # export vocab as csv
    def export(self, file_path):
        with open(file_path, 'wb') as csvfile:
            writer = csv.writer(csvfile)
            for k, v in self.vocab.items():
                writer.writerow([v, k])

def weight_func(x_ij, a, x_max):
    #x_ij is the occcurencc count, alpha is the scaling param, x_max is the cutoff
    if x_ij < x_max:
        return (x_ij/x_max)**a
    return 1


def get_batch(data_size, batch_size, shuffle=True):
#returns a list of indices for each batch of data during optimization
    data_indices = np.arange(data_size)
    np.random.shuffle(data_indices)

    for i in xrange(0, data_size, batch_size):
       yield data_indices[i:i+batch_size]

class make_embeddings:
    #Class for training glove vectors

    def __init__(self,  corpus = [], vocab = {}):

        self.vocab = vocab #vocab is assumed to be a dictionary
        self.vocab_size = len(self.vocab)
        self.coccurrence_mat = defaultdict(lambda :defaultdict(lambda :0)) #use a dictionary as representation
        self.corpus = corpus #corpus is assumed to be a list of individual sentences

    def make_cooccurrence_mat(self):
        #assume corpus is a list of individual sentences, if windowsize = 0, we take the entire sentence as the window
        #tokenize the entire corpus by sentence first;
        for i, line in enumerate(self.corpus):
            tokenized =  tokenize(line.lower())
            tokenized = append_start_end_tokens(tokenized)
            word_ids = [self.vocab[word] for word in tokenized]

            word_index = list(xrange(0, len(word_ids)))

            for word_ind in word_index: #assume that the window size is the entire sentence, #w_id serves as the center word
                v = Counter()
                words = word_ids[0: word_ind] + word_ids[word_ind + 1:len(word_ids)]
                v.update(words)
                for context_word_id, value in v.iteritems():
                    self.coccurrence_mat[word_ids[word_ind]][context_word_id] +=value

        rowids = self.coccurrence_mat.keys()
        colids = [self.coccurrence_mat[id].keys() for id in rowids]

        self.nonzeros = [zip([k]*len(colids[k]), colids[k]) for k in rowids]
        self.nonzeros = [pair for center_w in self.nonzeros for pair in center_w]

    def get_coccurrence_mat(self):
        return self.coccurrence_mat

    def minimize_batch(self, batch_size, learning_rate = 0.05):
        #using a batch_gradient descent method
        cost = 0
        for i, batch_inds in enumerate(get_batch(len(self.nonzeros), batch_size)):
            batch = [self.nonzeros[i] for i in batch_inds]
            center_inds = [b[0] for b in batch]
            #have to offset all the indices for the context words by self.vocab_size
            context_inds = [b[1] + self.vocab_size for b in batch]
            new_batch_inds = zip(center_inds, context_inds)
            #make cooccurences into a matrix
            J =  np.array([self.W[i].dot(self.W[j]) for i,j in new_batch_inds]) + self.b[center_inds] \
                + self.b[context_inds] - np.log(np.array([self.coccurrence_mat[i][j] for (i, j) in batch]))

            f_x_ij_vec = [self.fx[i][j] for i, j in batch]

            context_vecs = self.W[context_inds]
            center_vecs = self.W[center_inds]

            self.gradW[center_inds] = (context_vecs.T * ( f_x_ij_vec*J)).T
            self.gradW[context_inds] = (center_vecs.T * ( f_x_ij_vec*J)).T

            self.gradb[center_inds] = self.gradb[context_inds] = f_x_ij_vec*J

            self.W[center_inds] -= learning_rate*self.gradW[center_inds]
            self.W[context_inds] -= learning_rate*self.gradW[context_inds]
            self.b[center_inds] -=  learning_rate*self.gradb[center_inds]
            self.b[context_inds] -= learning_rate*self.gradb[context_inds]

            cost += sum(f_x_ij_vec*(J**2))

        return cost
    def test(self): #just to test some code during implementation

        cost = 0
        batch = [self.nonzeros[j] for j in [1,2]]

        center_inds = [b[0] for b in batch]
        context_inds = [b[1] + self.vocab_size for b in batch]

        np.array([self.coccurrence_mat[i][j] for (i, j) in batch])
        learning_rate = 0.05

        J =  np.array([self.W[i].dot(self.W[j]) for i,j in zip(center_inds, context_inds)]) + self.b[center_inds] + self.b[context_inds] - np.array([self.coccurrence_mat[i][j] for (i, j) in zip(center_inds, context_inds)])

        f_x_ij_vec = [self.fx[i][j] for i, j in batch]
        context_vecs = self.W[context_inds]
        center_vecs = self.W[center_inds]

        self.gradW[center_inds] = (context_vecs.T * ( f_x_ij_vec*J)).T
        self.gradW[context_inds] = (center_vecs.T * ( f_x_ij_vec*J)).T

        self.gradb[center_inds] = self.gradb[context_inds] = f_x_ij_vec*J

        self.W[center_inds] -= learning_rate*self.gradW[center_inds]
        self.W[context_inds] -= learning_rate*self.gradW[context_inds]
        self.b[center_inds] -=  learning_rate*self.gradb[center_inds]
        self.b[context_inds] -= learning_rate*self.gradb[context_inds]

        cost +=  sum(f_x_ij_vec*(J**2))

    def minimize(self, learning_rate = 0.05):

        #define cost J: which is = w_i^Tw_j + b_i + b^_j - log(x_ij) where b^ is the context bias of word j and b is the center bias of word
        cost = 0

        np.random.shuffle(self.nonzeros)

        for (i, j) in self.nonzeros:
            #J = f(xij)(w_i^Tw^_j + b_i + b^_j)^2

            J= (self.W[i,:].dot(self.W[self.vocab_size+j,:])) + self.b[i] + self.b[self.vocab_size+j] - np.log(self.coccurrence_mat[i][j])

            cost += self.fx[i][j]*(J**2)

            self.gradW[i,:] = self.fx[i][j]*J*self.W[self.vocab_size+j,:]
            self.gradW[self.vocab_size+j,:]  = self.fx[i][j]*J*self.W[i,:]
            self.gradb[i] = self.gradb[self.vocab_size+j] = self.fx[i][j]*J

            self.W[i,:] -=learning_rate*self.gradW[i,:]

            self.W[self.vocab_size+j,:] -=learning_rate*self.gradW[self.vocab_size+j,:]
            self.b[i] -=learning_rate*self.gradb[i]
            self.b[self.vocab_size+j] -= learning_rate*self.gradb[self.vocab_size+j]

        return cost

    def train(self, iters, v_dim, a, x_max, batch_size = 128, batch =True, save_intermediate_path=None):
        #how many iterations to run, and the dimension of the vectors to be trained

        #calculate weight matrix of f(x_ij)
        self.fx = copy.deepcopy(self.coccurrence_mat)

        for center_w, context_words in self.coccurrence_mat.iteritems():
            for context_word in context_words.keys():
                self.fx[center_w][context_word] = weight_func(float(self.coccurrence_mat[center_w][context_word]), a, x_max)

        # rowids = self.coccurrence_mat.keys()
        # colids = [self.coccurrence_mat[id].keys() for id in rowids]
        #
        # self.nonzeros = [zip([k]*len(colids[k]), colids[k]) for k in rowids]
        # self.nonzeros = [pair for center_w in self.nonzeros for pair in center_w]

        #generate W matrix of size 2*vocab, 0:vocab = center words, vocab +1:2*vocab are context words
        self.W = np.random.rand(2*self.vocab_size, v_dim)
        #bias b and b^
        self.b = np.zeros((2*self.vocab_size,), dtype=np.float64)

        self.gradW = np.ones((2*self.vocab_size, v_dim), dtype = np.float64)
        self.gradb = np.ones((2*self.vocab_size,), dtype=np.float64)

        self.test()
        if batch:
            for i in range(iters):
                start = time.clock()
                cost = self.minimize_batch(batch_size=batch_size)
                end = time.clock()
                print "Cost for Iteration " + str(i) + " : " + str(cost)
                print "Time elapsed: {}s".format(end - start)
                if save_intermediate_path:
                    if i % 10 == 0:
                        self.save_weights(save_intermediate_path + "_iter{}".format(i))
        else :

            for i in range(iters):
                start = time.clock()
                cost = self.minimize()
                end = time.clock()
                print "Cost for Iteration " + str(i) + " : " + str(cost)
                print "Time elapsed: {}s".format(end - start)

    def get_weights(self):
        return self.W

    def save_weights(self, path):
        with open(path, 'wb') as weights:
            cPickle.dump(self.W, weights, protocol=2)

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
