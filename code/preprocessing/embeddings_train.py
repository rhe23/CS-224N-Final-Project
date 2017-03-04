from tokenize_functions import tokenizer, tokenize, append_start_end_tokens
import os, csv, random, cPickle, copy, itertools
import numpy as np
from collections import defaultdict, Counter

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

    def export(self, path = os.getcwd()):

        with open(path + 'vocab.csv', 'wb') as csvfile:
            writer = csv.writer(csvfile)
            for k, v in self.vocab.items():
                writer.writerow([v, k])

def weight_func(x_ij, a, x_max):
    #x_ij is the occcurencc count, alpha is the scaling param, x_max is the cutoff
    if x_ij < x_max:
        return (x_ij/x_max)**a
    return 1


class make_embeddings:
    #Class for training glove vectors

    def __init__(self,  corpus = [], vocab = {}):

        self.vocab = vocab #vocab is assumed to be a dictionary
        self.vocab_size = len(self.vocab)
        self.coocurance_mat = defaultdict(lambda :defaultdict(lambda :0)) #use a dictionary as representation
        self.corpus = corpus #corpus is assumed to be a list of individual sentences

    def make_cooccurance_mat(self):
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
                    self.coocurance_mat[word_ids[word_ind]][context_word_id] +=value

    def get_coocurance_mat(self):
        return self.coocurance_mat

    def minimize(self, learning_rate = 0.05):

        #define cost J: which is = w_i^Tw_j + b_i + b^_j - log(x_ij) where b^ is the context bias of word j and b is the center bias of word
        cost = 0

        np.random.shuffle(self.nonzeros)

        for (i, j) in self.nonzeros:
            #J = f(xij)(w_i^Tw^_j + b_i + b^_j)^2

            J= (self.W[i,:].dot(self.W[self.vocab_size+j,:])) + self.b[i] + self.b[self.vocab_size+j] - np.log(self.coocurance_mat[i][j])

            cost += self.fx[i][j]*(J**2)

            self.gradW[i,:] = self.fx[i][j]*J*self.W[self.vocab_size+j,:]
            self.gradW[self.vocab_size+j,:]  = self.fx[i][j]*J*self.W[i,:]
            self.gradb[i] = self.gradb[self.vocab_size+j] = self.fx[i][j]*J

            self.W[i,:] -=learning_rate*self.gradW[i,:]

            self.W[self.vocab_size+j,:] -=learning_rate*self.gradW[self.vocab_size+j,:]
            self.b[i] -=learning_rate*self.gradb[i]
            self.b[self.vocab_size+j] -= learning_rate*self.gradb[self.vocab_size+j]

        return cost

    def train(self, iters, v_dim, a, x_max):
        #how many iterations to run, and the dimension of the vectors to be trained

        #calculate weight matrix of f(x_ij)
        self.fx = copy.deepcopy(self.coocurance_mat)

        for center_w, context_words in self.coocurance_mat.iteritems():
            for context_word in context_words.keys():
                self.fx[center_w][context_word] = weight_func(float(self.coocurance_mat[center_w][context_word]), a, x_max)

        rowids = self.coocurance_mat.keys()
        colids = [self.coocurance_mat[id].keys() for id in rowids]

        self.nonzeros = [zip([k]*len(colids[k]), colids[k]) for k in rowids]
        self.nonzeros = [pair for center_w in self.nonzeros for pair in center_w]

        #generate W matrix of size 2*vocab, 0:vocab = center words, vocab +1:2*vocab are context words
        self.W = np.random.rand(2*self.vocab_size, v_dim)
        #bias b and b^
        self.b = np.zeros((2*self.vocab_size,), dtype=np.float64)

        self.gradW = np.ones((2*self.vocab_size, v_dim), dtype = np.float64)
        self.gradb = np.ones((2*self.vocab_size,), dtype=np.float64)

        for i in range(iters):
            cost = self.minimize()
            print "Cost for Interation " + str(i) + " : " + str(cost)

    def get_weights(self):
        return self.W

    def save_weights(self, path):
        with open(path, 'wb') as weights:
            cPickle.dump(self.W, weights, protocol=2)

    """some diagnostic tools for the trained embeddings. One which captures most similar words, and the other which visualizes vectors """

    def get_similar(self, word, n_similar):
        #returns the n words that are the most similar to the input word using cosine similarity
        try:
            word_ind = self.vocab[word]
        except KeyError:
            return "Sorry this word does not exist."

        word_magnitude = np.linalg.norm(self.W[word_ind])

        scores = self.W.dot(self.W[word_ind,])/(np.linalg.norm(self.W, axis=0)*word_magnitude)
        #kind of a retarded way to get the top n argmax but numpy doesn't have a build in method for it:
        score_dict = dict(zip(scores, self.W.keys(xrange(0, self.vocab_size))))

        return [score_dict[i] for i in sorted(score_dict.keys())[1:n_similar+1]] #the first index will always be 1 because of the word itself

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
