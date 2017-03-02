from tokenize_functions import tokenizer, tokenize, append_start_end_tokens
import os, csv, random, cPickle
import numpy as np

class Vocab_Builder:
    #build a vocab list with index for each vocab from a list of strings, iterates until list is empty then
    #saves the vocab
    def __init__(self):

        # self.vcounts = Counter()
        self.vocab = {}
        self.words = []

    def add(self, sentence):
        #using the current stored phrase_list and build the vocab dictionary
        tokenized = tokenize(sentence.lower())
        tokenized = append_start_end_tokens(tokenized)
        self.words += tokenized
        # self.vcounts.update(tokenized)
        return tokenized
    def update_vocab(self):
        #takes the most up-to-date vcounts and make it a vocab dictionary
        # self.vocab = {word: (i, count) for word, (i, count) in enumerate(self.vcounts)}
        self.vocab = {i:word for word, i in enumerate(set(self.words))}

    def get_vocab(self):
        return self.vocab

    def export(self, path = os.getcwd()):

        with open(path + 'vocab.csv', 'wb') as csvfile:
            writer = csv.writer(csvfile)
            for k, v in self.vocab.items():
                writer.writerow([v, k])

class make_embeddings:
    #Class for training glove vectors

    def __init__(self,  corpus = [], vocab = {}):

        self.vocab = vocab #vocab is assumed to be a dictionary
        self.vocab_size = len(self.vocab.keys())
        self.cooccur_mat = np.zeros((self.vocab_size , self.vocab_size), dtype = np.float64)

        self.corpus = corpus #corpus is assumed to be a list of individual sentences

    def add_one_count(self, center_w, context_w):
        self.cooccur_mat[center_w, context_w] += 1

    def weight_func(self, x_ij, a, x_max):
        #x_ij is the occcurencc count, alpha is the scaling param, x_max is the cutoff
        if x_ij < x_max:
            return (x_ij/x_max)**a
        return 1

    def make_cooccurance_mat(self):

        #assume corpus is a list of individual sentences, if windowsize = 0, we take the entire sentence as the window
        #tokenize the entire corpus by sentence first;
        for i, line in enumerate(self.corpus):
            tokenized =  tokenize(line.lower())
            tokenized = append_start_end_tokens(tokenized)
            word_ids = [self.vocab[word] for word in tokenized]

            word_index = list(xrange(0, len(word_ids)))

            for word_ind in word_index: #assume that the window size is the entire sentence, #w_id serves as the center word
                left_words, right_words =  word_ids[0: word_ind],  word_ids[word_ind + 1:len(word_ids)]

                if left_words:

                    map(self.add_one_count, [word_ids[word_ind]]*len(left_words), left_words)

                if right_words:

                    map(self.add_one_count, [word_ids[word_ind]]*len(right_words), right_words)

    def get_coocurance_mat(self):
        return self.cooccur_mat

    def minimize(self, learning_rate = 0.05):

        #define cost J: which is = w_i^Tw_j + b_i + b^_j - log(x_ij) where b^ is the context bias of word j and b is the center bias of word
        cost = 0
        non_zero_inds = zip(self.nonzero[0], self.nonzero[1])
        np.random.shuffle(non_zero_inds)

        for (i, j) in non_zero_inds:
            #J = f(xij)(w_i^Tw^_j + b_i + b^_j)^2

            J= (self.W[i,:].dot(self.W[self.vocab_size+j,:])) + self.b[i] + self.b[self.vocab_size+j] - np.log(self.cooccur_mat[i,j])

            cost += self.fx_ij[i,j]*(J**2)

            self.gradW[i,:] = self.fx_ij[i,j]*J*self.W[self.vocab_size+j,:]
            self.gradW[self.vocab_size+j,:]  = self.fx_ij[i,j]*J*self.W[i,:]
            self.gradb[i] = self.gradb[self.vocab_size+j] = self.fx_ij[i,j]*J

            self.W[i,:] -=learning_rate*self.gradW[i,:]

            self.W[self.vocab_size+j,:] -=learning_rate*self.gradW[self.vocab_size+j,:]
            self.b[i] -=learning_rate*self.gradb[i]
            self.b[self.vocab_size+j] -= learning_rate*self.gradb[self.vocab_size+j]

        return cost

    def train(self, iters, v_dim, a, x_max):
        #how many iterations to run, and the dimension of the vectors to be trained

        #calculate weight matrix of f(x_ij)
        self.fx_ij = self.cooccur_mat.copy()
        f = np.vectorize(self.weight_func)
        self.nonzero = np.nonzero(self.cooccur_mat>0) #indices with nonzero entries

        self.fx_ij[self.nonzero] = f(self.cooccur_mat[self.nonzero], a = a, x_max = x_max)

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

        plt.scatter(m[:, 0], m[:, 1])
        for label, x, y in zip(self.vocab, m[:, 0], m[:, 1]):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

        plt.show()