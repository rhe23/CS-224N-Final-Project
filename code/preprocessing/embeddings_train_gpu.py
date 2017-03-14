from tokenize_functions import tokenize_2
import cPickle
import numpy as np
import time
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse
import sys
from operator import itemgetter
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda.compiler import SourceModule

mod = SourceModule("""
    // Updates matrix A by adding a scalar c multiplied by another matrix B (i.e. A -= c * B)
    // Only operates on the rows specified by batch: A[batch_j] -= lr * B[batch_i]
    // batch is assumed to 1 x p, A and B have q columns, c is a scalar
    __global__ void BatchMatSubtractInplaceKernel(const int p, const int q, const float c, float *A, 
    const float *B, const int *batch) {
      int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
      if (batch_index >= p) return;
      int row = batch[batch_index];
      int col = blockIdx.x * blockDim.x + threadIdx.x;
      if (col >= q) return;

      atomicAdd(&A[row * q + col], -c * B[row * q + col]);
    }

    // Perform the update step for b/b_tilde using gradient descent
    // a[batch_j] -= lr * a[batch_i]
    __global__ void BatchVecSubtractInplaceKernel(const int p, const float lr, float *a, const float *b, const int *batch_i, 
    const int *batch_j) {
      int batch_index = blockIdx.x * blockDim.x + threadIdx.x;
      if (batch_index >= p) return;

      int ind_a = batch_j[batch_index];
      int ind_b = batch_i[batch_index];

      atomicAdd(&a[ind_a], -lr * b[ind_b]);
    }

    // For matrix A and vector b, multiply the i'th row of A by b[i] and store the result into the j'th row of C       
    // Performs this operations only for the corresponding arrays of integers batch_i, batch_j
    // batch is assumed to be 1 x p, A has q columns
    __global__ void BatchMatVecRowMultKernel(const int p, const int q, const float *A, const float *b, float *C, 
    const int *batch_i, const int *batch_j){           
      int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
      if (batch_index >= p) return;
      int col = blockIdx.x * blockDim.x + threadIdx.x;
      if (col >= q) return;                              
      int row_A = batch_i[batch_index];
      int row_C = batch_j[batch_index];                                    
                   
      C[row_C * q + col] = A[row_A * q + col] * b[batch_index];                          
    }          

    // Copies the values from vector a to b, operating only on item indices specified by batch
    // b[batch] = a
    // batch and a are assumed to be 1 x p
    __global__ void BatchCopyVectorKernel(const int p, const float *a, float *b, const int *batch) {
      int batch_index = blockIdx.x * blockDim.x + threadIdx.x;
      if (batch_index >= p) return;

      int b_row = batch[batch_index];
      b[b_row] = a[batch_index];
    }

    // weighted_cost_inner calculation
    // result = np.array([f_x[k] for k in range(lower_ind, upper_ind)]) * cost_inner
    __global__ void BatchWeightedInnerCost(const int lower_ind, const int upper_ind, const float *weights, 
    const float *cost_inner, float *result) {
      int ind = blockIdx.x * blockDim.x + threadIdx.x;
      int weights_index = ind + lower_ind;
      if (weights_index >= upper_ind) return;

      result[ind] = weights[weights_index] * cost_inner[ind];
    }

    // result = np.array([W[i].dot(W_tilde[j]) for i, j in batch])
    // result must start off as a zero array
    __global__ void BatchMatColDotKernel(const int p, const int q, const float *W, const float *W_tilde, 
    const int *batch_i, const int *batch_j, float *result) {
      int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
      if (batch_index >= p) return;
      int col = blockIdx.x * blockDim.x + threadIdx.x;
      if (col >= q) return; 

      int row_W = batch_i[batch_index];
      int row_W_tilde = batch_j[batch_index];

      atomicAdd(&result[batch_index], W[row_W * q + col] * W_tilde[row_W_tilde * q + col]);
    }

    // cost_inner += self.b[batch_i] + self.b_tilde[batch_j] - np.log(np.array([self.cooccurrence_mat[k] 
    //    for k in range(lower_index, upper_index)]))
    __global__ void BatchCostInnerKernel(const int lower_ind, const int upper_ind, float *cost_inner, 
    const float *b, const float *b_tilde, const int *cooc_mat, const int *batch_i, const int *batch_j) {
      int ind = blockIdx.x * blockDim.x + threadIdx.x;
      int cooc_index = ind + lower_ind;
      if (cooc_index >= upper_ind) return;

      int row_b = batch_i[ind];
      int row_b_tilde = batch_j[ind];

      atomicAdd(&cost_inner[ind], b[row_b] + b_tilde[row_b_tilde] - logf(cooc_mat[cooc_index]));
    }
    """)

batchMatSubtractInplace = mod.get_function("BatchMatSubtractInplaceKernel")
batchVecSubtractInplace = mod.get_function("BatchVecSubtractInplaceKernel")
batchMatVecRowMult = mod.get_function("BatchMatVecRowMultKernel")
batchCopyVector = mod.get_function("BatchCopyVectorKernel")
batchWeightedInnerCost = mod.get_function("BatchWeightedInnerCost")
batchMatColDot = mod.get_function("BatchMatColDotKernel")
batchCostInner = mod.get_function("BatchCostInnerKernel")

class GloveTrainer:
    #Class for training glove vectors
    def __init__(self,  corpus = []):
        self.corpus = corpus  # corpus is assumed to be a list of individual sentences
        self.cooccurrence_mat = None  # in sparse dok (dictionary-of-keys) format
        self.nonzeros = None  # nonzero keys/indices of the cooccurrence matrix
        self.vocab = None
        self.V = 0  # size of vocab
        self.v_dim = 0

        self.W = None
        self.W_tilde = None
        self.gradW = None
        self.b = None
        self.b_tilde = None
        self.gradb = None

        self.f_x = None  # weighting function

        self.blockDim_x = 0
        self.blockDim_y = 0
        self.numBlocks_x = 0
        self.numBlocks_y = 0

        # Cumulative squared gradient sums for AdaGrad
        self.G_W = None
        self.G_W_tilde = None
        self.G_b = None
        self.G_b_tilde = None

    def export_vocab(self, path):
        with open(path, 'wb') as file:
            cPickle.dump(self.vocab, file, protocol=2)

    # export weights as a concatenation of the 2 weights matrices W and W_tilde
    def export_weights(self, path):
        W = self.W.get()
        W_tilde = self.W_tilde.get()
        concat_weights = np.vstack((W, W_tilde))
        with open(path + '.pkl', 'wb') as file:
            cPickle.dump(concat_weights, file, protocol=2)

    def setOptimalBlockAndGridDims(self, batch_size):
        # Values for a 2-d grid
        self.blockDim_x = 16
        self.blockDim_y = 32
        self.numBlocks_x = int(batch_size / self.blockDim_x) + 1
        self.numBlocks_y = int(self.v_dim / self.blockDim_y) + 1

        # Values for a 1-d grid
        self.blockDim = 512
        self.numBlocks = int(batch_size / self.blockDim) + 1

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
        vals = vals.astype(np.int32)
        
        self.nonzeros = zip(nonzeros_i, nonzeros_j)
        self.cooccurrence_mat = gpuarray.to_gpu(vals)

    # train on minibatches
    # opt_method can be adagrad or grad (vanilla gradient descent)
    def minimize_batch(self, batch_size, eta, opt_method):
        #using a batch_gradient descent method
        cost = 0.0
        eps = 1e-8  # for use in adagrad
        for lower_index in range(0, len(self.nonzeros), batch_size):
            upper_index = min(lower_index + batch_size, len(self.nonzeros))  # index of first element after the end of this batch
            batch = [self.nonzeros[k] for k in range(lower_index, upper_index)]
            batch_i = [index[0] for index in batch]
            batch_j = [index[1] for index in batch]
            cur_batch_len = np.int32(upper_index - lower_index)
            
            batch_i_gpu = gpuarray.to_gpu(np.array(batch_i, dtype=np.int32))
            batch_j_gpu = gpuarray.to_gpu(np.array(batch_j, dtype=np.int32))
            cost_inner = gpuarray.zeros(batch_size, dtype=np.float32)
            weighted_cost_inner = gpuarray.zeros_like(cost_inner)
            
            # calculate intermediate values
            # cost_inner =  + self.b[batch_i] + \
            #     self.b_tilde[batch_j] - np.log(np.array([self.cooccurrence_mat[k] for k in range(lower_index, upper_index)]))
            batchMatColDot(cur_batch_len, self.v_dim, self.W, self.W_tilde, batch_i_gpu, batch_j_gpu, cost_inner, \
                block=(self.blockDim_x, self.blockDim_y, 1), grid=(self.numBlocks_x, self.numBlocks_y))
            context.synchronize()
            if lower_index == 0:
                print cost_inner.get()
            batchCostInner(np.int32(lower_index), np.int32(upper_index), cost_inner, self.b, self.b_tilde, \
                self.cooccurrence_mat, batch_i_gpu, batch_j_gpu, block=(self.blockDim, 1, 1), grid=(self.numBlocks, 1))
            if lower_index == 0:
                print cost_inner.get()
            context.synchronize()
            # weighted_cost_inner = np.array([self.f_x[k] for k in range(lower_index_upper_index)]) * cost_inner
            batchWeightedInnerCost(np.int32(lower_index), np.int32(upper_index), self.f_x, cost_inner, weighted_cost_inner, \
                block=(self.blockDim, 1, 1), grid=(self.numBlocks, 1))
            if lower_index == 0:
                print weighted_cost_inner.get()
            context.synchronize()
            
            # calculate the gradients of each parameter
            # self.gradW[batch_i] = (self.W_tilde[batch_j].T * weighted_cost_inner).T
            batchMatVecRowMult(cur_batch_len, self.v_dim, self.W_tilde, weighted_cost_inner, self.gradW, batch_j_gpu, batch_i_gpu, \
                block=(self.blockDim_x, self.blockDim_y, 1), grid=(self.numBlocks_x, self.numBlocks_y))
            # self.gradW_tilde[batch_j] = (self.W[batch_i].T * weighted_cost_inner).T
            batchMatVecRowMult(cur_batch_len, self.v_dim, self.W, weighted_cost_inner, self.gradW_tilde, batch_i_gpu, batch_j_gpu, \
                block=(self.blockDim_x, self.blockDim_y, 1), grid=(self.numBlocks_x, self.numBlocks_y))
                        
            # self.gradb[batch_i] = self.gradb_tilde[batch_j] = weighted_cost_inner
            batchCopyVector(cur_batch_len, weighted_cost_inner, self.b, batch_i_gpu, \
                block=(self.blockDim, 1, 1), grid=(self.numBlocks, 1))
            batchCopyVector(cur_batch_len, weighted_cost_inner, self.b_tilde, batch_j_gpu, \
                block=(self.blockDim, 1, 1), grid=(self.numBlocks, 1))
            context.synchronize()

            # perform the main parameter updates
            # self.W[batch_i] -= eta * self.gradW[batch_i]
            batchMatSubtractInplace(cur_batch_len, self.v_dim, eta, self.W, self.gradW, batch_i_gpu, \
                block=(self.blockDim_x, self.blockDim_y, 1), grid=(self.numBlocks_x, self.numBlocks_y))
            # self.W_tilde[batch_j] -= eta * self.gradW_tilde[batch_j]
            batchMatSubtractInplace(cur_batch_len, self.v_dim, eta, self.W_tilde, self.gradW_tilde, batch_j_gpu, \
                block=(self.blockDim_x, self.blockDim_y, 1), grid=(self.numBlocks_x, self.numBlocks_y))
            # self.b[batch_i] -= eta * self.gradb[batch_i]
            batchVecSubtractInplace(cur_batch_len, eta, self.b, self.gradb, batch_i_gpu, \
                block=(self.blockDim, 1, 1), grid=(self.numBlocks, 1))
            # self.b_tilde[batch_j] -= eta * self.gradb_tilde[batch_j]      
            batchVecSubtractInplace(cur_batch_len, eta, self.b_tilde, self.gradb_tilde, batch_j_gpu, \
                block=(self.blockDim, 1, 1), grid=(self.numBlocks, 1))      
            context.synchronize()

            cost += gpuarray.dot(weighted_cost_inner, cost_inner).get()
        return cost

    # train on one sample at a time
    # opt_method can be adagrad or grad (vanilla gradient descent)
    def minimize(self, eta, opt_method):
        '''
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
        '''

    def train(self, iters, v_dim, alpha, x_max, batch_size, learning_rate, save_intermediate_path, opt_method):
        data_type=np.float32
        self.W = gpuarray.to_gpu(np.random.rand(self.V, v_dim))
        self.W_tilde = gpuarray.to_gpu(np.random.rand(self.V, v_dim))
        self.b = gpuarray.to_gpu(np.zeros((self.V,), dtype=data_type))
        self.b_tilde = gpuarray.to_gpu(np.zeros((self.V,), dtype=data_type))

        self.gradW = gpuarray.to_gpu(np.ones((self.V, v_dim), dtype=data_type))
        self.gradW_tilde = gpuarray.to_gpu(np.ones((self.V, v_dim), dtype=data_type))
        self.gradb = gpuarray.to_gpu(np.ones((self.V,), dtype=data_type))
        self.gradb_tilde = gpuarray.to_gpu(np.ones((self.V,), dtype=data_type))
        '''
        if opt_method == "adagrad":
            self.G_W = gpuarray.to_gpu(np.ones((self.V,), dtype=data_type))
            self.G_W_tilde = gpuarray.to_gpu(np.ones((self.V,), dtype=data_type))
            self.G_b = gpuarray.to_gpu(np.ones((self.V,), dtype=data_type))
            self.G_b_tilde = gpuarray.to_gpu(np.ones((self.V,), dtype=data_type))
        '''
        self.v_dim = np.int32(v_dim)
        learning_rate = np.float32(learning_rate)
        self.setOptimalBlockAndGridDims(batch_size)

        # These values don't have to be recomputed each iteration
        self.f_x = [(self.cooccurrence_mat[k] / float(x_max)) ** alpha for k in range(len(self.nonzeros))]
        self.f_x = gpuarray.to_gpu(np.array(self.f_x))

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
