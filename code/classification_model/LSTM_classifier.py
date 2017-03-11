'''
Author: Tyler Chase
Date: 2017/03/07

This is a model that takes in a reddit post title from one of 10 subreddits 
(AskReddit, LifeProTips, nottheonion, news, science, trees, tifu, personalfincance, 
mildlyinteresting, and interestingasfuck) and predicts which subreddit it belongs to. 

motivation for structure: https://gist.github.com/danijar/c7ec9a30052127c7a1ad169eeb83f159     #cite
dropout wrapper: https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/DropoutWrapper
MultiRnnCell: https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell
@lazy_property: https://danijar.com/structuring-your-tensorflow-models/                        #cite
property and decorators: https://www.programiz.com/python-programming/property
LSTM description: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
dynamic_rnn: https://www.tensorflow.org/versions/r0.10/api_docs/python/nn/recurrent_neural_networks#dynamic_rnn
''' 

# Import libraries
import numpy as np
import random
import tensorflow as tf
import functools
import pickle
import random


# Code recycled from hw3
def pad_sequences(data, max_length):
    # Use this zero vector when padding sequences.
    # this value corresponds to '<end>' in the dictionary
    zero_vector = 22

    X = data['x']
    X_2 = []
    for sentence in X:
        new_sentence = list(sentence)
        while len(new_sentence) < max_length:
            new_sentence.append(zero_vector)
        X_2.append(new_sentence[:max_length])
    data['x'] = X_2
    return data

# takes in data in [(subreddit, [sentence_list]),...] format and converts to 
# output one hot vector and dictionary addresses
def convert_to_XY(data, word_dict):
    sub_dict = {}
    sub_dict['AskReddit'] =         [1,0,0,0,0,0,0,0,0,0]
    sub_dict['LifeProTips'] =       [0,1,0,0,0,0,0,0,0,0]
    sub_dict['nottheonion'] =       [0,0,1,0,0,0,0,0,0,0]
    sub_dict['news'] =              [0,0,0,1,0,0,0,0,0,0]
    sub_dict['science'] =           [0,0,0,0,1,0,0,0,0,0]
    sub_dict['trees'] =             [0,0,0,0,0,1,0,0,0,0]
    sub_dict['tifu'] =              [0,0,0,0,0,0,1,0,0,0]
    sub_dict['personalfinance'] =   [0,0,0,0,0,0,0,1,0,0]
    sub_dict['mildlyinteresting'] = [0,0,0,0,0,0,0,0,1,0]
    sub_dict['interestingasfuck'] = [0,0,0,0,0,0,0,0,0,1]
    
    X = []
    Y = []
    data_2 = {}
    for subreddit, sentence in data:
        Y.append(sub_dict[subreddit])
        X.append([word_dict[j] for j in sentence])
    data_2['x'] = X
    data_2['y'] = Y
    return data_2
         
        

# Import dataset and create training set, development set, and test set
def import_dataset(address, word_dict, max_sentence_length, train_percent = 80, dev_percent = 10):
    SEED = 455
    random.seed(SEED)
    # Read csv file and create a list of tuples
    with open(address) as file:
        data = pickle.load(file)
    # Mix data and split into tran, dev, and test sets
    random.shuffle(data)
    length = len(data)
    train_end = int(train_percent*length/100)
    dev_end = train_end + int(dev_percent*length/100)
    train = data[:train_end]
    dev = data[train_end:dev_end]
    test = data[dev_end:]
    # Convert to dictionary containing output vectors and sentences of word addresses
    train = convert_to_XY(train, word_dict)
    dev = convert_to_XY(dev, word_dict)
    test = convert_to_XY(test, word_dict)
    # Pad sequences to make them the same max length
    train = pad_sequences(train, max_sentence_length)
    dev = pad_sequences(dev, max_sentence_length)
    test = pad_sequences(test, max_sentence_length)
    return train, dev, test

    
    
    
    
# reduces the text needed for running @property making code more readable
def lazy_property(function):
    # Attribute used to test if code chunk has been run or not
    attribute = '_lazy_' + function.__name__
    # run wrapper function when wrapper returned below
    @property
    # Keeps original function attributes such as function.__name__ 
    # Otherwise it would be replaced with the wrapper attributes
    @functools.wraps(function)
    def wrapper(self):
        # If doesn't have (attribute) then code chunk hasn't been run
        if not hasattr(self, attribute):
            # Run code chunk and store it in (attribute) of class
            setattr(self, attribute, function(self))     
        # return the value of the number stored in (attribute)
        return getattr(self, attribute)
    return wrapper


class classification:
    
    def __init__(self, data, target, dropout, num_hidden = 200, num_layers = 20, learning_rate = 0.003):
        self.data = data
        self.target = target
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.learning_rate = learning_rate
        self.prediction
        
    @lazy_property
    def prediction(self):
        # define LSTM cell
        lstm = tf.nn.rnn_cell.LSTMCell(self._num_hidden)
        # Implement a dropout regularization on cell outputs 
        lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob = self.dropout)
        # Link together (num_layers) number of cells to form the temporal model
        lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * self._num_layers)
        # Create neaural network from cell
        output, _ = tf.nn.dynamic_rnn(lstm, self.data, dtype = tf.float32)
        # Select last output for classification
        # Change output from [sample][time][dim_output] to [time][sample][dim_output]
        output = tf.transpose(output, [1,0,2])
        # Gather last output slice
        last_out = tf.gather(output, ( int(output.getshape()[0]) - 1) )
        weight, bias = self._initialize_weight_bias(self._num_hidden, 
                                               int(self.target.get_shape()[1]) )
        prediction = tf.nn.softmax( tf.matmul(last_out, weight) + bias )
        return(prediction)
        
    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum( self.target * tf.log(self.prediction) )
        return cross_entropy
        
    @lazy_property
    def optimize(self):
        opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        return(opt.minimize(self.cost))
        
    @lazy_property
    def error(self):
        incorrect = tf.not_equal(tf.argmax(self.prediction, axis = 1), 
                                 tf.argmax(self.target, axis = 1))
        return tf.reduce_mean( tf.cast(incorrect, tf.float32) )
        
    # @static method used because function doesn't require instance "self"
    @staticmethod
    def _initialize_weight_bias(input_length, output_length):
        weight = tf.get_variable("weight", shape = [input_length, output_length],\
            initializer=tf.contrib.layers.xavier_initializer() )
        bias = tf.get_variable("bias", shape = [output_length], \
            initializer=tf.constant_initializer(0) )
        return weight, bias
        
        
 
        
        
        
        
      
# Load embeddings
# Data address
address = r'/Users/tylerchase/Documents/Stanford_Classes/CS224n_Natural_Language_Processing_with_Deep_Learning/final project/data//'
embeddings = np.load(address + 'large_weights.pkl_iter100')
with open(address + 'embedding_dict') as input:
    embedAddress_dict = pickle.load(input)
    
    
# Model parameters
epoch_size = 10
minibatch_size = 100
dropout_const = 0.5
max_sentence_length = 20
embed_length = 25
    
train, dev, test = import_dataset(address + '2015_data', embedAddress_dict, max_sentence_length)

sample_size, class_size = np.shape(train['y'])
data = tf.placeholder(tf.int32, [None, max_sentence_length, embed_length])
target = tf.placeholder(tf.int32, [None, class_size])
dropout = tf.placeholder(tf.float32)   
embeddings = tf.Variable(embeddings)
embeddings = tf.nn.embedding_lookup(embeddings, data)
embeddings = tf.reshape(embeddings, [-1, max_sentence_length, embed_length]) 
model = classification(data, target, dropout)
sess = tf.Session()
sess.run( tf.global_variables_initializer() )
for epoch in epoch_size:
    for i in np.arange(0, sample_size, minibatch_size):
        batch_x = train['x'][i:(i+minibatch_size),:]
        batch_y = train['y'][i:(i+minibatch_size),:]
        sess.run(model.optimize, 
                 {data:batch_x, target:batch_y, dropout:dropout_const} )
    error = sess.run(model.error, 
                     {data:test.data, target:test.target, dropout:1} )
    print('Epoch:{:2d}), Error:{:3.1f}%'.format( (epoch + 1), (100*error))) 
 
        
'''
def main():
    
    # Load embeddings
    # Data address
    address = r'/Users/tylerchase/Documents/Stanford_Classes/CS224n_Natural_Language_Processing_with_Deep_Learning/final project/data//'
    embeddings = np.load(address + 'large_weights.pkl_iter100')
    with open(address + 'embedding_dict') as input:
        embedAddress_dict = pickle.load(input)
    
    
    # Model parameters
    epoch_size = 10
    minibatch_size = 100
    dropout_const = 0.5
    max_sentence_length = 20
    
    train, dev, test = import_dataset(address + '2015_data', embedAddress_dict)
    sample_size, class_size = np.shape(train['y'])
    _, _, embed_length = np.shape(train['x'])
    data = tf.placeholder(tf.float32, [None, max_sentence_length, embed_length])
    target = tf.placeholder(tf.float32, [None, class_size])
    dropout = tf.placeholder(tf.float32)   
    embeddings = tf.Variable(embeddings)
    embeddings = tf.nn.embedding_lookup(embeddings, data)
    embeddings = tf.reshape(embeddings, [-1, max_sentence_length, embed_length]) 
    model = classification(data, target, dropout)
    sess = tf.Session()
    sess.run( tf.global_variables_initializer() )
    for epoch in epoch_size:
        for i in np.arange(0, sample_size, minibatch_size):
            batch = train[i:(i+minibatch_size),:,:]
            sess.run(model.optimize, 
                     {data:batch.data, target:batch.target, dropout:dropout_const} )
        error = sess.run(model.error, 
                         {data:test.data, target:test.target, dropout:1} )
        print('Epoch:{:2d}), Error:{:3.1f}%'.format( (epoch + 1), (100*error))) 
        
        
    
    
# Run main() if current namespace is main
if __name__ == '__main__':
    main()
'''
