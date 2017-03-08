'''
Author: Tyler Chase
Date: 2017/03/07

motivation for structure: https://gist.github.com/danijar/c7ec9a30052127c7a1ad169eeb83f159     #cite
dropout wrapper: https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/DropoutWrapper
MultiRnnCell: https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell
@lazy_property: https://danijar.com/structuring-your-tensorflow-models/                        #cite
property and decorators: https://www.programiz.com/python-programming/property
LSTM description: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
dynamic_rnn: https://www.tensorflow.org/versions/r0.10/api_docs/python/nn/recurrent_neural_networks#dynamic_rnn

''' 

def add_padding(max_length, sentences):
   return np.matrix([[True]*len(i) + [False]*(max_length-len(i)) for i in sentences])

'''incomplete'''
def import_dataset():
    train.data
    train.target
    train.sample
    return train, test


# Import libraries
import numpy as np
import random
import tensorflow as tf

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
        self.num_hidden = _num_hidden
        self.num_layers = _num_layers
        self.learning_rate = learning_rate
        self.prediction
        
    @lazy_property
    def prediction(self):
        # define LSTM cell
        lstm = tf.contrib.rnn.LSTMCell(self._num_hidden)
        # Implement a dropout regularization on cell outputs 
        lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob = self.dropout)
        # Link together (num_layers) number of cells to form the temporal model
        lstm = tf.contrib.rnn.MultiRnnCell([lstm] * self._num_layers)
        # Create neaural network from cell
        output, _ = tf.nn.dynamic_rnn(lstm, self.data)
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
        
        
        
        

def main():
    # Model parameters
    epoch_size = 10
    minibatch_size = 100
    dropout_const = 0.5
    
    train, test = import_dataset()
    sample_size, max_sentence_size, word_size = train.data.shape
    class_size = train.target.shape[1]
    data = tf.placeholder(tf.float32, [None, max_sentence_size, word_size])
    target = tf.placeholder(tf.float32, [None, class_size])
    dropout = tf.placeholder(tf.float32)   
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

