'''
Author: Tyler Chase
Date: 2017/03/15

This code reads in a sentence and predicts which subreddit it most likely would
have success in. It is currently written as a demo to run in a terminal. 
''' 

import tensorflow as tf
import numpy as np
import pickle
from tokenize_functions import tokenize
import convertingTitleFuncs
import LSTM_classifier

# New model class
class Model:
    def __init__(self, address, max_length=20):
        self.address = address
        self.max_length = max_length
        
    # Load a priviously optimized model
    def loadModel(self):

        self.address = r'/Users/tylerchase/Documents/Stanford_Classes/CS224n_Natural_Language_Processing_with_Deep_Learning/final project/data//'
        self.max_length = 20

        # Load embeddings
        # Data address
        embeddings = np.load(self.address + 'large_weights.pkl_iter100')
        zeroVecAdd, zeroVecLength = np.shape(embeddings)
   
    
        config = LSTM_classifier.Config(1, 10, 25, self.max_length, 1)
        self.data = tf.placeholder(tf.int32, [None, self.max_length])
        self.target = tf.placeholder(tf.float32, [None, 10])
        self.dropout = tf.placeholder(tf.float32)   
        self.model = LSTM_classifier.Classification(config, embeddings, \
                                        self.data, self.target, self.dropout)       
    
        # Load and run model for list of input sentences returning a list of 
        # predicted likelihoods
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, '../../../data/classification_model')   
    
    # Classify a user defined sentence   
    def classify(self, titles):    
        if type(titles) == str:
            titles = [titles]
        
        with open(self.address + 'embedding_dict') as input:
            embedAddress_dict = pickle.load(input) 
            
        classes = ['AskReddit', 'LifeProTips', 'nottheonion', 'news', 'science', 'trees', 'tifu', 
                       'personalfinance', 'mildlyinteresting', 'interestingasfuck']
  
        # Convert sentence to list of lowercase words
        converted = []
        for sentence in titles:
            converted.append( tokenize(sentence.lower()) )
     
        # Convert list of lowercase words to padded list of embeddings addresses
        data_sentence = convertingTitleFuncs.convert_to_XY(converted, embedAddress_dict)
        if data_sentence == None:
            return None
        data_sentence = convertingTitleFuncs.pad_sequences_2(data_sentence, self.max_length)
    
        prediction = self.sess.run(self.model.prediction, \
            {self.data:data_sentence['x'], self.target:data_sentence['y'], self.dropout:1} )

        for i in range(np.shape(prediction)[0]):
            likelyLocation = np.argmax(prediction[i,:])
            likelihood = np.max(prediction[i,:])
            label = classes[likelyLocation]
            print( 'This sentence belongs to r/' + label + ' with {:.1f}% liklihood.'.format(likelihood * 100) )
            print('\n')
    
    
if __name__ == '__main__':
    address = r'/Users/tylerchase/Documents/Stanford_Classes/CS224n_Natural_Language_Processing_with_Deep_Learning/final project/data//'
    # define above class object
    model_object = Model(address)
    # load in trained tensorflow model
    model_object.loadModel()
    # Keep promping user for reddit post title input
    while True:
        while True:
            # Read in title string from terminal
            try:
                title = input ('Please enter a Reddit post title (in quotes) for consideration. \n')
            # Print error message if data is not input as a string
            except:
                print('ERROR please put the post title IN QUOTES')
                break
            model_object.classify(title)









