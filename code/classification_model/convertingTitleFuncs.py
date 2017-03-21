'''
Author: Tyler Chase
Date: 2017/03/15

These are functions that help convert an input reddit post to a padded list of
embeddings addresses (so the algorithm can understand them)
''' 

import numpy as np
import random
import pickle

def pad_sequences_2(data, max_length):
    # Use this zero vector when padding sequences.
    # this value corresponds to '<start>' in the dictionary
    # here we pad the start of the sentence
    zero_vector = 0

    X = data['x']
    X_2 = []
    for sentence in X:
        length = len(sentence)
        length_dif = max_length - length
        new_sentence = []
        while len(new_sentence) < length_dif:
            new_sentence.append(zero_vector)
        new_sentence.extend(sentence)
        X_2.append(new_sentence[:max_length])
    data['x'] = np.array(X_2)
    return data

# takes in data in [(subreddit, [sentence_list]),...] format and converts to 
# output one hot vector and dictionary addresses
def convert_to_XY(data, word_dict):
    if np.shape(data[0])[0] == 2:
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
        X_length = []
        data_2 = {}
        for subreddit, sentence in data:
            Y.append(sub_dict[subreddit])
            # if word is in dictionary convert to embedding address
            # otherwise return an error message saying which word is missing in embedding
            temp = []
            for j in sentence:
                try:
                    temp.append(word_dict[j])
                except:
                    print('ERROR: Word ' + j + 'is not in the dictionary!\n Please try a different word.')
                    return None
            X.append(temp)
            #X.append([word_dict[j] for j in sentence])
            X_length.append(len(sentence))
        data_2['x'] = np.array(X)
        data_2['y'] = np.array(Y)
        data_2['x_length'] = np.array(X_length)
        
    else:
        X = []
        Y = []
        X_length = []
        data_2 = {}
        for sentence in data:
            # if word is in dictionary convert to embedding address
            # otherwise return an error message saying which word is missing in embedding
            temp = []
            for j in sentence:
                try:
                    temp.append(word_dict[j])
                except:
                    print('ERROR: Word ' + j + ' is not in the dictionary!\n Please try a different word.')
                    return None
            X.append(temp)                        
            #X.append([word_dict[j] for j in sentence])
            Y.append([1,0,0,0,0,0,0,0,0,0])
            X_length.append(len(sentence))
        data_2['x'] = np.array(X)
        data_2['y'] = np.array(Y)
        data_2['x_length'] = np.array(X_length)

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
    train = pad_sequences_2(train, max_sentence_length)
    dev = pad_sequences_2(dev, max_sentence_length)
    test = pad_sequences_2(test, max_sentence_length)
    return train, dev, test