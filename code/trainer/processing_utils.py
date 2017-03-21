import cPickle, math, os
import numpy as np
print os.getcwd()

def get_embeddings(embed_path = './data/new_embeddings.pkl'):

    with open(embed_path, 'rb') as f:
        embeddings = cPickle.load(f)
        f.close()
    return embeddings

def get_data(path):

    with open(path, 'rb') as f:
        input = cPickle.load(f)
        f.close()

    return input

def get_batch(data_size, batch_size, shuffle=True):
#returns a list of indices for each batch of data during optimization
    data_indices = np.arange(data_size)
    np.random.shuffle(data_indices)

    for i in xrange(0, data_size, batch_size):
       yield data_indices[i:i+batch_size]

def get_dev_test_sets(dev_size, test_size, training_indices):
    #dev_size should be a float between 0.0 and 1.0
    #returns a list of indices for both training and dev sets

    total_sizes = dev_size+test_size
    temp_inds = np.random.choice(training_indices, int(math.floor(total_sizes*len(training_indices))), replace = False)
    training_inds = [i for i in training_indices if i not in temp_inds]
    dev_inds = temp_inds[:len(temp_inds)/2]
    # test_inds = temp_inds[len(temp_inds)/2:]

    return (training_inds, dev_inds, [])


def get_masks(sentences, max_length):
    return np.array([len(sentence)*[True] + (max_length-len(sentence))*[False] for sentence in sentences])

