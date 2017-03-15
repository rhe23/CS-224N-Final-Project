#filters the vocab and embeddings file to include only words from subrredit looked at. Returns new vocab and new embedding
import math, os, collections, csv, cPickle
import numpy as np
from tokenize_functions import tokenize_2

#
os.chdir("../..")
def get_embeddings(embed_path = './data/large_weights.pkl_iter100'):

    with open(embed_path, 'rb') as f:
        embeddings = cPickle.load(f)
        f.close()
    return embeddings

def get_data(path):

    with open(path, 'rb') as f:
        input = cPickle.load(f)
        f.close()

    return input

vocabs = collections.defaultdict(str)
with open('./data/large_vocab.csv') as csvfile:
    vocab = csv.reader(csvfile)
    for v in  vocab:
        vocabs[v[1]] = v[0]


original_embeddings = get_embeddings()
original_data =  get_data(path = './data/2015_data')
all_words = []
for r, post in original_data:
        all_words += post
all_words.append('<start>')
all_words.append('<end>')
all_words= np.unique(all_words)


word_inds = [(word, vocabs[word]) for word in all_words]


new_vocab = {}
vocab_counter = 0
new_embeddings = np.zeros((len(all_words), original_embeddings.shape[1]))

for pair in word_inds:

    new_embeddings[vocab_counter] = original_embeddings[int(pair[1]), :]

    new_vocab[pair[0]] = vocab_counter

    vocab_counter +=1


with open('./data/large_vocab_new.csv', 'wb') as csvfile:
    vocabwriter = csv.writer(csvfile)
    for word, index in new_vocab.iteritems():
        vocabwriter.writerow([index, word])


with open('./data/new_embeddings.pkl', 'wb') as f:
        cPickle.dump(new_embeddings, f)


# with open('./data/new_embeddings.pkl', 'rb') as f:
#         b = cPickle.load(f)
# print b