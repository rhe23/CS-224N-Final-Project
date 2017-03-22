'''
Author: Tyler Chase
Date: 2017/03/09

this code converts a csv of embedding addresses and words to a dictionary that 
takes in the word and returns the address of it's embedding in the embedding
matrix
''' 

import csv
import pickle

# address of csv file with embedding address as first column and word in the 
# second column
address = r'/Users/tylerchase/Documents/Stanford_Classes/CS224n_Natural_Language_Processing_with_Deep_Learning/final project/data//'

# Read csv file and create dictionary
embedding_dict = {}
with open(address + 'large_vocab.csv') as csvfile:
    data = csv.reader(csvfile, delimiter = ',')
    for row in data:
        embedding_dict[row[1]] = int(row[0])
        
# Pickle and save the new dictionary
file = open(address + 'embedding_dict', 'wb')
pickle.dump(embedding_dict, file)
file.close()

# Import pickled dictionary for testing
file = open(address + 'embedding_dict', 'r')
test_dict = pickle.load(file)
file.close()

        

        
        