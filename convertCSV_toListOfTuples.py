'''
Author: Tyler Chase
Date: 2017/03/09

This code converts a text file containing a subreddit and a title from this 
subreddit on every line deliminated by '|||||', to a list of tuples [ ('subreddit', 'title') ...]
''' 

import pickle
from tokenize_functions import tokenize

# address of csv file with a subreddit then a title from that subreddit on each
# line separated by |||||
address = r'/Users/tylerchase/Documents/Stanford_Classes/CS224n_Natural_Language_Processing_with_Deep_Learning/final project/data//'

# Read csv file and create a list of tuples
data = []
with open(address + 'all_2015_titles') as file:
    i = 0
    for row in file:
        temp = row.split(' ||||| ')
        data.append((temp[0], tokenize(temp[1])))
            
# Pickle and save the new dictionary
file = open(address + '2015_data', 'wb')
pickle.dump(data, file)
file.close()

# Import pickled dictionary for testing
file = open(address + '2015_data', 'r')
test_dict = pickle.load(file)
file.close()