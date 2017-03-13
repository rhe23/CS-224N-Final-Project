'''
Author: Tyler Chase
Date: 2017/03/07

This code calls the LSTM classifier and runs it for different hyperparameters
''' 
# Load libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
# Load user library
import LSTM_classifier

# Is the data already saved? and where is it saved?
saved = 0
address = r'/home/cs224n/CS-224N-Final-Project/data//'

# Inputs for dropout constants of interest
first_value = 0
second_value = 1
step_size = 0.1

if not saved:
    # Generate training and test error across different dropout constants of LSTM
    # classification model
    error = {}
    train_error = []
    test_error = []
    value_list = np.arange(first_value,second_value + step_size,step_size)
    for dropout_const in value_list:
        temp_1, temp_2 = LSTM_classifier.run_classifier(address, epoch_size = 10, \
            dropout_const = dropout_const, train_percent = 10, dev_percent = 80)
        train_error.append(temp_1)
        test_error.append(temp_2)
        tf.reset_default_graph()
    error['train'] = train_error
    error['test'] = test_error    
    file = open(address + 'errors', 'wb')
    pickle.dump(error, file
)
    file.close()
    
else:
    file = open(address + 'errors','r')
    error = pickle.load(file)
    file.close()
    
# Plot the training error and test error 
fig, (ax1) = plt.subplots()
ax1.plot(value_list, error['train'], label = "training error")
ax1.plot(value_list, error['test'], label = "test error")
plt.legend(loc='upper right')
savefig(address + 'errors.png', bbox_inches='tight')
    

    
    
    
