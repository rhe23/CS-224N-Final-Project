'''
Author: Tyler Chase
Date: 2017/03/07

This code calls the LSTM classifier and runs it for different hyperparameters
''' 
# Load libraries
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
# Load user library
import LSTM_classifier

# Is the data already saved? and where is it saved?
saved = 1
address = r'/Users/tylerchase/Documents/Stanford_Classes/CS224n_Natural_Language_Processing_with_Deep_Learning/final project/data/class_embed200//'

# Inputs for dropout constants of interest
first_value = 0
second_value = 1
step_size = 0.05

value_list = np.arange(first_value,second_value + step_size,step_size)
if not saved:
    # Generate training and test error across different dropout constants of LSTM
    # classification model
    error = {}
    train_error = []
    test_error = []
    for dropout_const in value_list:
        temp_1, temp_2 = LSTM_classifier.run_classifier(address, epoch_size = 10, \
            dropout_const = dropout_const, train_percent = 80, dev_percent = 10, layers = 2)
        train_error.append(temp_1)
        test_error.append(temp_2)
        tf.reset_default_graph()
    error['train'] = train_error
    error['test'] = test_error    

    file = open(address + 'errors_2layers', 'wb')
    pickle.dump(error, file)
    file.close()
    
else:
    file = open(address + 'errors_embed200','r')
    error = pickle.load(file)
    file.close()
    
# Plot the training error and test error 
fig, (ax1) = plt.subplots()
ax1.plot(value_list, error['train'], label = "training error")
ax1.plot(value_list, error['test'], label = "test error")
ax1.set_xlabel("dropout constant")
ax1.set_ylabel("Error")
ax1.set_title("Regularization Parameter Scan \n Dropout Rate") 
plt.legend(loc='upper right')
min_test = min(error['test'])
min_test_add = np.argmin(error['test'])
min_dropout = value_list[min_test_add]
min_train = error['train'][min_test_add]
plt.axvline(min_dropout, c='r', linestyle='--')
plt.text(0.6, 0.5, 'Dropout Rate: ' + str(min_dropout) + '\n' + 'Dev Error: ' + 
         str(round(min_test,3)) + '\n' + 'Training Error: ' + str(round(min_train,3)))
plt.savefig(address + 'dropout_scan.png', bbox_inches='tight')
    

    
    
    

