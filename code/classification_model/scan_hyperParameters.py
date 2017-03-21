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
saved = 0
address = r'/home/cs224n/CS-224N-Final-Project/data//'

# Inputs for dropout constants of interest
first_value = 0.001
second_value = 0.04
step_size = 20

value_list = np.linspace(first_value,second_value,step_size)
#value_list = np.logspace(first_value, second_value, step_size)
if not saved:
    # Generate training and test error across different dropout constants of LSTM
    # classification model
    error = {}
    train_error = []
    test_error = []
    for temp in value_list:
        temp_1, temp_2 = LSTM_classifier.run_classifier(address, epoch_size = 10, \
            dropout_const = 0.55, train_percent = 80, dev_percent = 10, layers = 1, 
	    learning_rate = temp)
        train_error.append(temp_1)
        test_error.append(temp_2)
        tf.reset_default_graph()
    error['train'] = train_error
    error['test'] = test_error    
    file = open(address + 'errors_embed200_learningRate', 'wb')
    pickle.dump(error, file)
    file.close()
    
else:
    file = open(address + 'errors_embed200_learningRate','r')
    error = pickle.load(file)
    file.close()
    
# Plot the training error and test error 
fig, (ax1) = plt.subplots()
ax1.plot(value_list, error['train'], label = "training error")
ax1.plot(value_list, error['test'], label = "dev error")
ax1.set_xlabel("learning_rate")
ax1.set_ylabel("Error")
ax1.set_title("Regularization Parameter Scan \n Learning Rate") 
plt.legend(loc='upper right')
plt.savefig(address + 'errors_embed200_learningRate.png', bbox_inches='tight')
    

    
    
    
