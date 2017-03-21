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


# Is the data already saved? and where is it saved?
address = r'/Users/tylerchase/Documents/Stanford_Classes/CS224n_Natural_Language_Processing_with_Deep_Learning/final project/data/class_embed200//'

# Inputs for dropout constants of interest
first_value = 0.001
second_value = 0.04
step_size = 20

value_list = np.linspace(first_value,second_value,step_size)


file = open(address + 'errors_embed200_learningRate','r')

error = pickle.load(file)
file.close()
    
# Plot the training error and test error 
fig, (ax1) = plt.subplots()
ax1.plot(value_list, error['train'], label = "training error")
ax1.plot(value_list, error['test'], label = "dev error")
ax1.set_xlabel("learning rate")
ax1.set_ylabel("Error")
ax1.set_title("Regularization Parameter Scan \n learning rate") 
plt.legend(loc='center right')
min_test = min(error['test'])
min_test_add = np.argmin(error['test'])
min_dropout = value_list[min_test_add]
min_train = error['train'][min_test_add]
plt.axvline(min_dropout, c='r', linestyle='--')
plt.text(0.006, 0.10, 'Learning rate: ' + str(round(min_dropout,3)) + '\n' + 'Dev Error: ' + 
         str(round(min_test,3)) + '\n' + 'Training Error: ' + str(round(min_train,3)))
plt.savefig(address + 'learning_scan.png', bbox_inches='tight')
    

    
    
    

