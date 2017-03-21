#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 02:37:50 2017

@author: tylerchase
"""

import matplotlib.pyplot as plt
import pickle

address = r'/Users/tylerchase/Documents/Stanford_Classes/CS224n_Natural_Language_Processing_with_Deep_Learning/final project/data//'

file = open(address + 'errors','r')
error = pickle.load(file)
file.close()

file = open(address + 'errors_2layers','r')
error_2layers = pickle.load(file)
file.close()

plt.figure()
plt.plot(error['test'], label = "test 1-layer")
plt.plot(error_2layers['test'], label = "test 2-layer")
plt.legend(loc = 'upper right')
plt.savefig(address + 'test_compare_layers.png', bbox_inches='tight')
