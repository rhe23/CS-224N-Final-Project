#validate models by looking at the predicted phrases.

import numpy as np
import math, os, collections, csv
import cPickle
import tensorflow as tf
from processing_utils import get_embeddings, get_data, get_batch, get_dev_test_sets, get_masks
from LSTM_model import RNN_LSTM
os.chdir("../..") #changes working dir to the top directory to avoid

