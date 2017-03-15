'''
Author: Tyler Chase
Date: 2017/03/14

This code reads in a confusion matrix saved as a .npy in the data folder and plots a nicely structured confusion matrix plot. 

reference: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
''' 

# Load libraries
#mport matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools

# Code to plot the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues, 
                          save_address = '/Users/tylerchase/Documents/Stanford_Classes/CS224n_Natural_Language_Processing_with_Deep_Learning/final project/data//'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=75)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j],2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_address + 'confusion_mat.png')

def main ():
    # Load confusion matrix from data folder
    address = r'/Users/tylerchase/Documents/Stanford_Classes/CS224n_Natural_Language_Processing_with_Deep_Learning/final project/data//'
    conf = np.load(address + 'confusion_mat.npy')
    classes = ['AskReddit', 'LifeProTips', 'nottheonion', 'news', 'science', 
	       'trees', 'tifu', 'personalfinance', 'mildlyinteresting', 'interestingasfuck']

    # Plot the confusion matrix
    plot_confusion_matrix(conf, classes, normalize = True)

if __name__ =='__main__':
    main()
    
