import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import keras.backend as K
from keras import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix

###########
# METHODS #
###########
def visualize_training_history(history_path: str, metric: str):
    history = np.load(history_path, allow_pickle = True).item()

    if (metric == 'accuracy'):
        fig, axs = plt.subplots(figsize = (10,5))
        axs.plot(history['accuracy'], color = 'tab:blue', label = 'accuracy')
        axs.plot(history['val_accuracy'], color = 'tab:orange', label = 'val_accuracy')
        axs.legend(loc = 'upper left', frameon = True)
        axs.set_xlabel('epoch')
        axs.set_ylabel('accuracy [a.u.]')
        axs.set_title('Model Accuracy')
        axs.grid(True)
        plt.show()

    elif (metric == 'precision'):
        fig, axs = plt.subplots(figsize = (10,5))
        axs.plot(history['precision'], color = 'tab:blue', label = 'precision')
        axs.plot(history['val_precision'], color = 'tab:orange', label = 'val_precision')
        axs.legend(loc = 'upper left', frameon = True)
        axs.grid(True)
        axs.set_xlabel('epoch')
        axs.set_ylabel('precision [a.u.]')
        axs.set_title('Model Precision')
        plt.show()

    elif (metric == 'recall'):
        fig, axs = plt.subplots(figsize = (10,5))
        axs.plot(history['recall'], color = 'tab:blue', label = 'recall')
        axs.plot(history['val_recall'], color = 'tab:orange', label = 'val_recall')
        axs.legend(loc = 'upper left', frameon = True)
        axs.grid(True)
        axs.set_xlabel('epoch')
        axs.set_ylabel('recall [a.u.]')
        axs.set_title('Model Recall')
        plt.show()

    return None

