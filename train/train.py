import pandas as pd 
import numpy as np 
import scipy 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping


SEED= 1
N_SAMPLES = 500
RATIO= 0.3
VERBOSE = 0
EPOCH = 200
BATCH_SIZE = 50
SPLIT_RATIO = 0.7
INPUT_DIM = 2
DPI = 120

# design the ML workflow 
# init model arch
def initialize_model(activation_func: str):
    model = models.Sequential()
    model.add(layers.Dense(500, input_dim = INPUT_DIM, activation = activation_func))
    model.add(layers.Dense(1, activation='sigmoid'))
    # compile
    model.compile(loss='binary_crossentropy', 
                    optimizer='adam', 
                    metrics=['accuracy'])

    return model
# model.summary() 

def run_model(X: pd.DataFrame, y, activation_func: str, batch_size = BATCH_SIZE):
    
    # create or load data => Data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = RATIO, random_state = SEED)
    model = initialize_model(activation_func)
    # Early stopping 
    es = EarlyStopping(monitor='val_loss', mode = 'min', verbose = 1, patience = 20)
    # Fitting the model 
    history = model.fit(X_train, y_train, 
                    validation_split = SPLIT_RATIO, #add validation set
                    epochs = EPOCH, 
                    batch_size = BATCH_SIZE,
                    verbose= VERBOSE, 
                    callbacks = [es])

    results = model.evaluate(X_test, y_test, verbose = VERBOSE)
    
    return results, history


X, y = make_moons(n_samples = N_SAMPLES, noise=0.25, random_state = SEED)
results, history= run_model(X, y, "relu")
print(results)
with open("metrics.txt", 'w') as outfile:
        outfile.write("Test loss: {}\n" .format(results[0]))
        outfile.write("Test accuracy (MAE): {}".format(results[1]))


      
# model inference => Plot training & validation loss values
def plot(metric: str, ylabel: str)-> None:
    
    plt.plot(history.history[f'{metric}'])
    plt.plot(history.history[f'val_{metric}'])
    plt.title(f'Model {metric}')
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc = 'best')
    
    plt.tight_layout()
    plt.savefig(f"{metric}.png",dpi = DPI) 
    plt.close()
    
plot("loss", "loss score")
plot("acc", "acc score")