import matplotlib.pyplot as plt 
import os
import pandas as pd 
import json
from network_class import NeuralNetworkManager
import numpy as np
from data_class import Dataset

def histogram(dataset:pd.DataFrame,column:str,save_to:os.PathLike,filename:str=""):
    plt.figure(figsize=(8, 6))
    dataset[column].hist(bins=10, color='skyblue', edgecolor='black')
    plt.title('Histogram of {}'.format(column), fontsize=14)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(False)
    plt.savefig(os.path.join(save_to,filename+column+".png"))

def training_performance(history,save_to:str):
    """
    Visualize training and validation accuracy across epochs.
    Args:
        history (dict): Dictionary containing training history 
            with 'accuracy' and 'val_accuracy' keys.
        save_to (str): File path to save the accuracy plot.
    """
    acc = np.log(history['loss'])
    val_acc = np.log(history['val_mse'])
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, '-', label='Training Accuracy')
    plt.plot(epochs, val_acc, ':', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig(save_to)
    plt.close()

def error_vs_atoms(x_test, y_test, predictions,save_to:str):
    error = np.abs(y_test-predictions)
    N_atoms = np.zeros(error.shape)
    for i in range(len(error)):
        indices = np.where(x_test[i] == 0)[0]
        index = indices[0] if len(indices) !=0 else len(x_test[i])
        N = (np.sqrt(1 + 8*index) - 1)/2
        N_atoms[i] = N 
    plt.scatter(N_atoms,error,marker="*")
    plt.title('Error vs Atoms')
    plt.xlabel('Errors')
    plt.ylabel('Atoms')
    plt.savefig(save_to)
    plt.close()
    plt.hist(N_atoms,bins=10)
    plt.savefig("hist.png")


if __name__ == "__main__":
    network = NeuralNetworkManager(folder="cnn_model",model_name="kernel_9")
    network.load_model(model_name="kernel_6",epochs=200)
    dataset = Dataset()
    x_test,y_test = dataset.load_data(dataset="testing")
    x_test_new = np.expand_dims(x_test,axis=-1)
    predictions, mse = network.predict(x_test_new,y_test)
    error_vs_atoms(x_test=x_test,y_test=y_test,predictions=predictions,save_to="error.png")