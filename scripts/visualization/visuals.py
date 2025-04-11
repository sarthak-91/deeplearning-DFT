import matplotlib.pyplot as plt 
import os
import pandas as pd 
import json
from scripts.ml_model.network_class import NeuralNetworkManager
import numpy as np
from scripts.data_processing.data_class import Dataset

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
    acc = np.log10(history['loss'])
    val_acc = np.log10(history['val_mse'])
    epochs = range(1, len(acc) + 1)
    fig,ax = plt.subplots()
    ax.plot(epochs, acc, '-', label='Training Loss')
    ax.plot(epochs, val_acc, ':', label='Validation Loss')
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Log(Loss)')
    y_max = max(acc.max(), val_acc.max())
    ax.set_yticks(np.arange(np.floor(0), np.ceil(y_max) + 1, 1))
    ax.legend(loc='upper right')
    fig.savefig(save_to)
    plt.close()

def error_hist(y_test,predictions,save_to:str):
    error = np.abs(y_test-predictions)
    def squish(x, c=1):
        return  np.log(1+c*x)/np.log(1+c)

    bins = np.arange(0, max(error)+10, 10)  
    print(max(error))
    print(len(error))
    hist, bin_edges = np.histogram(error, bins=bins)

    squished_counts = squish(hist)

    plt.bar(bin_edges[:-1], squished_counts, width=10, edgecolor='black', alpha=0.7)
    plt.xlabel('Error Values')
    plt.ylabel('Transformed Frequency')
    plt.title('Histogram with Log-like Squished Counts')
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
    plt.xlabel('Atoms')
    plt.ylabel('Errors')
    plt.savefig(save_to)
    plt.close()


if __name__ == "__main__":
    network = NeuralNetworkManager(folder="cnn_model",model_name="parameter")
    network.load_model(model_name="parameter",epochs=300)
    dataset = Dataset()
    x_test,y_test = dataset.load_data(dataset="testing")
    x_test_new = np.expand_dims(x_test,axis=-1)
    predictions, mse = network.predict(x_test_new,y_test)
    error_hist(y_test, predictions,save_to="cnn_model/parameter/error_hist.png")
