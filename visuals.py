import matplotlib.pyplot as plt 
import os
import pandas as pd 


def histogram(dataset:pd.DataFrame,column:str,save_to:os.PathLike,filename:str=""):
    plt.figure(figsize=(8, 6))
    dataset[column].hist(bins=10, color='skyblue', edgecolor='black')
    plt.title('Histogram of {}'.format(column), fontsize=14)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(False)
    plt.savefig(os.path.join(save_to,filename+column+".png"))
