import numpy as np
import os
from sklearn.model_selection import train_test_split
import pandas as pd 
from scripts.data_processing.create_dataset import create_train_test
from scripts.config import MOLECULE_CSV, DATASETS_DIR
def load_npz(npz_file:str,names:list) -> dict:
    """Loads specified arrays from a `.npz` file and returns them as a dict of NumPy arrays.

    Args:
        npz_file (str): Path to the `.npz` file containing the arrays.
        names (list): A list of strings representing the names of the arrays to load.

    Returns:
        dict: A Dictionary of NumPy arrays corresponding to the specified names.

    Raises:
        KeyError: If a specified name is not found in the `.npz` file.
    """

    array = np.load(npz_file)
    dict_of_arrays = {}
    for name in names:
        if name not in array:
            raise KeyError(f"Key '{name}' not found in {npz_file}")
        dict_of_arrays[name] = array[name]
    array.close()
    return dict_of_arrays


class Dataset:
    """
    A class for managing machine learning datasets, particularly for molecular data.

    This class handles dataset splitting, loading, and visualization of 
    molecular properties such as energy, atomic mass, and number of atoms.

    Attributes:
        input_file (str): Path to the input data file.
        output_file (str): Path to the output labels file.
        folder (str): Directory for storing split dataset files.
        x_train (numpy.ndarray): Training input data.
        y_labeled_train (numpy.ndarray): Training labels.
        x_test (numpy.ndarray): Testing input data.
        y_labeled_test (numpy.ndarray): Testing labels.
    """

    def __init__(self, complete_dataset:str=MOLECULE_CSV, 
                 npz_folder:str=DATASETS_DIR,csv_folder:str=DATASETS_DIR,
                 test_size:float=0.3,resplit:bool=False, random_state:int=40):
        """
        Initialize the Dataset with input and output files.

        Args:
            input_file (str, optional): Path to input data file. Defaults to "new_input.txt".
            output_file (str, optional): Path to output labels file. Defaults to "new_output.txt".
            folder (str, optional): Directory to store split dataset. Defaults to "data_model".
            test_size (float, optional): Proportion of dataset to include in test split. 
                                         Defaults to 0.3.
        """
        self.dataset_path = complete_dataset
        self.npz_folder=npz_folder
        self.csv_folder = csv_folder
        self.training_set_csv = os.path.join(csv_folder,'training_set.csv')
        self.testing_set_csv= os.path.join(csv_folder,'testing_set.csv')
        self.training_set_npz =  os.path.join(npz_folder,'training_set.npz')
        self.testing_set_npz = os.path.join(npz_folder,'testing_set.npz')
        self.test_size = test_size
        self.random_state=random_state
        
        if not os.path.exists(complete_dataset):
            raise FileNotFoundError("Complete dataset not found.")
        
        if not (os.path.exists(self.training_set_npz) and os.path.exists(self.testing_set_npz)):
            self.split_dataset()
        elif resplit == True:  
            self.resplit_dataset()

    def split_dataset(self,**kwargs):
        print("Splitting dataset")
        if kwargs != {}:
            if 'test_size' in kwargs.keys():
                self.test_size = kwargs['test_size']
            if 'random_state' in kwargs.keys():
                self.random_state = kwargs['random_state']
        molecules=pd.read_csv(self.dataset_path)
        create_train_test(molecules=molecules,npz_folder=self.npz_folder,dataset_folder=self.csv_folder,
                          test_size=self.test_size,random_state=self.random_state)
    
    
    def resplit_dataset(self,**kwargs):
        # Load CSV files
        if kwargs != {}:
            if 'test_size' in kwargs.keys():
                self.test_size = kwargs['test_size']
            if 'random_state' in kwargs.keys():
                self.random_state = kwargs['random_state']
        train_df = pd.read_csv(self.training_set_csv)
        test_df = pd.read_csv(self.testing_set_csv)
        # Load NPZ files
        train_npz = np.load(self.training_set_npz)
        test_npz = np.load(self.testing_set_npz)
        # Extract arrays from NPZ files
        train_input = train_npz['input']
        train_output = train_npz['output']
        test_input = test_npz['input']
        test_output = test_npz['output']
        train_npz.close()
        test_npz.close()
        # Combine datasets
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        del train_df,test_df
        combined_input = np.vstack([train_input, test_input])
        combined_output =  np.concatenate([train_output, test_output])
        # Resplit 
        train_indices, test_indices = train_test_split(
            np.arange(len(combined_df)),
            test_size=self.test_size,
            random_state=self.random_state
        )
        # Create new datasets
        new_train_df = combined_df.iloc[train_indices].reset_index(drop=True)
        new_test_df = combined_df.iloc[test_indices].reset_index(drop=True)
        new_train_input = combined_input[train_indices]
        new_train_output = combined_output[train_indices]
        new_test_input = combined_input[test_indices]
        new_test_output = combined_output[test_indices]
        # Save new CSV files
        new_train_df.to_csv(self.training_set_csv, index=False)
        new_test_df.to_csv(self.testing_set_csv, index=False)
        
        # Save new NPZ files
        np.savez(self.training_set_npz, input=new_train_input, output=new_train_output)
        np.savez(self.testing_set_npz, input=new_test_input, output=new_test_output)
        del new_train_input,new_train_output,new_test_input,new_test_output
        print(f"Successfully resplit data with random_state={self.random_state}")
        print(f"New training set: {len(new_train_df)} samples")
        print(f"New testing set: {len(new_test_df)} samples")
        del combined_df, new_train_df,new_test_df

    def load_data(self,dataset='training')->tuple[np.ndarray,np.ndarray]:
        """
        Loads the training or testing dataset from the saved `.npz` files.

        Args:
            dataset (str, optional): Specifies which dataset to load. Must be either 'training' or 'testing'.
                                    Defaults to 'training'.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the input (X) and output (Y) arrays.

        Raises:
            ValueError: If `dataset` is not 'training' or 'testing'.

        """
        if dataset not in ['training', 'testing']:
            raise ValueError("Dataset must be 'training' or 'testing'.")

        path_to_data = self.training_set_npz if dataset == 'training' else self.testing_set_npz

        if not os.path.exists(path_to_data):
            raise FileNotFoundError(f"{path_to_data} not found. Ensure dataset is created.")

        data = load_npz(path_to_data, ['input', 'output'])
        return data['input'], data['output']
    
    def load_csv(self,dataset='testing') -> pd.DataFrame:
        if dataset not in ['training', 'testing']:
            raise ValueError("Dataset must be 'training' or 'testing'.")
        path_to_data = self.training_set_csv if dataset == 'training' else self.testing_set_csv
        if not os.path.exists(path_to_data):
            raise FileNotFoundError(f"{path_to_data} not found. Ensure dataset is created.")
        return pd.read_csv(path_to_data, usecols=['Molecular Formula','Molecular Weight','Pubchem_id','Energy'])
 



