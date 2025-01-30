import numpy as np
import os
from sklearn.model_selection import train_test_split


def load_npz(npz_file:str,names:list) -> list:
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
    dict_of_arrays={}
    for name in names:dict_of_arrays[name] = array[name]
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

    def __init__(self, complete_dataset:str="complete_dataset.npz", training_set:str="training_set.npz", 
                 testing_set:str="testing_set.npz", test_size:float=0.3,resplit:bool=False, random_state:int=40):
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
        self.training_set_path = training_set
        self.testing_set_path= testing_set
        self.test_size = test_size
        self.random_state=random_state
        
        if not os.path.exists(complete_dataset):
            raise FileExistsError("Complete dataset not found, Run complete_set() from create_dataset module")
        
        if resplit==True or not (os.path.exists(training_set) and os.path.exists(testing_set)):
            self.split_dataset(test_size=test_size)

    
    def split_dataset(self,return_arrays=False):
        """ Splits the dataset into training and testing sets and saves them as `.npz` files.

        Args:
            return_arrays (bool, optional): If True, returns the split arrays (X_train, X_test, y_train, y_test).
                                           Defaults to False.

        Returns:
            Optional[tuple]: If `return_arrays` is True, returns a tuple of (X_train, X_test, y_train, y_test).
                            Otherwise, returns None.

        """
        print("Splitting dataset")
        # Load the complete dataset
        complete_set = load_npz(self.dataset_path,['input','output'])
        X = complete_set['input']
        Y = complete_set['output']

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=self.test_size, shuffle=True, random_state=self.random_state
        )
        # Save the training and testing sets as `.npz` files
        np.savez(self.training_set_path, input=X_train, output=y_train)
        np.savez(self.testing_set_path, input=X_test, output=y_test)

        if return_arrays:
            return X_train,X_test,y_train,y_test

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
        if dataset == 'training':
            path_to_data = self.training_set_path
        elif dataset == 'testing':
            path_to_data = self.testing_set_path
        else:
            raise ValueError("dataset should have value either 'training' or 'testing'") 
        
        # Load the dataset
        data = load_npz(path_to_data,['input','output'])
        X = data['input']
        Y = data['output']
        return X,Y

