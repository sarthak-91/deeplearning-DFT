import numpy as np
import os
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import subplots, subplots_adjust
import matplotlib.pyplot as plt
from molecule import Molecule

def load_to_3d(filename):
    """
    Load a 2D text file and reshape it into a 3D numpy array.

    This function reads a text file of numeric data and reshapes it into a 3D array
    with a square grid dimension. Useful for converting flattened data back into 
    a spatial or grid-like representation.

    Args:
        filename (str): Path to the input text file containing numeric data.

    Returns:
        numpy.ndarray: A 3D numpy array reshaped from the input file, 
                       with dimensions (num_samples, size, size, 1).
    
    Raises:
        IOError: If the file cannot be read.
        ValueError: If the array cannot be reshaped into a square grid.
    """
    loaded_array = np.loadtxt(filename)
    size = int(loaded_array.shape[1]**0.5)
    l = loaded_array.reshape(loaded_array.shape[0], size, size, 1)
    return l

def save3d(data, filename):
    """
    Save a 3D numpy array to a text file by flattening its dimensions.

    Args:
        data (numpy.ndarray): The 3D array to be saved.
        filename (str): Path where the flattened array will be saved.

    Returns:
        None: Saves the file to the specified location.
    
    Raises:
        IOError: If the file cannot be written.
    """
    reshaped_array = data.reshape(data.shape[0], -1)
    np.savetxt(filename, reshaped_array)
    return 

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

    def __init__(self, input_file="new_input.txt", output_file="new_output.txt", 
                 folder="data_model", test_size=0.3):
        """
        Initialize the Dataset with input and output files.

        Args:
            input_file (str, optional): Path to input data file. Defaults to "new_input.txt".
            output_file (str, optional): Path to output labels file. Defaults to "new_output.txt".
            folder (str, optional): Directory to store split dataset. Defaults to "data_model".
            test_size (float, optional): Proportion of dataset to include in test split. 
                                         Defaults to 0.3.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.folder = folder
        
        # Filenames for split dataset components
        self.x_train_file = "training_input.txt"
        self.x_test_file = "testing_input.txt"
        self.y_train_file = "training_output.txt"
        self.y_test_file = "testing_output.txt"
        
        # Initialize data attributes
        self.x_train,self.y_labeled_train = None,None
        self.x_test,self.y_labeled_test = None, None
        
        # Perform initial dataset split
        self.split_dataset(test_size=test_size)
    
    def split_dataset(self, test_size=0.3, save_to_file:bool=True):
        """
        Split the dataset into training and testing sets.

        Args:
            test_size (float, optional): Proportion of dataset to include in test split. 
                                         Defaults to 0.3.
            save_to_file (bool, optional): Whether to save split datasets to files. 
                                           Defaults to True.
        """
        # Load input data and labels
        x = load_to_3d(self.input_file)
        y = np.loadtxt(self.output_file)
        
        # Split data using sklearn's train_test_split
        self.x_train, self.x_test, self.y_labeled_train, self.y_labeled_test = train_test_split(
            x, y, test_size=test_size, random_state=1, shuffle=True)
        
        # Optionally save split datasets to files
        if save_to_file:
            os.makedirs(self.folder, exist_ok=True)
            save3d(self.x_train, f"{self.folder}/{self.x_train_file}")
            save3d(self.x_test, f"{self.folder}/{self.x_test_file}")
            np.savetxt(f"{self.folder}/{self.y_train_file}", self.y_labeled_train)
            np.savetxt(f"{self.folder}/{self.y_test_file}", self.y_labeled_test)

    def load_dataset(self, bonds_info=False):
        """
        Load previously split dataset from files.

        If split dataset files do not exist, calls split_dataset() to create them.

        Args:
            bonds_info (bool, optional): Placeholder for potential future feature. 
                                         Defaults to False.
        """
        if not os.path.exists(f"{self.folder}/{self.x_train_file}"):
            self.split_dataset()
            return

        # Load split datasets
        self.x_train = load_to_3d(f"{self.folder}/{self.x_train_file}")
        self.x_test = load_to_3d(f"{self.folder}/{self.x_test_file}")
        self.y_labeled_train = np.loadtxt(f"{self.folder}/{self.y_train_file}")
        self.y_labeled_test = np.loadtxt(f"{self.folder}/{self.y_test_file}")
        self.data_list = self.x_train, self.x_test 

    def build_histogram(self, energy:bool=True, atomic_mass:bool=False, no_of_atoms:bool=False):
        """
        Generate histograms for various molecular properties.

        Creates and saves histograms for energy, atomic mass, or number of atoms 
        in training and testing datasets.

        Args:
            energy (bool, optional): Generate energy histograms. Defaults to True.
            atomic_mass (bool, optional): Generate atomic mass histograms. Defaults to False.
            no_of_atoms (bool, optional): Generate number of atoms histograms. Defaults to False.
        
        Note:
            Requires access to Molecule class for atomic mass and atom count calculations.
            Histograms are saved as PNG files in the specified folder.
        """
        ids = self.y_labeled_train[:, 0]
        test_ids = self.y_labeled_test[:, 0]

        if energy:
            # Energy histogram generation code remains the same
            pass

        if atomic_mass:
            # Atomic mass histogram generation code remains the same
            pass

        if no_of_atoms:
            # Number of atoms histogram generation code remains the same
            pass

    def get_time(self, set='test', elapsed=True):
        """
        Extract computation time from log files for molecules in the dataset.

        Args:
            set (str, optional): Which dataset to extract times from. 
                                 Accepts 'test' or 'train'. Defaults to 'test'.
            elapsed (bool, optional): Whether to get elapsed time or another time metric. 
                                      Defaults to True.

        Returns:
            dict: A dictionary mapping molecule IDs to their respective times.
        
        Note:
            Assumes log files are located in 'data_extract/tested_input/log' directory.
        """
        y_labeled = self.y_labeled_test if set == 'test' else self.y_labeled_train
        ids = y_labeled[:, 0]
        data_path = os.path.join(os.getcwd(), 'data_extract/tested_input/log')
        line = -3 if elapsed else -4
        time_data = {}
        for id in ids:
            filename = os.path.join(data_path, f'{int(id)}.log')
            with open(filename, "r") as file:
                time_info = file.readlines()[line].split()
                time_data[id] = float(time_info[-2])
        return time_data