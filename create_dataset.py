from molecule import Molecule
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def fill_up(big_array:np.ndarray,coulomb_vector:np.ndarray, index:int=0):
    """ Fills a portion of a 1D or 2D array (`big_array`) with a 1D Coulomb vector (`coulomb_vector`).

    Args:
        big_array (np.ndarray): The target array to be filled. Can be 1D or 2D.
        coulomb_vector (np.ndarray): The 1D Coulomb vector to be projected into `big_array`.
        index (int, optional): The row index to fill if `big_array` is 2D. Defaults to 0.

    Raises:
        ValueError: If `coulomb_vector` is not 1D.
        ValueError: If `big_array` is not 1D or 2D.
        ValueError: If the size of `coulomb_vector` exceeds the corresponding dimension of `big_array`.
        ValueError: If `index` is out of bounds for a 2D `big_array`.

    """

    if len(coulomb_vector.shape) > 1: 
        raise ValueError("Coulomb Matrices must be passed as one dimensional instead got {}".format(coulomb_vector.shape))
    
    if len(big_array.shape) not in [1, 2]:
        raise ValueError("Array to be projected into must be 1D or 2D.")
    
    if coulomb_vector.shape[0] > (big_array.shape[0] if big_array.ndim == 1 else big_array.shape[1]):
        raise ValueError(
            f"The size of the coulomb vector ({coulomb_vector.shape[0]}) is larger than the corresponding dimension of big_array {big_array.shape}"
        )
    
    if big_array.ndim == 2 and index >= big_array.shape[0]:
        raise ValueError(
            f"Index ({index}) should be below the number of rows in big_array ({big_array.shape[0]})."
        )
    M = len(coulomb_vector)
    if len(big_array.shape) == 1:
        big_array[:M] =coulomb_vector
    else: 
        big_array[index][:M] = coulomb_vector
def create_npz(molecules:pd.DataFrame, npz_path:str, dataset_path:str, max_N):
    N = int(max_N*(max_N+1)/2)
    big_array = np.zeros((molecules.shape[0],N))
    energies=molecules['Energy'].to_numpy()
    i = 0
    for compound in molecules['Pubchem_id']:
        file = str(compound) + '.gjf'
        path = os.path.join('gaussian/input',file)
        #Create Molecule object and calculate coulomb matrix. 
        mol = Molecule(name='test', gjf_file=path)
        c_matrix = mol.coulomb_matrix
        fill_up(big_array,coulomb_vector=c_matrix,index=i)
        del c_matrix,mol
        i+=1
        if (i % 500 == 0):print(f"{i} done")
    molecules.to_csv(dataset_path,index=False)
    del molecules
    np.savez(npz_path,input=big_array,output=energies)



def create_train_test(molecules:pd.DataFrame,
                npz_folder='datasets',dataset_folder='datasets',
                test_size:float=0.3, random_state:int =40):
    
    molecules.dropna(inplace=True)
    max_atoms = molecules['Atom Count'].sort_values(ignore_index=True,ascending=False)[0]
    train_set, test_set = train_test_split(molecules, test_size=test_size,shuffle=True, random_state=random_state)
    print("Split into training and testing")
    print("Creating training set")
    create_npz(molecules=train_set,npz_path=f'{npz_folder}/training_set.npz',dataset_path=f'{dataset_folder}/training_set.csv',max_N=max_atoms)
    print("Creating testing set")

def create_npz(molecules:pd.DataFrame, npz_path:str, dataset_path:str, max_N):
    N = int(max_N*(max_N+1)/2)
    big_array = np.zeros((molecules.shape[0],N))
    energies=molecules['Energy'].to_numpy()
    i = 0
    for compound in molecules['Pubchem_id']:
        file = str(compound) + '.gjf'
        path = os.path.join('gaussian/input',file)
        #Create Molecule object and calculate coulomb matrix. 
        mol = Molecule(name='test', gjf_file=path)
        c_matrix = mol.coulomb_matrix
        fill_up(big_array,coulomb_vector=c_matrix,index=i)
        del c_matrix,mol
        i+=1
        if (i % 500 == 0):print(f"{i} done")
    molecules.to_csv(dataset_path,index=False)
    del molecules
    np.savez(npz_path,input=big_array,output=energies)



def create_train_test(molecules:pd.DataFrame,
                npz_folder='datasets',dataset_folder='datasets',
                test_size:float=0.3, random_state:int =40):
    
    molecules.dropna(inplace=True)
    max_atoms = molecules['Atom Count'].sort_values(ignore_index=True,ascending=False)[0]
    train_set, test_set = train_test_split(molecules, test_size=test_size,shuffle=True, random_state=random_state)
    print("Split into training and testing")
    print("Creating training set")
    create_npz(molecules=train_set,npz_path=f'{npz_folder}/training_set.npz',dataset_path=f'{dataset_folder}/training_set.csv',max_N=max_atoms)
    print("Creating testing set")
    create_npz(molecules=test_set,npz_path=f'{npz_folder}/testing_set.npz',dataset_path=f'{dataset_folder}/testing_set.csv',max_N=max_atoms)

def main():
    molecules=pd.read_csv('data/filtered_molecules.csv')
    os.makedirs('datasets',exist_ok=True)
    create_train_test(molecules=molecules)

if __name__ == "__main__":
    main()