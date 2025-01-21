from molecule import Molecule
import numpy as np
import pandas as pd
import os

def fill_up(big_array:np.ndarray,coulomb_vector:np.ndarray, index:int=0):
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

def main():
    molecules=pd.read_csv('data/filtered_molecules.csv')
    molecules.dropna(inplace=True)
    max_atoms = molecules['Atom Count'].sort_values(ignore_index=True,ascending=False)[0]
    N = int(max_atoms*(max_atoms+1)/2)
    big_array = np.zeros((molecules.shape[0],N))
    energies=molecules['Energy'].to_numpy()
    i = 0
    for compound in molecules['Pubchem_id']:
        file = str(compound) + '.gjf'
        path = os.path.join('gaussian/input',file)
        mol = Molecule(name='test', gjf_file=path)
        c_matrix = mol.coulomb_matrix
        fill_up(big_array,coulomb_vector=c_matrix,index=i)
        del c_matrix,mol
        i+=1
        if (i % 500 == 0):print(f"{i} done")
    del molecules
    np.savez("complete_dataset.npz",input=big_array,output=energies)

if __name__ == "__main__":
    main()
