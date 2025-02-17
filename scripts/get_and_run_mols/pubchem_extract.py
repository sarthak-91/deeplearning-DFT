import pandas as pd 
import os
import pubchempy as pcp 
import threading
import numpy as np
from scripts.config import PROJECT_ROOT


def write_to_gjf(compound:pcp.Compound,threads:int=7,method:str="hf",basis:str="sto-3g"):
    """From a Compound object write a Gaussian input file. 

    Args:
        compound (pcp.Compound): Compound object gotten using pubchempy 
        threads (int, optional): Number of threads dedicated for gaussian job Defaults to 7.
        method (str, optional): Method employed in Gaussian Defaults to "hf".
        basis (str, optional): Basis set to be used in Gaussian. Defaults to "sto-3g".
    """
    #Define paths for Input and checkout files

    gauss_dir=os.path.join(PROJECT_ROOT,"gaussian")
    input_dir=os.path.join(gauss_dir,"input")
    chk_dir=os.path.join(gauss_dir,"chk")

    #Create directories if they do not exist
    os.makedirs(gauss_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(chk_dir, exist_ok=True)

    #Define input files and checkout file
    cid=str(int(compound.cid))
    gjf_file = os.path.join(input_dir,cid+".gjf")
    chk_file = os.path.join(chk_dir,cid+".chk")

    #Define header lines
    threadline=f"%nprocshared={threads}\n"
    chk_line=f"%chk={chk_file}\n"
    action_line=f"# {method}/{basis} geom=connectivity\n"
    title_line=f"\nEnergy {cid}\n\n"
    multiplicity_line="0 1\n"

    with open(gjf_file,"w") as file:
        #Write the header lines
        file.write(threadline)
        file.write(chk_line)
        file.write(action_line)
        file.write(title_line)
        file.write(multiplicity_line)

        #Write atom co-ordinates
        for atom in compound.atoms:
            position_line=" {}\t\t{:.6f}\t\t{:.6f}\t\t{:.6f}\n".format(atom.element,atom.x,atom.y,atom.z)
            file.write(position_line)
        #Write bond information
        bond_last=0 
        bond=compound.bonds[0]
        first_connection=""
        other_connections=""
        for bond in compound.bonds:
            if bond.aid1 != bond_last:
                # Write previous connections and start a new line for the current atom
                file.write(other_connections) 
                first_connection="\n {} {} {:.1f} ".format(bond.aid1,bond.aid2, bond.order)
                other_connections=""

                # Fill in any atoms without connections
                if bond.aid1 - bond_last>1:
                    for i in range(bond_last+1,bond.aid1):file.write(f"\n {i}")
                file.write(first_connection)
                bond_last=bond.aid1
            else:
                aid2=bond.aid2
                other_connections += "{} {:.1f} ".format(aid2, bond.order)
        #Fill in any remaining atoms without connections
        for i in range(bond_last+1,len(compound.atoms)+1):
            file.write(f"\n {i}")

def get_compound(list_of_smiles:pd.Series,split_index:int,dictionary:dict):
    """Function to get Compound objects and write them as Gaussian input files from pubchem using Smiles from data

    Args:
        list_of_smiles (pd.Series): Series object containing list of Smiles descriptors 
        split_index (int): index to keep track of the thread running
        dictionary (dict): Dictionary mapping smiles to pubchem id
    """
    i=1
    for idx,value in list_of_smiles.items():
        try:
            if (i +1) % 50 == 0:print(split_index,(i+1)*100/len(list_of_smiles))
            compound_list=pcp.get_compounds(value,'smiles',record_type='3d')
            if compound_list==[]:continue
            compound=compound_list[0]
            write_to_gjf(compound=compound)
            dictionary[value]=int(compound.cid)
        except Exception as error:
            print(i,value,error)
        i+=1

def run_thread(molecules:pd.DataFrame,data_path:os.PathLike,n_threads:int=4):
    """Run get_compound function on multiple threads

    Args:
        molecules (pd.DataFrame): DataFrame containing molecular information
        data_path (os.PathLike): Path to save the new dataset contining Pubchem ids
        n_threads (int, optional): Number of threads to download Compound object from. Defaults to 4.

    Raises:
        ValueError: Raises Error if n_threads is set to more threads than what is available
    """
    if n_threads>8:raise ValueError("Too many threads")
    smiles=molecules['Smiles']

    #Split Smiles column into n_threads parts
    split_data = np.array_split(smiles,n_threads) 
    dict_map = [{} for i in range(n_threads)]

    #Create Threads, Start and Join them
    threads= [threading.Thread(target=get_compound,args=(split,idx,dict_map[idx],)) for idx,split in enumerate(split_data)] 
    [t.start() for t in threads]
    [t.join() for t in threads]
    
    #Combine all dictionaries into one dictionary 
    while len(dict_map) != 1:
        dict_map[0].update(dict_map[-1])
        del dict_map[-1]
    molecules['Pubchem_id'] = molecules['Smiles'].map(dict_map[0]) #Create new column Pubchem_id using the dictionary
    molecules.to_csv(data_path,index=False)

def main():
    data_path = os.path.join(PROJECT_ROOT,"data/filtered_molecules.csv")
    molecules=pd.read_csv(data_path)
    threads=4
    run_thread(molecules,data_path=data_path,n_threads=threads)
    return 0


if __name__ == "__main__":
    main()