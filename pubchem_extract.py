import pandas as pd 
import os
import pubchempy as pcp 
from threading import Thread
import numpy as np

def write_to_gjf(compound:pcp.Compound,threads="6",method="b3lyp",basis="6-31g"):
    parent_path=os.getcwd()
    gauss_files_path=os.path.join(parent_path,"gaussian")
    os.makedirs(gauss_files_path, exist_ok=True)
    input_files=os.path.join(gauss_files_path,"input")
    os.makedirs(input_files, exist_ok=True)
    chk_files=os.path.join(gauss_files_path,"chk")
    os.makedirs(chk_files, exist_ok=True)
    cid=str(compound.cid)
    gjf_file = os.path.join(input_files,cid+".gjf")
    chk_file = os.path.join(chk_files,cid+".chk")
    threadline=f"%nprocshared={threads}\n"
    chk_line=f"%chk={chk_file}\n"
    action_line=f"# {method}/{basis} geom=connectivity\n"
    title_line=f"\nEnergy {cid}\n\n"
    multiplicity_line="0 1\n"
    with open(gjf_file,"w") as file:
        file.write(threadline)
        file.write(chk_line)
        file.write(action_line)
        file.write(title_line)
        file.write(multiplicity_line)
        for atom in compound.atoms:
            position_line=" {}\t\t{:.6f}\t\t{:.6f}\t\t{:.6f}\n".format(atom.element,atom.x,atom.y,atom.z)
            file.write(position_line)
        bond_last=0
        bond=compound.bonds[0]
        first_connection=""
        other_connections=""
        for bond in compound.bonds:
            if bond.aid1 != bond_last:
                file.write(other_connections)
                first_connection="\n {} {} {:.1f} ".format(bond.aid1,bond.aid2, bond.order)
                other_connections=""
                if bond.aid1 - bond_last>1:
                    for i in range(bond_last+1,bond.aid1):file.write(f"\n {i}")
                file.write(first_connection)
                bond_last=bond.aid1
            else:
                aid2=bond.aid2
                other_connections += "{} {:.1f} ".format(aid2, bond.order)
        for i in range(bond_last+1,len(compound.atoms)+1):
            file.write(f"\n {i}")

def get_compound(list_of_smiles:pd.Series,split_index:int,dictionary):
    i=1
    for idx,value in list_of_smiles.items():
        try:
            if (i +1) % 50 == 0:print(split_index,(i+1)*100/len(list_of_smiles))
            compound_list=pcp.get_compounds(value,'smiles',record_type='3d')
            if compound_list==[]:continue
            compound=compound_list[0]
            write_to_gjf(compound=compound)
            dictionary[value]=compound.cid
        except Exception as error:
            print(i,value,error)
        i+=1

def run_thread(molecules:pd.DataFrame,data_path:os.PathLike,n_threads=4):
    if n_threads>8:raise ValueError("Too many threads")
    smiles=molecules['Smiles']
    split_data = np.array_split(smiles,n_threads)
    dict_map = [{} for i in range(n_threads)]
    threads= [Thread(target=get_compound,args=(split,idx,dict_map[idx],)) for idx,split in enumerate(split_data)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    while len(dict_map) != 1:
        dict_map[0].update(dict_map[-1])
        del dict_map[-1]
    molecules['Pubchem_id'] = molecules['Smiles'].map(dict_map[0])
    molecules.to_csv(data_path,index=False)

def main():
    current_path = os.getcwd()
    data_path = os.path.join(current_path,"data/filtered_molecules.csv")
    molecules=pd.read_csv(data_path)
    threads=3
    run_thread(molecules,data_path=data_path,n_threads=threads)
    return 0


if __name__ == "__main__":
    main()