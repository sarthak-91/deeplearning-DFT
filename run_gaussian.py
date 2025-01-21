import pandas as pd
import subprocess as sb
import os
import numpy as np

def get_energy(log_file:os.PathLike,pubchem_id:int,record:dict,energies:dict) -> float:
    energy_proc=sb.run("grep 'SCF Done' {} | tail -n 1 | cut -d ' ' -f 8".format(log_file),shell=True,executable='/usr/bin/bash',capture_output=True,text=True)
    energy=energy_proc.stdout
    record[pubchem_id] = 0
    energies[pubchem_id] = float(energy)


def check_termination(log_file:os.PathLike) -> bool:
    if not os.path.exists(log_file):
        return False
    check_done = sb.run("grep -q 'Normal termination of Gaussian' {}".format(log_file),shell=True,executable="/usr/bin/bash")
    if check_done.returncode != 0: 
        return False 
    return True

def gaussian(molecules:pd.DataFrame,data_path:os.PathLike):
    molecules.dropna(inplace=True)
    record={}
    energies={}
    input_dir=os.path.join(os.getcwd(),"gaussian/input")
    log_dir = os.path.join(os.getcwd(),"gaussian/log")
    g16_path="/home/sarthak/g16/g16"
    os.makedirs(log_dir, exist_ok=True)

    total=len(molecules['Pubchem_id'])
    print("No of molecules:",total)
    i=0
    for idx,pub in molecules['Pubchem_id'].items():
        pub=int(pub)
        i+=1
        input_file=os.path.join(input_dir,str(pub) + ".gjf")
        log_file=os.path.join(log_dir,str(pub) + ".log")
        if check_termination(log_file=log_file): 
            get_energy(log_file=log_file,pubchem_id=pub,record=record,energies=energies)
            if (i%1000==0): print(i, pub, "done")
            continue
        if i % 10 == 0:print("{}/{} done. {:.2f}%".format(i,total,i*100/total)) 
        process=sb.run("{} {} {}".format(g16_path,input_file,log_file), shell=True,executable='/usr/bin/bash')
        if process.returncode==0: get_energy(log_file=log_file,pubchem_id=pub,record=record,energies=energies)  
        else: 
            record[pub] = np.nan
            energies[pub] = np.nan
    molecules=molecules.astype({'Pubchem_id':int})
    molecules['Gaussian_Exited'] = molecules['Pubchem_id'].map(record)
    molecules['Energy'] = molecules['Pubchem_id'].map(energies)
    molecules.to_csv(data_path,index=False)
def main():
    current_path = os.getcwd()
    data_path = os.path.join(current_path,"data/filtered_molecules.csv")
    molecules=pd.read_csv(data_path)
    gaussian(molecules,data_path)

if __name__=="__main__":
    main()
