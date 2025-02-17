import pandas as pd
import subprocess as sb
import os
import numpy as np
from scripts.config import PROJECT_ROOT

def get_energy(log_file:os.PathLike,pubchem_id:int,record:dict,energies:dict) -> float:
    """_summary_

    Args:
        log_file (os.PathLike): Path to the Gaussian log file containing the energy information.
        pubchem_id (int): The PubChem ID of the compound, used as a key in the `record` and `energies` dictionaries.
        record (dict): A dictionary to store a record of processed PubChem IDs (e.g., for tracking purposes).
        energies (dict): A dictionary to store the extracted energy values, indexed by PubChem ID.
    """
    #extract the SCF energy from the log file using grep
    energy_proc=sb.run("grep 'SCF Done' {} | tail -n 1 | cut -d ' ' -f 8".format(log_file),shell=True,executable='/usr/bin/bash',capture_output=True,text=True)
    energy=energy_proc.stdout.strip()
    record[pubchem_id] = 0 #Mark pubchemid as completed
    energies[pubchem_id] = float(energy) #Record energy 


def check_termination(log_file:os.PathLike) -> bool:
    """Checks if a Gaussian log file indicates normal termination of the calculation.

    Args:
        log_file Path to the Gaussian log file to be checked.
    Returns:
        bool: True if the log file exists and contains the normal termination message, False otherwise.

    """
    if not os.path.exists(log_file):
        return False
    
    #Check for phrase "Normal termination of Gaussian" in the log file
    check_done = sb.run(
        "grep -q 'Normal termination of Gaussian' {}".format(log_file),
        shell=True,
        executable="/usr/bin/bash")
    if check_done.returncode != 0: 
        return False 
    return True

def gaussian(molecules:pd.DataFrame,data_path:os.PathLike):
    """Runs Gaussian calculations for a set of molecules and saves the results to a CSV file.
    Assumes dataframe contains Pubchem_id column and Gaussian input files are named {pubchem_id}.gjf and creates log files as {pubchem_id}.log

    Args:
        molecules (pd.DataFrame): A DataFrame containing molecular data, including a 'Pubchem_id' column
        data_path (os.PathLike):  Path to save the output CSV file containing the results.
    """
    # Initialize dictionaries to store records and energies
    record = {}
    energies = {}

    # Define directories for input and log files
    input_dir = os.path.join(PROJECT_ROOT, "gaussian/input")
    log_dir = os.path.join(PROJECT_ROOT, "gaussian/log")

    # Define the path to the Gaussian executable
    g16_path = "/home/sarthak/g16/g16"

    # Create the log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    total=len(molecules['Pubchem_id'])
    print("No of molecules:",total)
    i=0
    for idx,pub in molecules['Pubchem_id'].items():
        pub=int(pub)  # Ensure Pubchem_id is an integer
        i+=1

        # Define paths for the input and log files
        input_file=os.path.join(input_dir,str(pub) + ".gjf")
        log_file=os.path.join(log_dir,str(pub) + ".log")
       
        # Check if the calculation has already completed successfully
        if check_termination(log_file=log_file): 
            get_energy(log_file=log_file,pubchem_id=pub,record=record,energies=energies)
            if (i%1000==0): print(i, pub, "done")
            continue
        
        if i % 10 == 0:print("{}/{} done. {:.2f}%".format(i,total,i*100/total)) 
        
        # Run the Gaussian calculation
        process=sb.run("{} {} {}".format(g16_path,input_file,log_file), 
                       shell=True,
                       executable='/usr/bin/bash')
        # Check if the calculation completed successfully
        if process.returncode==0: get_energy(log_file=log_file,pubchem_id=pub,record=record,energies=energies)  
        else: 
            record[pub] = np.nan
            energies[pub] = np.nan
    # Update the DataFrame with the results and save to path
    molecules=molecules.astype({'Pubchem_id':int})
    molecules['Gaussian_Exited'] = molecules['Pubchem_id'].map(record)
    molecules['Energy'] = molecules['Pubchem_id'].map(energies)
    molecules.to_csv(data_path,index=False)

def main():
    data_path = os.path.join(PROJECT_ROOT,"data/filtered_molecules.csv")
    molecules=pd.read_csv(data_path)
    gaussian(molecules,data_path)

if __name__=="__main__":
    main()
