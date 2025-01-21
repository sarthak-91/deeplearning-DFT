import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from visuals import histogram 
def count(formula:str) -> int | int:
    """Function for counting number of atoms from a given formula
    Args:
        formula (str): A molecular formula as a string. 

    Returns:
        int: 
            - Returns number of atom from a formula is the formula is valid 
            - Returns -1 if formula is invalid 
    """
    # Add "X" to the end of the formula
    new_formula = formula + "X"
    
    build_num = 0 
    last_encountered = -1
    total_atoms = 0

    def is_alpha_numeric(value:str) -> tuple[bool,int]:
        """Check if a character is alpha numeric or not

        Args:
            value (str): A character 

        Returns:
            tuple[bool,int]: 
                - True, 0 if value is Uppercase letter
                - True, 1 if value is Lowercase letter 
                - True, 2 if value is a number
                - False, -1 is value is not alpha numeric 

        """
        if value.isalnum():
            if value.isupper():
                return True, 0 
            elif value.islower():
                return True, 1 
            elif value.isdigit():
                return True, 2 
        return False, -1  

    def is_valid(flag:int, last_encountered:int) -> bool | bool:
        """Checks if a formula is valid or not by comparing the type of previous character encountered to the next character

        Args:
            flag (int): type of current character, 0-> Uppercase, 1-> Lowercase, 2-> Number
            last_encountered (int): Type of last character encountered, 0-> Uppercase, 1-> Lowercase, 2-> Number

        Returns:
            bool: 
                -False if invalid combination of characters encountered in formula
                - True if formula is valid
        """
        if last_encountered == -1 and (flag == 1 or flag == 2): #if the first character is a small letter or a number
            return False
        if last_encountered == 1 and flag == 1: # if two lowercase characters encountered one after the other
            return False
        if last_encountered == 2 and flag == 1: #if lowercase character encountered after a number 
            return False
        return True

    for character in new_formula:
        is_valid_char, flag = is_alpha_numeric(character)
        if not is_valid_char:
            return -1

        if not is_valid(flag, last_encountered):
            return -1

        if flag == 2: 
            build_num = build_num * 10 + int(character)
        elif last_encountered in (flag, flag + 1):
            total_atoms += 1
            build_num = 0
        elif last_encountered == 2 and flag == 0:
            total_atoms += build_num
            build_num = 0

        last_encountered = flag

    return total_atoms

def count_from_dataset(data_path:os.PathLike,store_to:os.PathLike=None) -> pd.DataFrame:
    """Process molecular dataset. Drop NaN values in Molecular Formula and Smiles columns, count the number of atoms

    Args:
        data_path (os.PathLike): path to load the molecular dataset
        store_to (os.PathLike, optional):Path to store the dataset after processing

    Returns:
        pd.DataFrame: Processed dataset with added column of Atom Count
    """
    df = pd.read_csv(data_path,low_memory=False)
    print("loaded")
    print(df.columns)
    df.dropna(subset=["Molecular Formula","Smiles"],inplace=True)
    print("Removed nans")
    df['Atom Count'] = df['Molecular Formula'].apply(count)
    df.dropna(subset=["Atom Count"],inplace=True)
    print("Counted Atoms")
    valid_df = df[df['Atom Count'] > 0]
    print("Filtered")
    if store_to: valid_df.to_csv(store_to,index=False)
    return valid_df


def filter_by_column(dataset:pd.DataFrame,column:str,filters:list) -> pd.DataFrame:
    """Filter the dataset according to a column and range 

    Args:
        dataset (pd.DataFrame): dataset to filter
        column (str): column name to apply filter on
        filters (list): range of filter [low,high]

    Raises:
        ValueError: if filter has more than two values or if lower bound of filter is higher than upper bound

    Returns:
        pd.DataFrame: resulting filtered dataset
    """
    if len(filters)>2 or filter[0] > filter[1]:raise ValueError("The filters list should contain only two numbers: [low,high]. Provided list: {}".format(filters))
    filtered =  dataset.loc[(dataset[column] >= filters[0]) & (dataset[column] <= filters[1])]
    return filtered


if __name__ == "__main__":
    current_path = os.getcwd()
    data_path = os.path.join(current_path,"data/molecule_info.csv")
    store_path = os.path.join(current_path,"data/valid_molecules.csv")

    mol_valid_formulas = count_from_dataset(data_path,store_to=store_path)
    print("counted number of atoms")
    if not os.path.exists("visualizations"):os.makedirs("visualizations")
    histogram(dataset=mol_valid_formulas,column="Atom Count",save_to="visualizations",filename="entire_dataset_")
    
    first_filter_dataset = filter_by_column(dataset=mol_valid_formulas,column="Atom Count",filters=[0,200])
    histogram(dataset=first_filter_dataset,column="Atom Count",save_to="visualizations",filename="0_200_")
    
    second_filter_dataset = filter_by_column(dataset=first_filter_dataset,column="Atom Count",filters=[20,80])
    histogram(dataset=second_filter_dataset,column="Atom Count",save_to="visualizations",filename="20_80_")
    
    sampled_down_dataset = second_filter_dataset.sample(n=10000)
    print("Sampled")
    histogram(dataset=sampled_down_dataset,column="Atom Count",save_to="visualizations",filename="sampled_down_")
    print("Saving")
    sampled_down_dataset.to_csv(os.path.join(current_path,"data/filtered_molecules.csv"),index=False)
