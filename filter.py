import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
def count(formula):
    # Add "X" to the end of the formula
    new_formula = formula + "X"
    
    build_num = 0
    last_encountered = -1
    total_atoms = 0

    def is_alpha_numeric(value):
        if value.isalnum():
            if value.isupper():
                return True, 0  # Uppercase letter
            elif value.islower():
                return True, 1  # Lowercase letter
            elif value.isdigit():
                return True, 2  # Digit
        return False, -1  # Invalid character

    def is_valid(flag, last_encountered):
        if last_encountered == -1 and (flag == 1 or flag == 2):
            return False
        if last_encountered == 1 and flag == 1:
            return False
        if last_encountered == 2 and flag == 1:
            return False
        return True

    for character in new_formula:
        is_valid_char, flag = is_alpha_numeric(character)
        if not is_valid_char:
            return -1

        if not is_valid(flag, last_encountered):
            return -1

        if flag == 2:  # Digit
            build_num = build_num * 10 + int(character)
        elif last_encountered in (flag, flag + 1):
            total_atoms += 1
            build_num = 0
        elif last_encountered == 2 and flag == 0:
            total_atoms += build_num
            build_num = 0

        last_encountered = flag

    return total_atoms

def count_from_dataset(data_path:os.PathLike,store_to:os.PathLike=None):
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

def histogram(dataset:pd.DataFrame,column:str,save_to:os.PathLike,filename:str=""):
    plt.figure(figsize=(8, 6))
    dataset[column].hist(bins=10, color='skyblue', edgecolor='black')
    plt.title('Histogram of {}'.format(column), fontsize=14)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(False)
    plt.savefig(os.path.join(save_to,filename+column+".png"))

def filter_by_column(dataset:pd.DataFrame,column:str,filters:list):
    if len(filters)>2:return 
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
