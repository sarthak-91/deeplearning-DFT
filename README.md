# deep-learning-for-DFT
In this project we are using Multilayered Perceptron network to model single point energies of molecules calculated with the help of Density Functional Theory(DFT)

DFT can be used to calculate the value of the potential energy of a molecule. This value is called single point energy. It can be understood as a point in the potential energy surface of the molecule.
## Input of the Model
One of the problems in using machine learning techniques in finding properties of molecules is to
characterize the molecule uniquely in a way that we can feed into a machine learning model. The method we have chosen in this project to perform that description is using Coulomb matrices.
The entries of coulomb matrix are defined as:
```math
C_{ij} =
\begin{cases}
0.5Z_i^{2.4}, & i = j \\
\frac{Z_i Z_j}{||R_i - R_j||}, & i \neq j
\end{cases}
```

where $$Z_i, Z_j$$ are atomic numbers, and $$R_i, R_j$$ are atomic positions in 3D space. The diagonal elements represent nuclear charge energy, while off-diagonal elements represent interatomic repulsion [@rupp_2012_fast].However since not all the molecules in the input space will be of equal size, we need to pad the coulomb matrix with zeros so that the inputs are all of equal size. Thus an input sample during training
will be:\
```math
\begin{bmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,n} & \cdots & 0 \\
x_{2,1} & x_{2,2} & \cdots & x_{2,n} & \cdots & 0 \\
\vdots  & \vdots  & \ddots & \vdots  & \cdots & \vdots \\
x_{n,1} & x_{n,2} & \cdots & x_{n,n} & \cdots & 0 \\
\vdots  & \vdots  & \vdots & \vdots  & \ddots & \vdots \\
0 & 0 & \cdots & \cdots & \cdots & 0
\end{bmatrix}
```

## Invariance of Input
For any input representation scheme to be used on a molecule, the representation must be invariant to
translation, rotation, permutation of the atoms in the molecule

1. Translation Invariance: Shifting a molecule in space does not change distances between atoms, ensuring \( C_{ij} \) remains unchanged.
2. Rotational Invariance: Since rotations also preserve interatomic distances, the Coulomb matrix remains invariant under rotation.
3. Permutation Invariance: Different atom orderings can produce different matrices. To standardize, we sort atoms by atomic number and then by distance from the molecular center of mass.

## Output of the Model
The model outputs single point energy of a given input molecule. Single point energy means the potential
energy of the molecule at a given state. The model was trained on energy calculated using Gaussian 09
software. The output space during training will be of the form:
```math
E = \begin{bmatrix} E_1, E_2, \dots, E_n \end{bmatrix}
```

where each $ E_i $ corresponds to the energy of a specific molecule in the dataset. The Gaussian program outputs the energies in the
unit of Hatrees which is related to eV by 1 Hatree = 27.211 eV


### Data extraction
The molecular information is taken from chembl website, by selecting their small compounds filter and downloading the .tsv file. The .csv file contained 1898837 molecules with various columns: 
```
"ChEMBL ID";"Name";"Synonyms";"Type";"Max Phase";"Molecular Weight";"Targets";
"Bioactivities";"AlogP";"Polar Surface Area";"HBA";"HBD";"#RO5 Violations";
"#Rotatable Bonds";"Passes Ro3";"QED Weighted";"CX Acidic pKa";"CX Basic pKa";
"CX LogP";"CX LogD";"Aromatic Rings";"Structure Type";"Inorganic Flag";"Heavy Atoms";
"HBA (Lipinski)";"HBD (Lipinski)";"#RO5 Violations (Lipinski)";"Molecular Weight (Monoisotopic)";
"Np Likeness Score";"Molecular Species";"Molecular Formula";"Smiles";
"Inchi Key";"Inchi";"Withdrawn Flag";"Orphan"
```
The chembl website contains other filters such as molecular weight, ,but as we will see when we build the model, number of atoms matters more than molecular weight so we will filter them we download it later. The first thing to do is select the relevant features we want from this dataset. Since this is not our real dataset (we are concerned with 3d molecular structures which are not available in chembl) we will only used the ChEMBL ID, Molecular Weight, Molecular Formula and Smiles columns, with which we can filter out the molecules that we want. The .csv file is very big around 825mb, so intead of using pandas to select columns, I am going to use awk for this.The columns we concerned with is 1st, 6th, 31st and 32nd columns.
```shell
awk -F "\t" -v OFS=, '{print $1,$6,$31,$32}' data/full_chembl.tsv > data/molecule_info.csv
sed 's/\"//g' -i data/molecule_info.csv
```
Now as mentioned before we would like to filter the molecule based on their number of atoms, so we need to create a field containing the number of atoms in each molecule which can be figured out by its Molecular Formula. The script filter.py does that for us. After that  We can read molecular formulas from molecule_info.csv that we just created and pass them to calculate the number of atoms and store it somewhere else. 

filter.py 
1. count no of atoms from and ass a column 
2. create histogram of entire dataset
3. filter atoms with number 20 to 80 
4. downsample the entire set to 10000 molecules

pubchem_extract.py
1. download compounds from pubchem using pubchempy
2. add a column pubchem_id to dataset

run_gaussian.py 
1. run gaussian on the gjf files 
2. record energies from log file 
3. add a column energy into the dataset 

Molecule class 
1. create a molecule from gjf file 
2. sort according to element and distance to center 
3. find coulomb matrix 

create_dataset.py
1. calculate coloumb matrix 
2. fill with zeros according to the biggest atom 
3. create a .npz file that as matrices as the key input and energies as key output

Data class

Network class 

main.py