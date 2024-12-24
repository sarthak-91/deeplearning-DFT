# deep-learning-for-DFT
Using Multilayered Perceptron network to model single point energies of molecules calculated with the help of Density Functional Theory

### Data extraction
The molecular information is taken from chembl website, by selecting their small compounds filter and downloading the .csv file. The .csv file contained 1898837 molecules with various columns: 
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
awk -F ";" -v OFS=, '{print $1,$6,$31,$32}' data/chembl.csv > data/molecule_info.csv
```
Now as mentioned before we would like to filter the molecule based on their number of atoms, so we need to create a field containing the number of atoms in each molecule which can be figured out by its Molecular Formula