# Modeling molecular energies with Deep Learning
This project uses machine learning model to predict molecular single point energies, typically calculated by computationally intensive Density Functional Theory (DFT). The project aims to put forth ML methods as a fater method to calculate energies of molecules specially when a quick and dirty calculation is only desired. The molecular energies are calculated using Gaussian 16 software and the molecules are prepared as inputs to ML model by calculating their coulomb matrices. 

### Workflow Overview: 

1. Data Extraction: Extraction of initial molecular data from the ChEMBL database.
2. Data Filtering: Molecule filtering based on atom count and other criteria.
3. Getting 3D Structure: Retrieval of molecular structures from PubChem.
4. Energy Calculation: Calculation of molecular energies using Gaussian computational chemistry software.
5. Feature Engineering: Calculation of Coulomb matrices as molecular descriptors.
6. Dataset Creation: Preparation and storage of training and testing datasets.
7. Model Development: Training neural networks for molecular energy prediction.

For details of each step in the workflow:
https://sarthak-91.github.io/deeplearning-DFT/