#Modeling molecular energies with Deep Learning
This project uses machine learning model to predict molecular single point energies, typically calculated by computationally intensive Density Functional Theory (DFT). The goal is a faster energy estimation method. The molecular energies are coputed using Gaussian 16 software and the molecules are prepared as inputs to ML model by calculating their coulomb matrices. 

### Workflow Overview: 

Data Extraction: Extraction of initial molecular data from the ChEMBL database.
Data Filtering: Molecule filtering based on atom count and other criteria.
Getting 3D Structure: Retrieval of molecular structures from PubChem.
Energy Calculation: Calculation of molecular energies using Gaussian computational chemistry software.
Feature Engineering: Calculation of Coulomb matrices as molecular descriptors.
Dataset Creation: Preparation and storage of training and testing datasets.
Model Development: Training neural networks for molecular energy prediction.

For details of each step in the workflow:
https://sarthak-91.github.io/deeplearning-DFT/