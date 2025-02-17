import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
GAUSSIAN_DIR = os.path.join(PROJECT_ROOT, "gaussian")
VISUALIZATION_DIR = os.path.join(PROJECT_ROOT, "visualizations")

ELEMENTS_JSON = os.path.join(PROJECT_ROOT,"elements.json")
MOLECULE_CSV = os.path.join(DATA_DIR, "filtered_molecules.csv")
TRAINING_NPZ = os.path.join(DATASETS_DIR, "training_set.npz")
TESTING_NPZ = os.path.join(DATASETS_DIR, "testing_set.npz")
TRAINING_CSV = os.path.join(DATASETS_DIR,"training_set.csv")
TESTING_CSV = os.path.join(DATASETS_DIR, "testing_set.csv")
