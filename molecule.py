import numpy as np
import json
from numpy.linalg import norm

with open('elements.json', 'r') as file:
    element_dict = json.load(file)
def sort_by_norm(coulomb_matrix: np.ndarray):
    """
    Sort rows of the Coulomb matrix by their vector norms in descending order.

    Args:
        coulomb_matrix (np.ndarray): The Coulomb matrix to be sorted.

    Returns:
        np.ndarray: The sorted Coulomb matrix.
    """
    return np.array(sorted(coulomb_matrix, key=lambda x: norm(x), reverse=True))


class Molecule:
    """
    Represents a molecule, including its atoms, geometry, and properties.

    Attributes:
        name (str): Name of the molecule.
        atoms (list): List of Atom objects representing the atoms in the molecule.
        atom_count (int): Number of atoms in the molecule.
        mass (float): Total mass of the molecule.
        center (np.ndarray): Center of mass of the molecule.
        coordinates (np.ndarray): Positions of atoms in the molecule.
        coulomb_matrix (np.ndarray): Coulomb matrix representing molecular interactions.
        bond_matrix (np.ndarray): Matrix representing the bond connections between atoms.
    """
    def __init__(self, name, gjf_file: str = '', distance_sorting=1, norm_sorting=0):
        """
        Initializes the molecule with optional geometry from a Gaussian input file.

        Args:
            name (str): Name of the molecule.
            gjf_file (str): Path to the Gaussian input (.gjf) file. Default is an empty string.
            distance_sorting (int): Whether to sort atoms by distance to the center of mass. Default is 1.
            norm_sorting (int): Whether to sort the Coulomb matrix by row norms. Default is 0.
        """
        self.name = name
        self.atoms = []
        self.atom_count = 0
        self.mass = 0
        self.gjf_file = gjf_file

        # Load molecule geometry if gjf file is provided
        if gjf_file:
            self.read_from_gjf(gjf_file)

        self.center = self.calculate_center_of_mass()
        self.update_distances_from_center()
        self.calculate_spherical_coordinates()

        # Generate Coulomb matrix and sort if necessary
        if distance_sorting:
            self.sort_by_distance()
            self.coulomb_matrix = self.calculate_coulomb_matrix()
        elif norm_sorting:
            self.coulomb_matrix = self.calculate_coulomb_matrix(sort_by_norm=True)
        else:
            self.coulomb_matrix = self.calculate_coulomb_matrix()

        # Parse bond matrix if file provided
        self.bond_matrix = self.extract_bonds()

    def read_from_gjf(self, gjf_file: str):
        """
        Reads atomic data (symbols and coordinates) from a Gaussian input file.

        Args:
            gjf_file (str): Path to the Gaussian input file.
        """
        with open(gjf_file, encoding="utf8", errors="ignore") as file:
            lines = file.readlines()

        # Parse atomic symbols and coordinates
        self.atoms = [
            Atom(line.split()[0])
            for line in lines[7:]
            if len(line.split()) == 4 and not line.split()[0].isnumeric()
        ]
        self.atom_count = len(self.atoms)

        coordinates = np.array([
            [float(coord) for coord in line.split()[1:]]
            for line in lines[7:7 + self.atom_count]
        ])
        for atom, coord in zip(self.atoms, coordinates):
            atom.position = coord
        self.coordinates = coordinates

    def calculate_center_of_mass(self):
        """
        Calculates the center of mass of the molecule.

        Returns:
            np.ndarray: The center of mass coordinates.
        """
        total_mass = sum(atom.mass for atom in self.atoms)
        weighted_positions = sum(atom.mass * atom.position for atom in self.atoms)
        self.mass = total_mass
        return weighted_positions / total_mass

    def update_distances_from_center(self):
        """
        Updates the distance of each atom from the center of mass.
        """
        for atom in self.atoms:
            atom.distance_to_center = np.linalg.norm(atom.position - self.center)

    def calculate_spherical_coordinates(self):
        """
        Calculates the spherical coordinates of each atom relative to the center of mass.
        """
        for atom in self.atoms:
            relative_position = atom.position - self.center
            r = np.linalg.norm(relative_position)
            theta = np.arctan2(relative_position[1], relative_position[0])
            phi = np.arccos(relative_position[2] / r if r != 0 else 0)
            atom.theta_to_center = theta
            atom.phi_to_center = phi

    def sort_by_distance(self):
        """
        Sorts atoms by atomic number and distance from the center of mass.
        """
        self.atoms.sort(
            key=lambda atom: (atom.atomic_number, atom.distance_to_center)
        )
        self.coordinates = np.array([atom.position for atom in self.atoms])

    def calculate_coulomb_matrix(self, sort_by_norm=False):
        """
        Calculates the Coulomb matrix for the molecule.

        Args:
            sort_by_norm (bool): Whether to sort rows by vector norm. Default is False.

        Returns:
            np.ndarray: The Coulomb matrix.
        """
        positions = np.array([atom.position for atom in self.atoms])
        atomic_numbers = np.array([atom.atomic_number for atom in self.atoms])
        c_matrix = np.zeros((self.atom_count, self.atom_count))

        for i in range(self.atom_count):
            for j in range(self.atom_count):
                if i == j:
                    c_matrix[i, j] = 0.5 * atomic_numbers[i] ** 2.4
                else:
                    distance = np.linalg.norm(positions[i] - positions[j])
                    c_matrix[i, j] = atomic_numbers[i] * atomic_numbers[j] / distance

        return sort_by_norm(c_matrix) if sort_by_norm else c_matrix

    def extract_bonds(self):
        """
        Extracts bond connectivity from the Gaussian input file.

        Returns:
            np.ndarray: The bond matrix.
        """
        bond_matrix = np.zeros((self.atom_count, self.atom_count))
        with open(self.gjf_file, 'r') as file:
            lines = file.readlines()[7 + self.atom_count:]
        for line in lines:
            data = line.split()
            print(data)
            if data == [] or len(data) == 1:
                continue
            elif len(data) == 3:
                i, j, bond = map(float, data)
                bond_matrix[int(i) - 1, int(j) - 1] = bond
                bond_matrix[int(j) - 1, int(i) - 1] = bond
            else:
                i = int(data[0]) - 1
                for k in range(1, len(data), 2):
                    j, bond = int(data[k]) - 1, float(data[k + 1])
                    bond_matrix[i, j] = bond
                    bond_matrix[j, i] = bond
        return bond_matrix

    def __str__(self):
        """
        Prints details about the molecule and its atoms.
        """
        print(f"Name: {self.name}")
        print(f"Number of atoms: {self.atom_count}")
        for idx, atom in enumerate(self.atoms):
            print(f"{idx}: {atom}")
        return ""


class Atom:
    """
    Represents an atom in the molecule.

    Attributes:
        symbol (str): Chemical symbol of the atom.
        atomic_number (int): Atomic number of the atom.
        mass (float): Atomic mass of the atom.
        position (np.ndarray): Cartesian coordinates of the atom.
        distance_to_center (float): Distance of the atom from the center of mass.
        theta_to_center (float): Theta angle (spherical coordinates).
        phi_to_center (float): Phi angle (spherical coordinates).
    """
    def __init__(self, symbol: str):
        """
        Initializes an atom with its symbol and default properties.

        Args:
            symbol (str): Chemical symbol of the atom.
        """
        self.symbol = symbol
        self.atomic_number = element_dict[symbol][0]
        self.mass = element_dict[symbol][1]
        self.position = np.zeros(3)
        self.distance_to_center = 0
        self.theta_to_center = 0
        self.phi_to_center = 0

    def __str__(self):
        """
        Returns a string representation of the atom.
        """
        return f"{self.symbol} - Position: {self.position}, Distance to Center: {self.distance_to_center:.3f}"
