import numpy as np
import json
from numpy.linalg import norm

with open('elements.json', 'r') as file:
    elements= json.load(file)


def upper_triangular_to_vector(symmetric_array):

    if not isinstance(symmetric_array, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    
    if symmetric_array.ndim != 2 or symmetric_array.shape[0] != symmetric_array.shape[1]:
        raise ValueError("Input must be a square matrix.")
    
    return symmetric_array[np.triu_indices_from(symmetric_array)]


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
    def __init__(self, gjf_file: str,name:str='', distance_sorting=1):
        """
        Initializes the molecule with optional geometry from a Gaussian input file.

        Args:
            name (str): Name of the molecule.
            gjf_file (str): Path to the Gaussian input (.gjf) file.
            distance_sorting (int): Whether to sort atoms by distance to the center of mass. Default is 1.
        """
        self.name = name
        self.atoms = []
        self.bonds = []
        self.mass = 0
        self.gjf_file = gjf_file

        # Load molecule geometry if gjf file is provided
        
        self.read_from_gjf(elements,gjf_file)
        self.center = self.calculate_center_of_mass()
        self.update_distances_from_center()
        self.atom_count=len(self.atoms)

        # Generate Coulomb matrix and sort if necessary
        if distance_sorting:
            self.sort_by_distance()
        self.coulomb_matrix = self.calculate_coulomb_matrix()

    def create_atom(self,elements, line, atom_id):
        split_line = line.split()
        symbol = split_line[0]
        position = np.array(list(map(float, split_line[1:4])))
        return Atom(symbol, atom_id, position, elements)

    def create_bond(self, line):
     """
     Parses a line to create bonds and adds them to the bond list.

     Parameters:
     - bond_list (list): List to store Bond instances.
     - line (str): Line containing bond data.
     """
     split_line = line.split()
     atom_id1 = int(split_line[0])
     for i in range(1, len(split_line), 2):
         atom_id2 = int(split_line[i])
         order = float(split_line[i + 1])
         bond = Bond(atom_id1, atom_id2, order)
         self.bonds.append(bond)


    def read_from_gjf(self,elements, gjf_file):
        """
        Reads a Gaussian input file (.gjf) and populates atom_list and bond_list.

        Parameters:
        - atom_list (list): List to store Atom objects.
        - bond_list (list): List to store Bond objects.
        - elements (dict): Dictionary with element symbols as keys and [number, mass] as values.
        - gjf_file (str): Path to the Gaussian input file.
        """
        try:
            with open(gjf_file, 'r') as file:
                filled_line = 0
                atoms_encountered = False
                bonds_encountered = False
                atom_id = 0

                for line in file:
                    line = line.strip()
                    if not line:
                        if atoms_encountered:
                            bonds_encountered = True
                            atoms_encountered = False
                    elif filled_line >= 5:
                        atoms_encountered = not bonds_encountered
                        if atoms_encountered:
                            atom_id += 1
                            self.atoms.append(self.create_atom(elements, line, atom_id))
                        elif bonds_encountered:
                            if len(line.split()) != 1:
                                self.create_bond(line)
                    if line:
                        filled_line += 1
        except FileNotFoundError:
            print(f"Could not open file {gjf_file}")


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


    def sort_by_distance(self):
        """
        Sorts atoms by atomic number and distance from the center of mass.
        """
        self.atoms.sort(
            key=lambda atom: (-atom.number, atom.distance_to_center)
        )
        self.coordinates = np.array([atom.position for atom in self.atoms])

    def calculate_coulomb_matrix(self):
        """
        Calculates the Coulomb matrix for the molecule.

        Args:
            sort_by_norm (bool): Whether to sort rows by vector norm. Default is False.

        Returns:
            np.ndarray: The Coulomb matrix.
        """
        positions = np.array([atom.position for atom in self.atoms])
        numbers = np.array([atom.number for atom in self.atoms])
        c_matrix = np.zeros((self.atom_count, self.atom_count))

        for i in range(self.atom_count):
            for j in range(self.atom_count):
                if i == j:
                    c_matrix[i, j] = 0.5 * numbers[i] ** 2.4
                else:
                    distance = np.linalg.norm(positions[i] - positions[j])
                    c_matrix[i, j] = numbers[i] * numbers[j] / distance

        return c_matrix



    def __str__(self):
        """
        Prints details about the molecule and its atoms.
        """
        print(f"Name: {self.name}")
        print(f"Number of atoms: {self.atom_count}")
        for idx, atom in enumerate(self.atoms):
            print(f"{idx}: {atom}")
        for  ifx,bond in enumerate(self.bonds):
            print(f"{idx}: {bond}")
        return ""


class Atom:
    """
    Represents an atom with an element, ID, and position.
    """
    def __init__(self, symbol, atom_id, position, elements):
        """
        Initialize an Atom instance.

        Parameters:
        - symbol (str): Element symbol (e.g., "H", "O").
        - atom_id (int): Atom ID.
        - position (list of float): Position of the atom [x, y, z].
        - elements (dict): Dictionary with element symbols as keys and [number, mass] as values.
        """
        if symbol not in elements:
            raise ValueError(f"Element symbol '{symbol}' not found in elements dictionary.")
        
        self.symbol = symbol
        self.number = float(elements[symbol]['N'])
        self.mass = float(elements[symbol]['M'])
        self.atom_id = atom_id
        self.position = position  

    def __repr__(self):
        return f"Atom(symbol={self.symbol}, id={self.atom_id}, position={self.position})"

class Bond:
    """
    Represents a bond between two atoms.
    """
    def __init__(self, atom_id1, atom_id2, order):
        """
        Initialize a Bond instance.

        Parameters:
        - atom_id1 (int): ID of the first atom.
        - atom_id2 (int): ID of the second atom.
        - order (float): Bond order.
        """
        self.atom_id1 = atom_id1
        self.atom_id2 = atom_id2
        self.order = order

    def __repr__(self):
        return f"Bond(atom_id1={self.atom_id1}, atom_id2={self.atom_id2}, order={self.order})"
