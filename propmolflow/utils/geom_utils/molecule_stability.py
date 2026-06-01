"""
Molecular stability assessment module for 3D molecular generation evaluation.

This module implements stability validation by separating aromatic and non-aromatic 
bond contributions. This approach resolves the ambiguities inherent in treating aromatic 
bonds as having fractional bond orders (e.g., 1.5).
"""

import torch
import warnings
from rdkit import Chem

from propmolflow.utils.geom_utils.geom_drugs_valency_table import geom_drugs_h_tuple_valencies
from propmolflow.utils.geom_utils.utils import is_valid


def _is_valid_valence_tuple(combo, allowed, charge, element_symbol=None):
    """
    Validate a valence tuple against allowed valence configurations.
    
    Args:
        combo (tuple): (n_aromatic_bonds, valence_from_non_aromatic_bonds)
        allowed: Allowed valence configurations (tuple, list, set, or dict)
        charge (int): Formal charge of the atom
        element_symbol (str, optional): Element symbol for error reporting
        
    Returns:
        bool: True if the valence combination is valid
    """
    if isinstance(allowed, tuple):
        return combo == allowed
    elif isinstance(allowed, (list, set)):
        return combo in allowed
    elif isinstance(allowed, dict):
        # Check if charge exists in the dictionary - avoid silent failures
        if charge not in allowed:
            element_info = f" for element {element_symbol}" if element_symbol else ""
            warnings.warn(
                f"Missing charge state {charge}{element_info} in valency table. "
                f"Available charges: {list(allowed.keys())}. Assuming invalid.",
                UserWarning
            )
            return False
        return _is_valid_valence_tuple(combo, allowed[charge], charge, element_symbol)
    elif allowed == []:
        # Handle empty list case explicitly - this was previously silent
        element_info = f" for element {element_symbol}" if element_symbol else ""
        warnings.warn(
            f"Empty valency configuration{element_info} with charge {charge}. Assuming invalid.",
            UserWarning
        )
        return False
    return False


def compute_molecules_stability_from_graph(adjacency_matrices, numbers, charges, allowed_bonds=None,
        aromatic=True):
    """
    Compute molecular stability from graph representations using aromatic-aware valency validation.
    
    This function implements the core stability logic that properly handles aromatic systems. 
    It separates aromatic bonds from regular bonds, enabling chemically accurate validation.
    
    Args:
        adjacency_matrices (torch.Tensor): Bond adjacency matrices [batch, n_atoms, n_atoms]
                                         Values: 1=single, 2=double, 3=triple, 1.5=aromatic, 4=aromatic(legacy)
        numbers (torch.Tensor): Atomic numbers [batch, n_atoms]
        charges (torch.Tensor): Formal charges [batch, n_atoms]
        allowed_bonds (dict, optional): Valency lookup table. Defaults to geom_drugs_h_tuple_valencies
        aromatic (bool): Whether to expect aromatic bonds. If False, asserts no 1.5 or 4 bonds exist
        
    Returns:
        tuple: (stable_mask, n_stable_atoms, n_atoms)
            - stable_mask: Boolean mask indicating which molecules are fully stable
            - n_stable_atoms: Number of atoms with valid valency per molecule
            - n_atoms: Total number of atoms per molecule
            
    Example:
        >>> # Benzene ring with proper aromatic representation
        >>> adj = torch.tensor([[[0, 1.5, 0, 0, 0, 1.5],
        ...                      [1.5, 0, 1.5, 0, 0, 0], ...]])
        >>> numbers = torch.tensor([[6, 6, 6, 6, 6, 6]])  # All carbons
        >>> charges = torch.zeros_like(numbers)
        >>> stable, n_stable, n_atoms = compute_molecules_stability_from_graph(adj, numbers, charges)
    """
    # Handle single molecule input by adding batch dimension
    if adjacency_matrices.ndim == 2:
        adjacency_matrices = adjacency_matrices.unsqueeze(0)
        numbers = numbers.unsqueeze(0)
        charges = charges.unsqueeze(0)

    if allowed_bonds is None:
        allowed_bonds = geom_drugs_h_tuple_valencies

    # Validate aromatic bond consistency
    if not aromatic:
        # When aromatic=False, ensure no aromatic bonds are present
        # Note: Bond order 4 is legacy encoding used by many generative models for aromatic bonds
        # RDKit internally uses enum 12 but GetBondTypeAsDouble() returns 1.5 for aromatic bonds
        assert (adjacency_matrices == 1.5).sum() == 0 and (adjacency_matrices == 4).sum() == 0

    batch_size = adjacency_matrices.shape[0]
    stable_mask = torch.zeros(batch_size)
    n_stable_atoms = torch.zeros(batch_size)
    n_atoms = torch.zeros(batch_size)

    for i in range(batch_size):
        adj = adjacency_matrices[i]
        atom_nums = numbers[i]
        atom_charges = charges[i]

        mol_stable = True
        n_atoms_i, n_stable_i = 0, 0

        for j, (a_num, charge) in enumerate(zip(atom_nums, atom_charges)):
            if a_num.item() == 0:
                continue
                
            row = adj[j]
            
            # AROMATIC HANDLING LOGIC:
            # Count aromatic bonds (bond order 1.5) separately from regular bonds
            # This is crucial for chemical accuracy - aromatic systems cannot be
            # properly validated by simply summing fractional bond orders
            aromatic_count = int((row == 1.5).sum().item())
            
            # Calculate valence from non-aromatic bonds only
            # Mask out aromatic bonds (1.5) before summing
            normal_valence = float((row * (row != 1.5)).sum().item())
            
            # Create valence tuple: (n_aromatic_bonds, valence_from_regular_bonds)
            combo = (aromatic_count, int(normal_valence))
            
            # Look up element symbol and allowed valence configurations
            symbol = Chem.GetPeriodicTable().GetElementSymbol(int(a_num))
            allowed = allowed_bonds.get(symbol, {})

            if _is_valid_valence_tuple(combo, allowed, int(charge), symbol):
                n_stable_i += 1
            else:
                mol_stable = False

            n_atoms_i += 1

        stable_mask[i] = float(mol_stable)
        n_stable_atoms[i] = n_stable_i
        n_atoms[i] = n_atoms_i

    return stable_mask, n_stable_atoms, n_atoms


def compute_molecules_stability(rdkit_molecules, aromatic=True, allowed_bonds=None):
    """
    Compute stability metrics for a list of RDKit molecule objects.
    
    This function serves as a convenient wrapper around compute_molecules_stability_from_graph,
    converting RDKit molecules to the required tensor format for batch processing.
    
    RDKit Bond Type Conversion:
    - Single bonds (SINGLE) → 1.0
    - Double bonds (DOUBLE) → 2.0  
    - Triple bonds (TRIPLE) → 3.0
    - Aromatic bonds (AROMATIC) → 1.5
    
    Args:
        rdkit_molecules (list): List of RDKit Mol objects
        aromatic (bool): Whether aromatic bonds are expected in the molecules
        allowed_bonds (dict, optional): Custom valency table. Defaults to geom_drugs_h_tuple_valencies
        
    Returns:
        tuple: (validity, stability, n_stable_atoms, n_atoms)
            - validity: RDKit sanitization success per molecule
            - stability: Chemical valency validity per molecule  
            - n_stable_atoms: Count of atoms with valid valency per molecule
            - n_atoms: Total atom count per molecule
            
    Note:
        This function processes molecules individually to handle variable sizes,
        then aggregates results into tensors for consistency with the graph-based function.
    """
    stable_list, stable_atoms_list, atom_counts_list, validity_list = [], [], [], []

    for mol in rdkit_molecules:
        if mol is None:
            continue
            
        n_atoms = mol.GetNumAtoms()
        
        # Convert RDKit molecule to tensor representation
        adj = torch.zeros((1, n_atoms, n_atoms))
        numbers = torch.zeros((1, n_atoms), dtype=torch.long)
        charges = torch.zeros((1, n_atoms), dtype=torch.long)

        # Extract atomic information
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            numbers[0, idx] = atom.GetAtomicNum()
            charges[0, idx] = atom.GetFormalCharge()

        # Extract bond information with proper aromatic handling
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()
            # Create symmetric adjacency matrix
            adj[0, i, j] = adj[0, j, i] = bond_type
        # Compute stability using the graph-based function
        stable, stable_atoms, atom_count = compute_molecules_stability_from_graph(
            adj, numbers, charges, allowed_bonds, aromatic
        )
        
        # Collect results
        stable_list.append(stable.item())
        stable_atoms_list.append(stable_atoms.item())
        atom_counts_list.append(atom_count.item())
        validity_list.append(float(is_valid(mol)))

    return (
        torch.tensor(validity_list),
        torch.tensor(stable_list),
        torch.tensor(stable_atoms_list),
        torch.tensor(atom_counts_list)
    )
