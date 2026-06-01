"""
Molecular geometry comparison module for analyzing structural changes.
"""

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdMolTransforms


from propmolflow.utils.geom_utils.utils import generate_canonical_key


def compute_bond_angles_diff(pair):
    """
    Compute bond angle differences between initial and optimized molecular structures.
    
    This function analyzes all bond angles in the molecule by examining each atom
    with 2 or more neighbors. Bond angles are critical for understanding molecular
    geometry changes during optimization, as they reflect hybridization changes,
    strain relief, and conformational adjustments.
    
    Args:
        pair (tuple): (initial_mol, optimized_mol) RDKit molecule objects
        
    Returns:
        dict: Angle differences organized by geometric feature type
            Key: (atom1_type, bond1_type, central_atom_type, bond2_type, atom3_type)
            Value: ([angle_differences], count)
            
    Example Classifications:
        - (6, 1, 6, 1, 6): C-C-C single bond angles
        - (6, 1, 7, 1, 6): C-N-C single bond angles  
        - (8, 1, 6, 2, 8): O-C=O carbonyl angles
        
    Note:
        Angle differences are computed as the minimum of direct difference and
        360° complement to handle angle wraparound correctly.
    """
    init_mol, opt_mol = pair
    bond_angles = {}
    init_conf = init_mol.GetConformer()
    opt_conf = opt_mol.GetConformer()

    # Iterate through all atoms to find bond angles
    for atom in init_mol.GetAtoms():
        neighbors = atom.GetNeighbors()
        # Skip terminal atoms (degree < 2) - no meaningful bond angles
        if len(neighbors) < 2:
            continue

        # Examine all pairs of neighbors to form bond angles
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                # Bond angle: neighbor[i] - central_atom - neighbor[j]
                idx1, idx2, idx3 = neighbors[i].GetIdx(), atom.GetIdx(), neighbors[j].GetIdx()
                
                # Extract atomic types for chemical classification
                atom1_type, atom2_type, atom3_type = init_mol.GetAtomWithIdx(
                    idx1).GetAtomicNum(), init_mol.GetAtomWithIdx(
                    idx2).GetAtomicNum(), init_mol.GetAtomWithIdx(idx3).GetAtomicNum()
                    
                # Extract bond types for geometric classification
                bond_type_1 = int(init_mol.GetBondBetweenAtoms(idx1, idx2).GetBondType())
                bond_type_2 = int(init_mol.GetBondBetweenAtoms(idx2, idx3).GetBondType())

                # Compute bond angles in both conformations
                angle_init = rdMolTransforms.GetAngleDeg(init_conf, idx1, idx2, idx3)
                angle_opt = rdMolTransforms.GetAngleDeg(opt_conf, idx1, idx2, idx3)

                # Handle angle periodicity: choose minimum of direct diff and 360° complement
                # This correctly handles cases where angles cross 0°/360° boundary
                diff = min(np.abs(angle_init - angle_opt), 360 - np.abs(angle_init - angle_opt))

                # Generate canonical key for consistent aggregation
                key = generate_canonical_key(atom1_type, bond_type_1, atom2_type, bond_type_2,
                                             atom3_type)
                if key not in bond_angles:
                    bond_angles[key] = [[], 0]
                bond_angles[key][0].append(diff)
                bond_angles[key][1] += 1

    return bond_angles


def compute_bond_lengths_diff(pair):
    """
    Compute bond length differences between initial and optimized molecular structures.
    
    This function analyzes changes in all covalent bonds during molecular optimization.
    Bond length changes are fundamental indicators of structural quality, revealing
    strain relief, electronic effects, and optimization convergence.

    
    Args:
        pair (tuple): (initial_mol, optimized_mol) RDKit molecule objects
        
    Returns:
        dict: Bond length differences organized by bond type
            Key: (atom1_type, bond_type, atom2_type) - canonical ordering
            Value: ([length_differences], count)
            
    Example Classifications:
        - (6, 1, 6): C-C single bonds
        - (6, 2, 8): C=O double bonds
        - (6, 12, 6): C:C aromatic bonds (RDKit enum)
        - (7, 1, 1): N-H single bonds
        
    Note:
        Uses canonical key generation to ensure C-N and N-C bonds are
        treated as the same bond type for statistical aggregation.
    """
    init_mol, opt_mol = pair
    bond_lengths = {}

    init_conf = init_mol.GetConformer()
    opt_conf = opt_mol.GetConformer()
    
    # Analyze each bond in the molecule
    for bond in init_mol.GetBonds():
        idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        # Extract atomic types for chemical classification
        atom1_type, atom2_type = init_mol.GetAtomWithIdx(
            idx1).GetAtomicNum(), init_mol.GetAtomWithIdx(idx2).GetAtomicNum()
            
        # Extract bond type (1=single, 2=double, 3=triple, 12=aromatic)
        bond_type_numeric = int(bond.GetBondType())
        
        # Compute bond lengths in both conformations
        init_length = rdMolTransforms.GetBondLength(init_conf, idx1, idx2)
        opt_length = rdMolTransforms.GetBondLength(opt_conf, idx1, idx2)
        diff = np.abs(init_length - opt_length)

        # Generate canonical key to aggregate equivalent bond types
        key = generate_canonical_key(atom1_type, bond_type_numeric, atom2_type)
        if key not in bond_lengths:
            bond_lengths[key] = [[], 0]
        bond_lengths[key][0].append(diff)
        bond_lengths[key][1] += 1
    return bond_lengths


def compute_torsion_angles_diff(pair):
    """
    Compute torsion angle differences using SMARTS-based rotatable bond identification.
    
    This function identifies chemically meaningful torsion angles by using a
    SMARTS pattern to find rotatable bonds, then analyzes all torsion angles around these
    bonds. This approach focuses on conformationally relevant degrees of freedom.
    
    SMARTS PATTERN ANALYSIS:
    Pattern: "[!$(*#*)&!D1]~[!$(*#*)&!D1]"
    
    This pattern identifies bonds between two atoms where:
    1. Neither atom is involved in a triple bond (avoids rigid acetylene-like bonds)
    2. Neither atom is terminal (degree > 1, ensures meaningful rotation)
    3. Connected by any bond type
    
    Args:
        pair (tuple): (initial_mol, optimized_mol) RDKit molecule objects
        
    Returns:
        dict: Torsion angle differences organized by chemical environment
            Key: (atom1_type, bond1_type, atom2_type, bond2_type, atom3_type, bond3_type, atom4_type)
            Value: ([torsion_differences], count)
            
    Example Classifications:
        - (6, 1, 6, 1, 6, 1, 6): C-C-C-C alkyl chain torsion
        - (6, 1, 6, 12, 6, 12, 6): Aliphatic-aromatic torsion
        - (1, 1, 6, 1, 7, 1, 6): H-C-N-C torsion around C-N bond
        
    Note:
        Handles torsion angle periodicity correctly by choosing minimum of
        direct difference and 360° complement.
    """
    init_mol, opt_mol = pair
    
    # SMARTS PATTERN FOR ROTATABLE BONDS
    # "[!$(*#*)&!D1]~[!$(*#*)&!D1]"
    # Matches: (not triple bonded AND not terminal) ~ (not triple bonded AND not terminal)
    torsionSmarts = "[!$(*#*)&!D1]~[!$(*#*)&!D1]"
    torsion_query = Chem.MolFromSmarts(torsionSmarts)

    torsion_angles = {}

    init_conf = init_mol.GetConformer()
    opt_conf = opt_mol.GetConformer()

    # Find all rotatable bonds using SMARTS pattern
    torsion_matches = init_mol.GetSubstructMatches(torsion_query)

    # For each rotatable bond, find all possible torsion angles
    for match in torsion_matches:
        # Central bond atoms from SMARTS match
        idx2, idx3 = match[0], match[1]
        bond = init_mol.GetBondBetweenAtoms(idx2, idx3)

        # Find all atoms connected to the first central atom (idx2)
        for b1 in init_mol.GetAtomWithIdx(idx2).GetBonds():
            # Skip the central bond itself
            if b1.GetIdx() == bond.GetIdx():
                continue
            idx1 = b1.GetOtherAtomIdx(idx2)
            
            # Find all atoms connected to the second central atom (idx3)
            for b2 in init_mol.GetAtomWithIdx(idx3).GetBonds():
                # Skip the central bond and avoid double-counting
                if b2.GetIdx() == bond.GetIdx() or b2.GetIdx() == b1.GetIdx():
                    continue
                idx4 = b2.GetOtherAtomIdx(idx3)
                # Avoid degenerate 4-membered ring torsions
                if idx4 == idx1:
                    continue

                # Extract atomic and bond type information for classification
                atom1_type, atom2_type, atom3_type, atom4_type = init_mol.GetAtomWithIdx(
                    idx1).GetAtomicNum(), init_mol.GetAtomWithIdx(
                    idx2).GetAtomicNum(), init_mol.GetAtomWithIdx(
                    idx3).GetAtomicNum(), init_mol.GetAtomWithIdx(idx4).GetAtomicNum()
                bond_type_1 = int(b1.GetBondType())  # Bond idx1-idx2
                bond_type_2 = int(bond.GetBondType())  # Central bond idx2-idx3
                bond_type_3 = int(b2.GetBondType())  # Bond idx3-idx4

                # Compute torsion angles: idx1-idx2-idx3-idx4
                init_angle = rdMolTransforms.GetDihedralDeg(init_conf, idx1, idx2, idx3, idx4)
                opt_angle = rdMolTransforms.GetDihedralDeg(opt_conf, idx1, idx2, idx3, idx4)
                
                # Handle torsion angle periodicity (360° wraparound)
                diff = min(np.abs(init_angle - opt_angle), 360 - np.abs(init_angle - opt_angle))
                
                # Generate canonical key for consistent aggregation
                key = generate_canonical_key(atom1_type, bond_type_1, atom2_type, bond_type_2,
                                             atom3_type, bond_type_3, atom4_type)

                if key not in torsion_angles:
                    torsion_angles[key] = [[], 0]
                torsion_angles[key][0].append(diff)
                torsion_angles[key][1] += 1

    return torsion_angles
