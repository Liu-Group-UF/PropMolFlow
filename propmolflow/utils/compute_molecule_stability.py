import argparse

import numpy as np
from rdkit import Chem
from rdkit import RDLogger

# Disable all RDKit warnings
RDLogger.DisableLog('rdApp.*')

from propmolflow.utils.geom_utils.molecule_stability import compute_molecules_stability


def compute_stability(sdf, n_subsets=1, aromatic="auto"):
    mols = [m for m in Chem.SDMolSupplier(sdf, sanitize=False, removeHs=False) if
            m is not None]

    if aromatic == "true":
        aromatic = True
    elif aromatic == "false":
        aromatic = False
    else:
        aromatic = any(b.GetIsAromatic() for m in mols for b in m.GetBonds())

    if n_subsets is None:
        validity, stable_mask, n_stable_atoms, n_atoms = compute_molecules_stability(mols,
                                                                                     aromatic=aromatic)
        print(f"Valid molecules: {int(validity.sum())} / {len(mols)} ({validity.mean():.4f})")
        print(
            f"Stable molecules: {int(stable_mask.sum())} / {len(mols)} ({stable_mask.mean():.4f})")
        print(f"Total atoms: {int(n_atoms.sum())}")
        print(
            f"Stable atoms: {int(n_stable_atoms.sum())} ({(n_stable_atoms.sum() / n_atoms.sum()):.4f})")
    else:
        mols_per_set = len(mols) // n_subsets
        valids, mols_stable, atoms_stable = [], [], []

        for i in range(n_subsets):
            chunk = mols[i * mols_per_set:(i + 1) * mols_per_set]
            validity, stable_mask, n_stable_atoms, n_atoms = compute_molecules_stability(chunk,
                                                                                         aromatic=aromatic)
            valids.append(validity.mean().item())
            mols_stable.append(stable_mask.mean().item())
            atoms_stable.append((n_stable_atoms.sum() / n_atoms.sum()).item())

        print(f"Avg validity: {np.mean(valids):.4f} ± {np.std(valids):.4f}")
        print(f"Avg molecule stability: {np.mean(mols_stable):.4f} ± {np.std(mols_stable):.4f}")
        print(f"Avg atom stability: {np.mean(atoms_stable):.4f} ± {np.std(atoms_stable):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=str, required=True, help="Path to input SDF file")
    parser.add_argument("--n_subsets", type=int, default=1,
                        help="Number of subsets to compute std over.")
    parser.add_argument("--aromatic", type=str, choices=["true", "false", "auto"], default="auto",
                        help="Whether to consider aromatic bonds: true, false, or auto")

    args = parser.parse_args()
    compute_stability(sdf=args.sdf, n_subsets=args.n_subsets, aromatic=args.aromatic)


if __name__ == "__main__":
    main()
