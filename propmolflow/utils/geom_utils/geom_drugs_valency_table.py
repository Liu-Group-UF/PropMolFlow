"""
Comprehensive Valency Tables for Molecular Stability Assessment

This module provides multiple valency lookup tables designed for different approaches
to molecular stability validation. The tables address the fundamental challenge of
validating atomic valencies in aromatic systems where traditional bond order summation
is chemically inaccurate.


Three approaches are provided:

1. geom_drugs_h_valencies: Simple total valency counting (includes hydrogens)
   - Used for non-aromatic systems or when aromatic bonds are treated as single bonds
   - Counts total valence including implicit hydrogens

2. geom_drugs_h_legacy_valencies: Legacy compatibility table
   - Includes some chemically questionable entries from prior work
   - Retained for benchmarking against older models (MiDi, EQGAT, etc.)

3. geom_drugs_h_tuple_valencies: Aromatic-aware tuple-based validation
   - Separates aromatic and non-aromatic bond contributions
   - Format: (n_aromatic_bonds, valence_from_non_aromatic_bonds)
   - Enables chemically accurate validation of aromatic systems

"""

geom_drugs_h_valencies = {
    "Br": {0: [1], 1: [2]},
    "C": {0: [4], -1: [3], 1: [3]},
    "N": {0: [3], 1: [4], -1: [2], -2: [1]},
    "H": {0: [1]},
    "S": {0: [2, 6, 3], 1: [3], 2: [4], 3: [5, 2], -1: [1]},
    "O": {0: [2], -1: [1], 1: [3]},
    "F": {0: [1]},
    "Cl": {0: [1], 1: [2]},
    "P": {0: [5, 3], 1: [4]},
    "I": {0: [1], 1: [2], 2: [3]},
    "Si": {0: [4], 1: [5]},
    "B": {-1: [4], 0: [3]},
    "Bi": {2: [5], 0: [3]
           }
}

"""
This dictionary includes atomic valency assumptions used in prior works (e.g., MiDi, EQGAT, SemlaFlow).
It may include erroneous or outdated values due to improper handling of aromatic bond orders.
"""

geom_drugs_h_legacy_valencies = {
    "H": {0: 1, 1: 0, -1: 0},
    "C": {0: [3, 4], 1: 3, -1: 3},
    "N": {0: [2, 3], 1: [2, 3, 4], -1: 2},
    "O": {0: 2, 1: 3, -1: 1},
    "F": {0: 1, -1: 0},
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": {0: [3, 5], 1: 4},
    "S": {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
    "Cl": 1,
    "As": 3,
    "Br": {0: 1, 1: 2},
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}

"""
Aromatic-Aware Tuple-Based Valency Table

This table implements one possible approach to valency validation by
separating aromatic and non-aromatic bond contributions. This design choice addresses
fundamental limitations in traditional bond order summation for aromatic systems.

Each allowed valency state is represented as (n_aromatic_bonds, valence_non_aromatic):
- n_aromatic_bonds: Integer count of bonds participating in aromatic systems
- valence_non_aromatic: Sum of bond orders for non-aromatic bonds (including hydrogens)

This table covers all element/charge combinations observed in the filtered GEOM-Drugs
dataset.
"""

geom_drugs_h_tuple_valencies = {
    "Br": {
        0: [(0, 1)],
        1: [(0, 2)],
    },
    "C": {
        0: [(0, 4), (2, 2), (2, 1), (3, 0)],
        -1: [(0, 3), (2, 1), (3, 0)],
        1: [(0, 3), (2, 1), (3, 0)],
    },
    "N": {
        0: [(0, 3), (2, 0), (2, 1), (3, 0)],
        1: [(0, 4), (2, 0), (2, 1), (2, 2), (3, 0)],
        -1: [(0, 2), (2, 0)],
        -2: [(0, 1)],
    },
    "H": {
        0: [(0, 1)],
    },
    "S": {
        0: [(0, 2), (0, 3), (0, 6), (2, 0)],
        1: [(0, 3), (2, 0), (2, 1), (3, 0)],
        2: [(0, 4), (2, 1), (2, 2)],
        3: [(0, 2), (0, 5)],
        -1: [(0, 1)],
    },
    "O": {
        0: [(0, 2), (2, 0)],
        -1: [(0, 1)],
        1: [(0, 3)],
    },
    "F": {
        0: [(0, 1)],
    },
    "Cl": {
        0: [(0, 1)],
        1: [(0, 2)],
    },
    "P": {
        0: [(0, 3), (0, 5)],
        1: [(0, 4)],
    },
    "I": {
        0: [(0, 1)],
        1: [(0, 2)],
        2: [(0, 3)],
    },
    "Si": {
        0: [(0, 4)],
        1: [(0, 5)],
    },
    "B": {
        -1: [(0, 4)],
        0: [(0, 3)],
    },
    "Bi": {
        0: [(0, 3)],
        2: [(0, 5)],
    }
}
