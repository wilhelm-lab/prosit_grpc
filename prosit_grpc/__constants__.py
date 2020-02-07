import numpy as np
from collections import OrderedDict as ODict

ALPHABET_UNMOD = {
    "A": 1,
    "(Cam)":2,
    "Cac":2,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
    "U": 21, # NEW
    "O": 22  # NEW
}

ALPHABET_MOD = {
    "M(ox)": 23,
    "M(O)":  23,
    "OxM":   23,
    
    'PhS':   24, 
    "S(ph)": 24,
    
    "PhT":   25,
    "T(ph)": 25,
    
    "PhY":   26,
    "Y(ph)": 26,
    
    "R(ci)": 27,
    "K(gl)": 28,
    "T(gl)": 29,
    "S(gl)": 30,
    "Q(gl)": 31,
    "R(me)": 32,
    "K(me)": 33,
    "T(ga)": 34,
    "S(ga)": 35,
    "K(ac)": 36,
}

ALPHABET = {**ALPHABET_UNMOD, **ALPHABET_MOD}
AMINO_ACIDS_INT = {integer: char for char, integer in ALPHABET.items()}
AMINO_ACIDS_INT[0] = ""
AMINO_ACIDS_ALPH = {char: integer for char, integer in ALPHABET.items()}
# What about masses for modifications?
AMINO_ACIDS_MASS = ODict(
    {
        "G": 57.021464,
        "R": 156.101111,
        "V": 99.068414,
        "P": 97.052764,
        "S": 87.032028,
        "L": 113.084064,
        "M": 131.040485,
        "Q": 128.058578,
        "N": 114.042927,
        "Y": 163.063329,
        "E": 129.042593,
        "C": 103.009185 + 57.0214637236 ,  # feynmann
        "(Cam)":  103.009185 + 57.0214637236,
        "Cac":  103.009185 + 57.0214637236,
        "F": 147.068414,
        "I": 113.084064,
        "A": 71.037114,
        "T": 101.047679,
        "W": 186.079313,
        "H": 137.058912,
        "D": 115.026943,
        "K": 128.094963,
        "U": 168.064, # NEW
        "O": 255.313, # NEW
        # Mods
        "M(ox)": 131.040485 + 15.99491,
        "M(O)":  131.040485 + 15.99491,
        "OxM":   131.040485 + 15.99491
        
    }
)

# Array containing masses --- at index one is mass for A, etc.
VEC_MZ = np.zeros(len(AMINO_ACIDS_MASS) + 1)
for i, a in AMINO_ACIDS_INT.items():
    if a in AMINO_ACIDS_MASS:
        VEC_MZ[i] = AMINO_ACIDS_MASS[a]

MASSES = ODict(
    {
        "PROTON":   1.007276467,
        "ELECTRON": 0.00054858,
        "H":        1.007825035,
        "C":        12.0,
        "O":        15.99491463,
        "N":        14.003074,
    }
)
MASSES["N_TERMINUS"] = MASSES["H"]
MASSES["C_TERMINUS"] = MASSES["O"] +MASSES["H"]
MASSES["CO"] = MASSES["C"] + MASSES["O"]
MASSES["CHO"] = MASSES["C"] + MASSES["H"] + MASSES["O"]
MASSES["NH2"] = MASSES["N"] + MASSES["H"] *2 
MASSES["H2O"] = MASSES["H"] * 2 + MASSES["O"]
MASSES["NH3"] = MASSES["N"] + MASSES["H"] * 3

SEQ_LEN = 30 # Sequence length for prosit
MAX_CHARGE = 6
BATCH_SIZE = 6000
