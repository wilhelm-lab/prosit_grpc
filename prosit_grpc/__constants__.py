from pyteomics.mass import calculate_mass
from pyteomics.mass.unimod import Unimod
import re
import numpy as np

def get_modstring_mass(modstring, aa_mass, unimod_connection=None):
    """
    Uses pyteomics to calculate the mass of a modstring
    """
    mod = [int(x[0]) for x in re.findall('U:(\d+?)(,|\))', modstring)]
    mod_mass = [unimod_connection.get(m).monoisotopic_mass for m in mod]
    return(aa_mass + sum(mod_mass))



#############
# ALPHABETS #
#############

ALPHABET_UNMOD = {
    "A": 1,
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
    "U": 22,
    "O": 23,
    "C": 24
}

ALPHABET_MOD = {
    "M(U:35)": 21,
    "C(U:4)": 2
}

# ALPHABET contains all amino acid and ptm abbreviations and
ALPHABET = {**ALPHABET_UNMOD, **ALPHABET_MOD}
AMINO_ACIDS_INT = {integer: char for char, integer in ALPHABET.items()}
AMINO_ACIDS_INT[0] = ""

ALPHABET_MASS = {}
for a in ALPHABET_UNMOD:
    m = calculate_mass(a) - calculate_mass("H2O")  # peptide bond loses water
    ALPHABET_MASS[a] = m

for key in ALPHABET_MOD.keys():
    ALPHABET_MASS[key] = get_modstring_mass(key,
                                            ALPHABET_MASS[key[0]],
                                            unimod_connection=Unimod())

#######################################
# HELPERS FOR FRAGMENT MZ CALCULATION #
#######################################

# Array containing masses --- at index one is mass for A, etc.
VEC_MZ = np.zeros(len(ALPHABET_MASS) + 1)
for i, a in AMINO_ACIDS_INT.items():
    if a in ALPHABET_MASS:
        VEC_MZ[i] = ALPHABET_MASS[a]

MASSES = {
        "PROTON": 1.007276467,
        "ELECTRON": 0.00054858,
        "H": 1.007825035,
        "C": 12.0,
        "O": 15.99491463,
        "N": 14.003074,
}

MASSES["N_TERMINUS"] = MASSES["H"]
MASSES["C_TERMINUS"] = MASSES["O"] + MASSES["H"]

#####################
# GENERAL CONSTANTS #
#####################

SEQ_LEN = 30  # Sequence length for prosit
NUM_CHARGES_ONEHOT = 6
MAX_CHARGE = 6
BATCH_SIZE = 6000
VEC_LENGTH = 174

############################
# GENERATION OF ANNOTATION #
############################

IONS = ['y', 'b']  # limited to single character unicode string when array is created
CHARGES = [1, 2, 3]  # limited to uint8 (0-255) when array is created
POSITIONS = [x for x in range(1, 30)]  # fragment numbers 1-29 -- limited to uint8 (0-255) when array is created

ANNOTATION_FRAGMENT_TYPE = []
ANNOTATION_FRAGMENT_CHARGE = []
ANNOTATION_FRAGMENT_NUMBER = []
for pos in POSITIONS:
    for ion in IONS:
        for charge in CHARGES:
            ANNOTATION_FRAGMENT_TYPE.append(ion)
            ANNOTATION_FRAGMENT_CHARGE.append(charge)
            ANNOTATION_FRAGMENT_NUMBER.append(pos)

ANNOTATION = [ANNOTATION_FRAGMENT_TYPE,
              ANNOTATION_FRAGMENT_CHARGE, ANNOTATION_FRAGMENT_NUMBER]

##################################################################
# FOR WHEN MANUAL SPECIFICATION OF MODIFICATIONS BECOMES TEDIOUS #
##################################################################
#
# POSSIBLE_MOD = {
#     "A": ["U:1", "U:2", "U:3"],
#     "C": ["U:4", "U:5", "U:6"]
# }
#
# # performed for each aa
# def generate_modstrings(aa, possible_mod):
#     mod_combinations = []
#     for i in range(len(possible_mod)):
#         mod_combinations.extend([x for x in itertools.combinations(possible_mod, i+1)])
#     return([aa + "(" + ",".join(x) + ")" for x in mod_combinations])
#
# # tmp = [item for sublist in [generate_modstrings(aa, mods) for aa, mods in POSSIBLE_MOD.items()] for item in sublist]
