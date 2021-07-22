import re
import numpy as np

from fundamentals.constants import *

#####################
# GENERAL CONSTANTS #
#####################

SEQ_LEN = 30  # Sequence length for prosit
NUM_CHARGES_ONEHOT = 6
MAX_CHARGE = 6
BATCH_SIZE = 6000
VEC_LENGTH = 174

#######################################
# HELPERS FOR FRAGMENT MZ CALCULATION #
#######################################

# Array containing masses --- at index one is mass for A, etc.
VEC_MZ = np.zeros(max(ALPHABET.values()) + 1)
for a, i in AA_ALPHABET.items():
    VEC_MZ[i] = AA_MASSES[a]

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
