# from spectrum_fundamentals.constants import *

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
