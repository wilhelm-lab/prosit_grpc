aa_unmod = ["A", "C(U:4)", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
aa_mod = {"M(U:35)": "M"}  # the Prosit WEbsite only knows M(ox)
# last value is not included
ce_range = [i for i in range(10, 51) if i % 10 == 0]
charge_range = [i for i in range(1, 7)]  # last value is not included

# sequences = []
# # generate sequences of each possible length
# for i in range(7,31):
#     sequence = random.choices(aa_unmod, k=i)
#     sequences.append("".join(sequence))

sequences = ["THDLGKW", "VLQKQFFYCTMEKWNGRT", "QMQCNWNVMQGAPSMTCEHRVEYSMEWIID"]

# modfiy sequences
for modification, unmod in aa_mod.items():
    sequence = list("QMQCNWNVMQGAPSMTCEHRVEYSMEWIID")
    new_sequence = []
    for amino_acid in sequence:
        if amino_acid == unmod:
            new_sequence.append(modification)
        else:
            new_sequence.append(amino_acid)
    sequences.append("".join(new_sequence))

# write csv file
with open("input_test.csv", "w") as file:
    file.write("modified_sequence,collision_energy,precursor_charge\n")
    for charge in charge_range:
        for ce in ce_range:
            for sequence in sequences:
                file.write(f"{sequence},{ce},{charge}\n")
