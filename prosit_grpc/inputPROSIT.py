import numpy as np
from . import __constants__ as C  # For constants
from . import __utils__ as U  # Utility/Static functions


class PROSITinput:
    def __init__(self, sequences=None, charges=None, collision_energies=None):
        self.sequences = PROSITsequences(sequences)
        self.charges = PROSITcharges(charges)
        self.collision_energies = PROSITcollisionenergies(collision_energies)

    def prepare_input(self):
        if self.sequences is not None:
            self.sequences.prepare_sequences()
        if self.charges is not None:
            self.charges.prepare_charges()
        if self.collision_energies is not None:
            self.collision_energies.prepare_collisionenergies()


class PROSITcharges:
    def __init__(self, charges):
        self.numeric = None
        self.onehot = None
        self.array = None

        charge_type = PROSITcharges.determine_type(charges)
        if charge_type == "numeric":
            self.numeric = charges
        elif charge_type == "one-hot":
            self.onehot = charges
        elif charge_type == "array":
            self.array = charges

    @staticmethod
    def determine_type(charges):
        if type(charges) == np.ndarray:
            return "array"
        elif type(charges[1]) is list:
            return "one-hot"
        elif type(charges[1]) is int:
            return "numeric"

    def numeric_to_onehot(self):
        self.onehot = [U.indices_to_one_hot(x, C.MAX_CHARGE) for x in self.numeric]

    def onehot_to_array(self):
        self.array = np.array(self.onehot, dtype=np.float32)

    def prepare_charges(self):
        if self.array is None:
            if self.onehot is None:
                if self.numeric is None:
                    pass  # No charges known
                self.numeric_to_onehot()
            self.onehot_to_array()


class PROSITsequences:
    def __init__(self, sequences):

        self.character = None
        self.numeric = None
        self.array_int32 = None
        self.array_float32 = None
        self.lengths = None

        seq_type = PROSITsequences.determine_type(sequences)
        if seq_type == "character":
            self.character = sequences
        elif seq_type == "numeric":
            self.numeric = sequences
        elif seq_type == "array":
            self.array = sequences

    @staticmethod
    def determine_type(sequences):
        if type(sequences) == np.ndarray:
            return "array"
        elif type(sequences[1]) is str:
            return "character"
        else:
            return "numeric"

    def character_to_numeric(self):
        self.numeric = []
        for i, sequence in enumerate(self.character):
            num_seq = U.map_peptide_to_numbers(sequence)
            if len(num_seq) > C.SEQ_LEN:
                raise Exception(f"The Sequence {sequence}, has {i} Amino Acids."
                                f"The maximum number of amino acids allowed is {C.SEQ_LEN}")

            while len(num_seq) < C.SEQ_LEN:
                num_seq.append(0)
            self.numeric.append(num_seq)

    def numeric_to_array(self):
        self.array_int32 = np.array(self.numeric, dtype=np.int32)
        self.array_float32 = np.array(self.numeric, dtype=np.float32)

    def calculate_lengths(self):
        """Calculates the length of all sequences saved in an instance of PROSITsequences

        :requires PROSITsequences.array

        :sets PROSITsequences.lengths
        """
        self.lengths = []

        if self.array_int32 is not None:
            array = self.array_int32
        elif self.array_float32 is not None:
            array = self.array_float32

        for sequence in array:
            counter = 0
            for aa in sequence:
                if aa != 0:
                    counter += 1
            self.lengths.append(counter)

    def prepare_sequences(self):
        if self.array_float32 is None and self.array_int32 is None:
            if self.numeric is None:
                if self.character is None:
                    raise ValueError("No Sequences known")
                self.character_to_numeric()
            self.numeric_to_array()
            self.calculate_lengths()

        elif self.array_float32 is None:
            self.array_int32 = np.copy(self.array_float32).dtype = np.int32

        elif self.array_int32 is None:
            self.array_float32 = np.copy(self.array_int32).dtype = np.float32


class PROSITcollisionenergies:
    def __init__(self, collision_energies):
        self.numeric = None
        self.procentual = None
        self.array = None

        ce_type = PROSITcollisionenergies.determine_type(collision_energies)
        if ce_type == "numeric":
            self.numeric = collision_energies
        elif ce_type == "procentual":
            self.procentual = collision_energies
        elif ce_type == "array":
            self.array = collision_energies
        else:
            raise ValueError("The ce_type is not known")

    @staticmethod
    def determine_type(collision_energies):
        if type(collision_energies) == np.ndarray:
            return "array"
        elif collision_energies[1] < 1:
            return "procentual"
        else:
            return "numeric"

    def numeric_to_procentual(self):
        self.procentual = [i / 100 for i in self.numeric]

    def procentual_to_array(self):
        self.array = np.array(self.procentual, dtype=np.float32)

    def prepare_collisionenergies(self):
        if self.array is None:
            if self.procentual is None:
                if self.numeric is None:
                    pass  # No Collision Energies known
                self.numeric_to_procentual()
            self.procentual_to_array()
