import numpy as np
from tqdm import tqdm
from . import __constants__ as C
from . import __utils__ as U  # Utility/Static functions


class PROSITinput:
    def __init__(self, sequences=None, charges=None, collision_energies=None, fragmentation=None):
        self.sequences = PROSITsequences(sequences)
        self.charges = PROSITcharges(charges)
        self.collision_energies = PROSITcollisionenergies(collision_energies)
        self.fragmentation = PROSITfragmentation(fragmentation)
        self.tmt = ''

    def prepare_input(self, flag_disable_progress_bar):
        if self.sequences is not None:
            self.sequences.prepare_sequences(flag_disable_progress_bar)
        if self.charges is not None:
            self.charges.prepare_charges()
        if self.collision_energies is not None:
            self.collision_energies.prepare_collisionenergies()
        self.tmt = self.sequences.tmt

    def expand_matrices(self, param):
        """
        Expects a list with dictionaries with 3 input parameters each
        {'AA_to_permutate': 'M', 'into': 'M(ox)', 'max_in_parallel': 2}
        The first one is the number that should be replaced
        The second one is the number that should be used to replace
        The third one is the number of changes that are performed at the same time
        """

        self.sequences.array, num_copies_created = U.generate_newMatrix_v2(npMatrix=self.sequences.array,
                                                                           iFromReplaceValue=C.ALPHABET[
                                                                               param['AA_to_permutate']],
                                                                           iToReplaceValue=C.ALPHABET[param['into']],
                                                                           numberAtTheSameTime=param['max_in_parallel'])

        charges_array = np.repeat(self.charges.array, num_copies_created, 0)
        collision_energies_array = np.repeat(self.collision_energies.array, num_copies_created, 0)

        self.charges.array = np.vstack([self.charges.array, charges_array])
        self.collision_energies.array = np.vstack([self.collision_energies.array, collision_energies_array])


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

        if charges is None:
            return None
        else:
            if type(charges) == np.ndarray:
                return "array"
            elif type(charges[0]) is list:
                return "one-hot"
            elif type(charges[0]) is int or type(charges[0]) is float:
                return "numeric"

    def numeric_to_onehot(self):
        self.onehot = [U.indices_to_one_hot(
            x, C.MAX_CHARGE) for x in self.numeric]

    def onehot_to_array(self):
        self.array = np.array(self.onehot, dtype=np.float32)

    def prepare_charges(self):
        if self.array is None:
            if self.onehot is None:
                if self.numeric is None:
                    return  # No charges known
                self.numeric_to_onehot()
            self.onehot_to_array()


class PROSITsequences:
    def __init__(self, sequences):

        self.character = None
        self.numeric = None
        self.array = None
        self.lengths = None
        self.tmt = ''

        seq_type = PROSITsequences.determine_type(sequences)
        if seq_type == "character":
            self.character = sequences
        elif seq_type == "numeric":
            self.numeric = sequences
        elif seq_type == "array":
            self.array = sequences

    @staticmethod
    def determine_type(sequences):
        try:
            if type(sequences) == np.ndarray:
                return "array"
            elif type(sequences[0]) is str:
                return "character"
            else:
                return "numeric"
        except:
            print(sequences)

    def character_to_array(self, flag_disable_progress_bar, filter=False):
        self.array = np.zeros((len(self.character), C.SEQ_LEN+2), dtype=np.uint8)
        if '2016' in self.character[0]:
            self.tmt = 'tmtpro'
            self.character = [x[13:] for x in self.character]
        elif '737' in self.character[0]:
            self.tmt = 'tmt'
            self.character = [x[12:] for x in self.character]
        elif '730' in self.character[0]:
            self.tmt = 'itraq8'
            self.character = [x[12:] for x in self.character]
        elif '214' in self.character[0]:
            self.tmt = 'itraq4'
            self.character = [x[12:] for x in self.character]

        generator_sequence_numeric = U.parse_modstrings(self.character, alphabet=C.ALPHABET, translate=True, filter=filter)
        enum_gen_seq_num = enumerate(generator_sequence_numeric)

        for i, sequence_numeric in tqdm(enum_gen_seq_num,
                                     disable=flag_disable_progress_bar,
                                     total=len(self.character)):
            
            if len(sequence_numeric) > C.SEQ_LEN+2:
                if filter:
                    pass # don't overwrite 0 in the array that is how we can differentiate
                else:
                    raise Exception(f"The Sequence {sequence_numeric}, has {len(sequence_numeric)} Amino Acids."
                                f"The maximum number of amino acids allowed is {C.SEQ_LEN}")
            else:
                self.array[i, 0:len(sequence_numeric)] = sequence_numeric

    def numeric_to_array(self):
        self.array = np.array(self.numeric, dtype=np.uint8)

    def calculate_lengths(self):
        """Calculates the length of all sequences saved in an instance of PROSITsequences

        :requires PROSITsequences.array

        :sets PROSITsequences.lengths
        """
        truth_array = np.in1d(
            self.array, [0], invert=True).reshape(self.array.shape)
        self.lengths = np.sum(truth_array, axis=1)

    def prepare_sequences(self, flag_disable_progress_bar=False, filter=False):
        if self.array is None:
            if self.numeric is None:
                if self.character is None:
                    raise ValueError("No Sequences known")
                self.character_to_array(flag_disable_progress_bar, filter=filter)
            else:
                self.numeric_to_array()
        self.calculate_lengths()


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

    @staticmethod
    def determine_type(collision_energies):

        if collision_energies is None:
            return None
        else:
            if type(collision_energies) == np.ndarray:
                return "array"
            elif collision_energies[0] < 1:
                return "procentual"
            else:
                return "numeric"

    def numeric_to_procentual(self):
        self.procentual = [i / 100 for i in self.numeric]

    def procentual_to_array(self):
        self.array = np.array(self.procentual, dtype=np.float32)
        self.array = self.array.reshape(len(self.array), 1)

    def prepare_collisionenergies(self):
        if self.array is None:
            if self.procentual is None:
                if self.numeric is None:
                    return  # No Collision Energies known
                self.numeric_to_procentual()
            self.procentual_to_array()


class PROSITfragmentation:
    def __init__(self, fragmentations):
       self.array = fragmentations
