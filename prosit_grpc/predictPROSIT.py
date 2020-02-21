"""
predict-PROSIT.py is a gRPC client for obtaining PROSIT predictions and related information

__author__ = "Ludwig Lautenbacher"
__email__ = "Ludwig.Lautenbacher@tum.de"
"""

import numpy as np
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc
from typing import Iterable, Optional, Union

from . import __constants__ as C  # For constants
from . import __utils__ as U # Utility/Static functions

class PROSITpredictor:
    """PROSITpredictor is a class that contains all fetures to generate predictions with a Prosit server
    """
    def __init__(self,
                 server: str,
                 irt_model: str = None,
                 intensity_model: str = None,
                 proteotypicity_model: str = None,
                 sequences: list = None,
                 charges: list = None,
                 collision_energies: list = None,
                 path_to_ca_certificate: str = None,
                 path_to_certificate: str = None,
                 path_to_key_certificate: str = None):
        """PROSITpredictor is a class that contains all fetures to generate predictions with a Prosit server

        -- Non optional Parameters --
        :param server
        :param model_name

        -- optional parameters --
        :param path_to_ca_certificate
        :param path_to_certificate
        :param path_to_key_certificate

        :param sequences_list
        :param charges_list
        :param collision_energies_list
        """
        self.server = server
        self.create_channel(path_to_ca_certificate=path_to_ca_certificate,
                            path_to_key_certificate=path_to_key_certificate,
                            path_to_certificate=path_to_certificate)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)

        self.irt_model = irt_model
        self.intensity_model = intensity_model
        self.proteotypicity_model = proteotypicity_model

        self.input = PROSITinput(sequences=sequences,
                                 charges=charges,
                                 collision_energies=collision_energies)

        self.output = PROSIToutput()

    def create_channel(self, path_to_certificate, path_to_key_certificate, path_to_ca_certificate):
        try:
            # read certificates and create credentials
            with open(path_to_certificate, "rb") as f:
                cert = f.read()
            with open(path_to_key_certificate, "rb") as f:
                key = f.read()
            with open(path_to_ca_certificate, "rb") as f:
                ca_cert = f.read()
            creds = grpc.ssl_channel_credentials(ca_cert, key, cert)
            # create secure channel
            self.channel = grpc.secure_channel(self.server, creds)
        except:
            print("Establishing a secure channel was not possible")
            self.channel = grpc.insecure_channel(self.server)

class PROSITinput:
    def __init__(self, sequences=None, charges=None, collision_energies=None):
        l1 = len(sequences)
        l2 = len(charges)
        l3 = len(collision_energies)

        if l1 != l2 or l1 != l3 or l2 != l3:
            raise Exception(f"Please ensure that all list have the same length.\n"
                            f"The Sequences have:            {l1} entries.\n"
                            f"The charges have:              {l2} entries.\n"
                            f"The collision_energies have:   {l3} entries\n")

        self.sequences = PROSITsequences(sequences)
        self.charges = PROSITcharges(charges)
        self.collision_energies = PROSITcollisionenergies(collision_energies)

    def prepare_input(self):
        if self.sequences != None:
           self.sequences.prepare_sequences()
        if self.charges != None:
           self.charges.prepare_charges()
        if self.collision_energies != None:
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
        self.array = np.array(self.onehot, dtype=np.int32)

    def prepare_charges(self):
        if self.array == None:
            if self.onehot == None:
                if self.numeric == None:
                    raise ValueError("No charges known")
                self.numeric_to_onehot()
            self.onehot_to_array()

class PROSITsequences:
    def __init__(self, sequences):

        self.character = None
        self.numeric = None
        self.array = None
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
        for i,sequence in enumerate(self.character):
            num_seq = U.map_peptide_to_numbers(sequence)
            if len(num_seq) > C.SEQ_LEN:
                raise Exception(f"The Sequence {sequence}, has {i} Amino Acids."
                                f"The maximum number of amino acids allowed is {C.SEQ_LEN}")

            while len(num_seq) < C.SEQ_LEN:
                num_seq.append(0)
            self.numeric.append(num_seq)

    def numeric_to_array(self):
        self.array = np.array(self.numeric, dtype=np.int32)

    def calculate_lengths(self):
        """Calculates the length of all sequences saved in an instance of PROSITsequences

        :requires PROSITsequences.array

        :sets PROSITsequences.lengths
        """
        self.lengths = []
        for sequence in self.array:
            counter = 0
            for aa in sequence:
                if aa != 0:
                    counter += 1
            self.lengths.append(counter)

    def prepare_sequences(self):
        if self.array == None:
            if self.numeric == None:
                if self.character == None:
                    raise ValueError("No Sequences known")
                self.character_to_numeric()
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
        self.procentual = [i/100 for i in self.numeric]

    def procentual_to_array(self):
        self.array = np.array(self.procentual, dtype=np.float32)

    def prepare_collisionenergies(self):
        if self.array == None:
            if self.procentual == None:
                if self.numeric == None:
                    raise ValueError("No Collision Energies known")
                self.numeric_to_procentual()
            self.procentual_to_array()

# Output
class PROSIToutput:
    pass

class PROSITproteotypicity:
    pass

class PROSITirt:
    pass

class PROSITspectrum:
    pass

class PROSITintensity:
    pass

class PROSITfragmentmz:
    pass

class PROSITfragmentannotation:
    pass

