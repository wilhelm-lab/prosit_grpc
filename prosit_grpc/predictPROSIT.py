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
                 model_name: str,
                 sequences_list: Optional[Union[np.ndarray, Iterable]] = None,
                 charges_list: Optional[Union[np.ndarray, Iterable]] = None,
                 collision_energies_list: Optional[Union[np.ndarray, Iterable]] = None,
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
        self.model_name = model_name
        self.model_type = model_name.split("_")[0]

        self.input = PROSITinput()

class PROSITinput:
    def __init__(self, sequences = None, charges = None, collision_energies = None):
        self.sequences = PROSITsequences(sequences)
        self.charges = PROSITcharges(charges)
        self.collision_energies = PROSITcollisionenergies(collision_energies)

class PROSITcharges:
    def __init__(self, charges):
        self.numeric
        self.onehot
        self.array

    @staticmethod
    def determine_type():
        pass

    def numeric_to_onehot(self):
        pass

    def onehot_to_array(self):
        pass

class PROSITsequences:
    def __init__(self, sequences):
        self.character
        self.numeric
        self.array
        self.lengths

    @staticmethod
    def determine_type():
        pass

    def numeric_to_array(self):
        pass

    def character_to_numeric(self):
        pass

    def calculate_lengths(self):
        pass

class PROSITcollisionenergies:
    def __init__(self, collision_energies):

        type = PROSITcollisionenergies.determine_type(collision_energies)

        self.numeric
        self.procentual
        self.array

    @staticmethod
    def determine_type(collision_energies):
        if collision_energies[1] > 1:


        elif type(collision_energies) == "numeric":
            return "numeric"

        elif type(collision_energies) == "array":
            return "array"

    def numeric_to_procentual(self):
        pass

    def procentual_to_array(self):
        pass

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

