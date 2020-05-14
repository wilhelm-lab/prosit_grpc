"""
predict-PROSIT.py is a gRPC client for obtaining PROSIT predictions and related information

__author__ = "Ludwig Lautenbacher"
__email__ = "Ludwig.Lautenbacher@tum.de"
"""

import numpy as np
import grpc
from tqdm import tqdm
from tensorflow_serving.apis import prediction_service_pb2_grpc

from . import __constants__ as C  # For constants
from . import __utils__ as U  # Utility/Static functions
from .inputPROSIT import PROSITinput
from .outputPROSIT import PROSIToutput
from . import PredObject

class PROSITpredictor:
    """PROSITpredictor is a class that contains all fetures to generate predictions with a Prosit server
    """

    def __init__(self,
                 server: str = "proteomicsdb.org:8500",
                 path_to_ca_certificate: str = None,
                 path_to_certificate: str = None,
                 path_to_key_certificate: str = None,
                 keepalive_timeout_ms=10000):
        """PROSITpredictor is a class that contains all features to generate predictions with a Prosit server

        -- Non optional Parameters --
        :param server

        -- optional parameters --
        :param path_to_certificate
        :param path_to_key_certificate
        :param path_to_ca_certificate
        :param

        """
        self.server = server
        self.create_channel(path_to_ca_certificate=path_to_ca_certificate,
                            path_to_key_certificate=path_to_key_certificate,
                            path_to_certificate=path_to_certificate,
                            keepalive_timeout_ms=keepalive_timeout_ms)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(
            self.channel)
        self.channel = None

    def create_channel(self, path_to_certificate, path_to_key_certificate, path_to_ca_certificate, keepalive_timeout_ms):
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
            self.channel = grpc.secure_channel(self.server, creds, options=[
                                               ('grpc.keepalive_timeout_ms', keepalive_timeout_ms)])
        except:
            print("Establishing a secure channel was not possible")
            self.channel = grpc.insecure_channel(
                self.server, options=[('grpc.keepalive_timeout_ms', keepalive_timeout_ms)])

    def pred_object_factory(self, model):
        model_type = model.split("_")[2]
        if model_type == "intensity":
            return PredObject.Intensity(stub=self.stub,
                                        model_name=model,
                                        input=self.input)

        elif model_type == "irt":
            return PredObject.Irt(stub=self.stub,
                                  model_name=model,
                                  input=self.input)

        elif model_type == "proteotypicity":
            return PredObject.Proteotypicity(stub=self.stub,
                                             model_name=model,
                                             input=self.input)

        elif model_type == "charge":
            return PredObject.Charge(stub=self.stub,
                                     model_name=model,
                                     input=self.input)

        else:
            raise ValueError("The model type is not yet implemented in the Prosit GRPC cient.")

    def predict(self,
                models: list,
                sequences = None,
                charges = None,
                collision_energies = None,
                matrix_expansion_param: list = [],
                disable_progress_bar = False
                ):

        self.input = PROSITinput(sequences=sequences,
                                 charges=charges,
                                 collision_energies=collision_energies)

        self.input.prepare_input(disable_progress_bar)
        for paramset in matrix_expansion_param:
            self.input.expand_matrices(param=paramset)
        self.input.sequences.calculate_lengths()

        pred_objects = {}
        predictions = {}
        for model in models:
            print(f"Predicting for model: {model}")
            pred_objects[model] =  self.pred_object_factory(model=model)
            pred_objects[model].prepare_input()
            pred_objects[model].predict()
            pred_objects[model].prepare_output()
            predictions[model] = pred_objects[model].output

        # # initialize output
        # self.output = PROSIToutput(
        #     pred_intensity=predictions_intensity,
        #     pred_irt=predictions_irt,
        #     pred_proteotyp=predictions_proteotypicity,
        #     pred_charge=predictions_charge,
        #     sequences_array=self.input.sequences.array,
        #     charges_array=self.input.charges.array)
        #
        # # prepare output
        # self.output.prepare_output(charges_array=self.input.charges.array,
        #                            sequences_lengths=self.input.sequences.lengths)
        #
        # return self.output.assemble_dictionary()
        return predictions

    def predict_to_hdf5(self,
                        path_hdf5: str,
                        irt_model: str = None,
                        intensity_model: str = None,
                        sequences: list = None,
                        charges: list = None,
                        collision_energies: list = None,
                        disable_progress_bar=False):
        import h5py

        self.predict(irt_model=irt_model,
                     intensity_model=intensity_model,
                     sequences=sequences,
                     charges=charges,
                     collision_energies=collision_energies,
                     disable_progress_bar=disable_progress_bar)

        # weird formating of ce and irt is due to compatibility with converter tool
        hdf5_dict = {
            "sequence_integer": self.input.sequences.array,
            "precursor_charge_onehot": self.input.charges.array,
            "collision_energy_aligned_normed": np.array([np.array(el).astype(np.float32) for el in self.input.collision_energies.array]).astype(np.float32),
            'intensities_pred': self.output.spectrum.intensity.normalized,
            'masses_pred': self.output.spectrum.mz.masked,
            'iRT': np.array([np.array(el).astype(np.float32) for el in self.output.irt.normalized]).astype(np.float32)}

        hdf5_dict["collision_energy_aligned_normed"].shape = (
            len(hdf5_dict["collision_energy_aligned_normed"]), 1)
        hdf5_dict["iRT"].shape = (len(hdf5_dict["iRT"]), 1)

        with h5py.File(path_hdf5, "w") as data_file:
            for key, data in hdf5_dict.items():
                data_file.create_dataset(
                    key, data=data, dtype=data.dtype, compression="gzip")
