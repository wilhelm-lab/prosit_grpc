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


class PROSITpredictor:
    """PROSITpredictor is a class that contains all fetures to generate predictions with a Prosit server
    """

    def __init__(self,
                 server: str = "proteomicsdb.org:8500",
                 path_to_ca_certificate: str = None,
                 path_to_certificate: str = None,
                 path_to_key_certificate: str = None,
		         keepalive_timeout_ms = 10000):
        """PROSITpredictor is a class that contains all fetures to generate predictions with a Prosit server

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
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
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
            self.channel = grpc.secure_channel(self.server, creds, options=[('grpc.keepalive_timeout_ms',keepalive_timeout_ms)])
        except:
            print("Establishing a secure channel was not possible")
            self.channel = grpc.insecure_channel(self.server, options=[('grpc.keepalive_timeout_ms',keepalive_timeout_ms)])

    @staticmethod
    def create_requests(model_name,
                        sequences_array,
                        charge_array=None,
                        ce_array=None,
                        ):
        batch_start = 0
        num_seq = len(sequences_array)
        requests = []
        while batch_start < num_seq:
            batch_end = batch_start + C.BATCH_SIZE
            batch_end = min(num_seq, batch_end)
            batchsize = batch_end - batch_start
            model_type = model_name.split("_")[0]
            seq_array_batch = sequences_array[batch_start:batch_end]

            if model_type == "intensity":
                ce_array_batch = ce_array[batch_start:batch_end]
                charges_array_batch = charge_array[batch_start:batch_end]
                request = U.create_request_intensity(seq_array_batch, ce_array_batch, charges_array_batch, batchsize, model_name)
            elif model_type == "iRT":
                request = U.create_request_irt(seq_array_batch, batchsize, model_name)
            elif model_type == "proteotypicity":
                request = U.create_request_proteotypicity(seq_array_batch, batchsize, model_name)

            requests.append(request)
            batch_start = batch_end
        return requests

    def send_requests(self, requests):
        timeout = 5  # in seconds

        predictions = []
        for request in tqdm(requests):
            model_type = request.model_spec.name.split("_")[0]
            response = self.stub.Predict.future(request, timeout).result()  # asynchronous request
            prediction = U.unpack_response(response, model_type)
            predictions.append(prediction)


        predictions = np.vstack(predictions)
        return predictions

    def predict(self,
                irt_model: str = None,
                intensity_model: str = None,
                proteotypicity_model: str = None,
                sequences: list = None,
                charges: list = None,
                collision_energies: list = None,
                matrix_expansion_param: list = []
                ):

        self.input = PROSITinput(sequences=sequences,
                                 charges=charges,
                                 collision_energies=collision_energies)

        self.input.prepare_input()
        for paramset in matrix_expansion_param:
            self.input.expand_matrices(param=paramset)
        self.input.sequences.calculate_lengths()

        # actual prediction
        predictions_irt = None
        predictions_intensity = None
        predictions_proteotypicity = None
        if irt_model is not None:
            requests = PROSITpredictor.create_requests(model_name=irt_model,
                                                       sequences_array=self.input.sequences.array_int32,
                                                       charge_array=self.input.charges.array,
                                                       ce_array=self.input.collision_energies.array
                                                       )
            predictions_irt = self.send_requests(requests)

        if intensity_model is not None:
            requests = PROSITpredictor.create_requests(model_name=intensity_model,
                                                       sequences_array=self.input.sequences.array_int32,
                                                       charge_array=self.input.charges.array,
                                                       ce_array=self.input.collision_energies.array
                                                       )
            predictions_intensity = np.array(self.send_requests(requests))

        if proteotypicity_model is not None:
            requests = PROSITpredictor.create_requests(model_name=proteotypicity_model,
                                                       sequences_array=self.input.sequences.array_float32,
                                                       charge_array=self.input.charges.array,
                                                       ce_array=self.input.collision_energies.array
                                                       )
            predictions_proteotypicity = self.send_requests(requests)

        # initialize output
        self.output = PROSIToutput(
            pred_intensity=predictions_intensity,
            pred_irt=predictions_irt,
            pred_proteotyp=predictions_proteotypicity,
            sequences_array_int32=self.input.sequences.array_int32,
            charges_array=self.input.charges.array)

        # prepare output
        self.output.prepare_output(charges_array=self.input.charges.array,
                                   sequences_lengths=self.input.sequences.lengths)

        return self.output.assemble_dictionary()

    def predict_to_hdf5(self,
                        path_hdf5: str,
                        irt_model: str = None,
                        intensity_model: str = None,
                        sequences: list = None,
                        charges: list = None,
                        collision_energies: list = None):
        import h5py

        self.predict(irt_model=irt_model,
                     intensity_model=intensity_model,
                     sequences=sequences,
                     charges=charges,
                     collision_energies=collision_energies)

        # weird formating of ce and irt is due to compatibility with converter tool
        hdf5_dict = {
            "sequence_integer": self.input.sequences.array_int32,
            "precursor_charge_onehot": self.input.charges.array,
            "collision_energy_aligned_normed": np.array([np.array(el).astype(np.float32) for el in self.input.collision_energies.array]).astype(np.float32),
            'intensities_pred': self.output.spectrum.intensity.normalized,
            'masses_pred': self.output.spectrum.mz.masked,
            'iRT': np.array([np.array(el).astype(np.float32) for el in self.output.irt.normalized]).astype(np.float32)}

        hdf5_dict["collision_energy_aligned_normed"].shape = (len(hdf5_dict["collision_energy_aligned_normed"]), 1)
        hdf5_dict["iRT"].shape = (len(hdf5_dict["iRT"]), 1)


        with h5py.File(path_hdf5, "w") as data_file:
            for key, data in hdf5_dict.items():
                data_file.create_dataset(key, data=data, dtype=data.dtype, compression="gzip")
