"""
predict-PROSIT.py is a gRPC client for obtaining PROSIT predictions and related information

__author__ = "Ludwig Lautenbacher"
__email__ = "Ludwig.Lautenbacher@tum.de"
"""

import numpy as np
import grpc
import tensorflow as tf
from tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2

from . import __constants__ as C  # For constants
from . import __utils__ as U  # Utility/Static functions
from .inputPROSIT import PROSITinput
from .outputPROSIT import PROSIToutput

class PROSITpredictor:
    """PROSITpredictor is a class that contains all fetures to generate predictions with a Prosit server
    """
    def __init__(self,
                 server: str,
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

            request = U.create_request_general(
                seq_array=sequences_array[batch_start:batch_end],
                ce_array=ce_array[batch_start:batch_end],
                charges_array=charge_array[batch_start:batch_end],
                model_name=model_name,
                batchsize=(batch_end - batch_start))
            requests.append(request)
            batch_start = batch_end

        return requests


    def send_requests(self, requests):
        timeout = 5  # in seconds

        predictions = np.array()
        while len(requests) > 0:
            request = requests.pop()
            model_type = request.model_spec.name.split("_")[0]
            response = self.stub.Predict.future(request, timeout).result()  # asynchronous request
            prediction = U.unpack_response(response, model_type)
            np.append(predictions, prediction, axis=0)
        return predictions


    def predict(self,
                irt_model: str = None,
                intensity_model: str = None,
                proteotypicity_model: str = None,
                sequences: list = None,
                charges: list = None,
                collision_energies: list = None,
                ):

        input = PROSITinput(sequences=sequences,
                                 charges=charges,
                                 collision_energies=collision_energies)
        input.prepare_input()

        # actual prediction
        predictions_irt = None
        predictions_intensity = None
        predictions_proteotypicity = None
        if irt_model is not None:
            requests = PROSITpredictor.create_requests(model_name=irt_model,
                                                       sequences_array=input.sequences.array,
                                                       charge_array=input.charges.array,
                                                       ce_array=input.collision_energies.array
                                                       )
            predictions_irt = self.send_requests(requests)

        if intensity_model is not None:
            requests = PROSITpredictor.create_requests(model_name=intensity_model,
                                                       sequences_array=input.sequences.array,
                                                       charge_array=input.charges.array,
                                                       ce_array=input.collision_energies.array
                                                       )
            predictions_intensity = self.send_requests(requests)

        if proteotypicity_model is not None:
            requests = PROSITpredictor.create_requests(model_name=proteotypicity_model,
                                                       sequences_array=input.sequences.array,
                                                       charge_array=input.charges.array,
                                                       ce_array=input.collision_energies.array
                                                       )
            predictions_proteotypicity = self.send_requests(requests)


        output = PROSIToutput()
        # output.spectrum.intensity.raw = predictions_intensity
        # output.spectrum.mz.raw = np.array([U.compute_ion_masses()])
        # output.spectrum.annotation.raw =



        # prepare output
        # return output
        # return dictionary(key is model_name)
