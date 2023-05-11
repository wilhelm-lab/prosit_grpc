"""
Script predict-PROSIT.py is a gRPC client for obtaining PROSIT predictions and related information.

__author__ = "Ludwig Lautenbacher"
__email__ = "Ludwig.Lautenbacher@tum.de"
"""

import json
import re
from typing import Optional

import grpc
import h5py
from google.protobuf.json_format import MessageToJson
from tensorflow_serving.apis import get_model_status_pb2, model_service_pb2_grpc, prediction_service_pb2_grpc

from . import PredObject
from .inputPROSIT import PROSITinput


class PROSITpredictor:
    """PROSITpredictor is a class that contains all features to generate predictions with a Prosit server."""

    def __init__(
        self,
        path_to_ca_certificate: Optional[str],
        path_to_certificate: Optional[str],
        path_to_key_certificate: Optional[str],
        keepalive_timeout_ms: int = 10000,
        server: str = "proteomicsdb.org:8500",
    ):
        """
        The class PROSITpredictor contains all features to generate predictions with a Prosit server.

        -- Non optional Parameters --
        :param server: proteomicsdb server as string

        -- optional parameters --
        :param path_to_certificate: path to the certificate as string
        :param path_to_key_certificate: path to the key certificate as string
        :param path_to_ca_certificate: path to ca certificate as string
        :param keepalive_timeout_ms: keepalive timeout in ms as int
        """
        self.server = server
        self.create_channel(
            path_to_ca_certificate=path_to_ca_certificate,
            path_to_key_certificate=path_to_key_certificate,
            path_to_certificate=path_to_certificate,
            keepalive_timeout_ms=keepalive_timeout_ms,
        )
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)

    @staticmethod
    def check_model_availability(channel, model: str):
        """Checks model availability."""
        try:
            stub = model_service_pb2_grpc.ModelServiceStub(channel)
            request = get_model_status_pb2.GetModelStatusRequest()
            request.model_spec.name = model
            result = stub.GetModelStatus(request, 5)  # 5 secs timeout
            assert json.loads(MessageToJson(result))["model_version_status"][0]["state"] == "AVAILABLE"
            return True
        except grpc._channel._InactiveRpcError as e:
            if re.search(r"StatusCode\.NOT_FOUND", repr(e)) is not None:
                return False
            else:
                raise e

    def create_channel(
        self,
        path_to_certificate: str,
        path_to_key_certificate: str,
        path_to_ca_certificate: str,
        keepalive_timeout_ms: int,
    ):
        """
        Creates a channel.

        :param path_to_certificate: path to the certificate as string
        :param path_to_key_certificate: path to the key certificate as string
        :param path_to_ca_certificate: path to ca certificate as string
        :param keepalive_timeout_ms: keepalive timeout in ms as int
        """
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
            self.channel = grpc.secure_channel(
                self.server, creds, options=[("grpc.keepalive_timeout_ms", keepalive_timeout_ms)]
            )
        except Exception:
            # print("Establishing a secure channel was not possible")
            self.channel = grpc.insecure_channel(
                self.server, options=[("grpc.keepalive_timeout_ms", keepalive_timeout_ms)]
            )

    def pred_object_factory(self, model: str) -> PredObject.IntensityTMT:
        """
        Predicts object factory.

        :param model: model as string
        :raises ValueError: if the model type is not yet implemented in the Prosit GRPC client
        :return: object of a specific class, such as PredObject.IntensityTMT, PredObject.IrtTMT,
                 PredObject.Proteotypicity, PredObject.Charge or PredObject.Ir
        """
        model_type = model.split("_")[2]
        if "intensity" in model:
            if "TMT" in model:
                return PredObject.IntensityTMT(stub=self.stub, model_name=model, input=self.input)
            elif "PTM" in model:
                return PredObject.Intensity(stub=self.stub, model_name=model, input=self.input)
            elif 'tims' in model:
                return PredObject.IntensityTims(stub=self.stub, model_name=model, input=self.input)

            else:
                return PredObject.Intensity(stub=self.stub, model_name=model, input=self.input)
        elif "irt" in model:
            if "TMT" in model:
                return PredObject.IrtTMT(stub=self.stub, model_name=model, input=self.input)
            else:
                return PredObject.Irt(stub=self.stub, model_name=model, input=self.input)
        elif model_type == "proteotypicity":
            return PredObject.Proteotypicity(stub=self.stub, model_name=model, input=self.input)

        elif model_type == "charge":
            return PredObject.Charge(stub=self.stub, model_name=model, input=self.input)

        else:
            raise ValueError("The model type is not yet implemented in the Prosit GRPC cient.")

    def predict(
        self,
        models: list,
        sequences: list = None,
        charges: list = None,
        fragmentation: list = None,
        collision_energies: list = None,
        matrix_expansion_param: list = None,
        disable_progress_bar: bool = False,
    ) -> dict:
        """
        Predicts based on models.

        :param models: list of models
        :param sequences: list of sequences
        :param charges: list of charges
        :param fragmentation: list of fragmentations
        :param collision_energies: list of collision energies
        :param matrix_expansion_param: list of matrix expansion parameters
        :param disable_progress_bar: whether to disable progress bar
        :raises ValueError: if model is not available at the Prosit server
        :return: a dictionary of predictions
        """
        models_not_available = [not self.check_model_availability(self.channel, model=mo) for mo in models]
        models_not_available = [model for model, not_available in zip(models, models_not_available) if not_available]
        if len(models_not_available) > 0:
            raise ValueError(f"The models {models_not_available} are not available at the Prosit server")

        self.input = PROSITinput(
            sequences=sequences, charges=charges, collision_energies=collision_energies, fragmentation=fragmentation
        )

        self.input.prepare_input(disable_progress_bar)
        if matrix_expansion_param is not None:
            for paramset in matrix_expansion_param:
                self.input.expand_matrices(param=paramset)
            self.input.sequences.calculate_lengths()
        # print(self.input.sequences)

        predictions = {}
        for model in models:
            if not disable_progress_bar:
                print(f"Predicting for model: {model}")
            pred_object = self.pred_object_factory(model=model)
            pred_object.predict(disable_progress_bar)
            pred_object.prepare_output()
            predictions[model] = pred_object.output

        return predictions

    def predict_to_hdf5(
        self,
        path_hdf5: str,
        irt_model: str = None,
        intensity_model: str = None,
        proteotypicicty_model: str = None,
        fragmentation: list = None,
        sequences: list = None,
        charges: list = None,
        collision_energies: list = None,
        disable_progress_bar: bool = False,
    ):
        """
        Predict and save prediction as hdf5.

        :param path_hdf5: path to the hdf5 file
        :param irt_model: irt Prosit model
        :param intensity_model: intensity Prosit model
        :param proteotypicicty_model: proteotypicicty model
        :param fragmentation: list of fragmentation types
        :param sequences: list of sequences
        :param charges: list of charges
        :param collision_energies: list of collision energies
        :param disable_progress_bar: whether to disable progress bar
        """
        out_dict = self.predict(
            sequences=sequences,
            charges=charges,
            collision_energies=collision_energies,
            disable_progress_bar=disable_progress_bar,
            fragmentation=fragmentation,
            models=[irt_model, intensity_model, proteotypicicty_model],
        )

        hdf5_dict = {
            "sequence_integer": self.input.sequences.array,
            "precursor_charge_onehot": self.input.charges.array,
            "collision_energy_aligned_normed": self.input.collision_energies.array,
            "intensities_pred": out_dict[intensity_model]["intensity"],
            "masses_pred": out_dict[intensity_model]["fragmentmz"],
            "iRT": out_dict[irt_model],
            "proteotypicity": out_dict[proteotypicicty_model],
        }

        hdf5_dict["collision_energy_aligned_normed"].shape = (len(hdf5_dict["collision_energy_aligned_normed"]), 1)
        # hdf5_dict["iRT"].shape = (len(hdf5_dict["iRT"]),)

        with h5py.File(path_hdf5, "w") as data_file:
            for key, data in hdf5_dict.items():
                data_file.create_dataset(key, data=data, dtype=data.dtype, compression="gzip")
