#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predictROSIT.py is a client obtaining PROSIT predictions

__author__ = "Daniela Andrade Salazar"
__email__ = "daniela.andrade@tum.de"
"""
import numpy as np
import threading
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from typing import Iterable, Optional, Union

from . import __constants__ as C  # For constants
from .__utils__ import normalize_intensities, indices_to_one_hot  # For ion mass computation

class PredictPROSIT:
    def __init__(self,
                 server: str,
                 model_name: str,
                 sequences_list: Union[np.ndarray, Iterable],
                 charges_list: Optional[Union[np.ndarray, Iterable]] = None,
                 collision_energies_list: Optional[Union[np.ndarray, Iterable]] = None,
                 ):
        """
        :param concurrency: Maximum number of concurrent inference requests
        :param server: PredictionService host:port, e.g. '131.159.152.7:8500'
        :param sequences_list: List of Lists.
                               Every List containing peptide sequences encoded as numbers e.g. [1,4,2,3,...]
        :param charges_list: List of Lists.
                             Every List containing 6 charges per sequence indicating the charges of
                             the y1,y2,y3 and b1,b2,b3 fragments of every sequence
        :param collision_energies_list: List of Lists.
                                        Every List containing one collision energy value for every sequence between 0.2 and 0.4
        """

        # server settings
        self.server = server
        self.model_name = model_name
        self.model_type = model_name.split("_")[0]

        self.concurrency = 1
        self._condition = threading.Condition()
        self._done = 0
        self._active = 0
        self.predictions_done = False


        # prediction input instructions
        self.sequences_list = sequences_list
        self.charges_list = charges_list

        if self.model_type == "intensity":
               self.collision_energies_list = [i / 100 for i in collision_energies_list]
        self.num_seq = len(sequences_list)

        # prepared/encoded input instructions
        # set with seperate function
        self.sequences_list_numeric = []
        self.charges_list_one_hot = []

        # prediction variables in array form for calling tensorflow
        self.sequences_array_int32 = None
        self.charges_array_float32 = None
        self.collision_energies_array_float32 = None

        # Create channel and stub
        self.channel = grpc.insecure_channel(self.server)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)

        self.outputs = {}  # for PROSIT OUTPUTS, key is the dataset number
        self.raw_predictions = []
        self.filtered_invalid_predictions = []

    @staticmethod
    def sequence_alpha_to_numbers(x):
        return [C.AMINO_ACIDS_ALPH[n] for n in x]

    @staticmethod
    def _create_request(model_name, signature_name="serving_default"):
        """
        :param model_name: Model name (taken from PROSIT)
        :param signature_name: Signature Name for the estimator (serving_default is by default set with custom tf estimator)
        :return created request
        """
        # Create Request
        request = predict_pb2.PredictRequest()
        # print("[INFO] Created Request")
        # Model and Signature Name
        request.model_spec.name = model_name
        request.model_spec.signature_name = signature_name
        # print("[INFO] Set model and signature name")
        return request

    @staticmethod
    def create_request_intensity(seq_array, ce_array, charges_array, batchsize, model_name):
        """
        seq_array
        ce_array
        charges_array
        batchsize       size of the created request
        model_name      specify the model that should be used to predict
        return:         request ready to be sent ther server
        """
        request = PredictPROSIT._create_request(model_name=model_name)
        request.inputs['peptides_in:0'].CopyFrom(
            tf.contrib.util.make_tensor_proto(seq_array, shape=[batchsize, C.SEQ_LEN]))
        request.inputs['collision_energy_in:0'].CopyFrom(
            tf.contrib.util.make_tensor_proto(ce_array, shape=[batchsize, 1]))
        request.inputs['precursor_charge_in:0'].CopyFrom(
            tf.contrib.util.make_tensor_proto(charges_array, shape=[batchsize, 6]))
        return request

    @staticmethod
    def create_request_proteotypicity(seq_array, batchsize, model_name):
        """
        seq array
        batchsize
        model_name  specify the model used for prediction
        """
        request = PredictPROSIT._create_request(model_name=model_name)
        request.inputs['peptides_in_1:0'].CopyFrom(
                tf.contrib.util.make_tensor_proto(seq_array, shape=[batchsize, C.SEQ_LEN]))
        return request

    @staticmethod
    def create_request_irt(seq_array, batchsize, model_name):
        """
        seq array
        batchsize
        model_name  specify the model used for prediction
        """
        request = PredictPROSIT._create_request(model_name=model_name)
        request.inputs['sequence_integer'].CopyFrom(
            tf.contrib.util.make_tensor_proto(seq_array, shape=[batchsize, C.SEQ_LEN]))
        return request

    def set_sequence_list_numeric(self):
        """
        Function that converts the sequences saved in self.sequence_list to a numerical encoding
        saves the encoded sequence in self.sequence_list_numeric
        """
        self.sequences_list_numeric = []
        for sequence in self.sequences_list:
            sequence_temp = self.sequence_alpha_to_numbers(sequence)
            while len(sequence_temp)<30:
                sequence_temp.append(0)
            self.sequences_list_numeric.append(sequence_temp)

    def set_charges_list_one_hot(self):
        """
        convert charges to one hot encoding
        One hot encoding of every charge value --> 6 possible charges for every sequence for PROSIT
        """
        self.charges_list_one_hot = [indices_to_one_hot(x, C.MAX_CHARGE) for x in self.charges_list]

    def set_charges_array_float32(self):
        self.charges_array_float32 = np.array(self.charges_list_one_hot).astype(np.float32)

    def set_collision_energies_array_float32(self):
        self.collision_energies_array_float32 = np.array(self.collision_energies_list).astype(np.float32)

    def set_sequences_array_int32(self):
        self.sequences_array = np.array(self.sequences_list_numeric).astype(np.int32)

    def set_sequences_array_float32(self):
        self.sequences_array = np.array(self.sequences_list_numeric).astype(np.float32)

    @staticmethod
    def reshape_predict_response_to_raw_predictions(predict_response, model_type):
        if model_type == "intensity":
            outputs_tensor_proto = predict_response.outputs["out/Reshape:0"]
            shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)
            return np.array(outputs_tensor_proto.float_val).reshape(shape.as_list())

        elif model_type == "proteotypicity":
            outputs_tensor_proto = predict_response.outputs["pep_dense4/BiasAdd:0"]
            shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)
            return np.array(outputs_tensor_proto.float_val).reshape(shape.as_list())

        elif model_type == "iRT":
            outputs_tensor_proto = predict_response.outputs["prediction/BiasAdd:0"]
            shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)
            return np.array(outputs_tensor_proto.float_val).reshape(shape.as_list())

    # normalization functions
    def filter_invalid(self):
        """
        assume reshaped output of prosit, shape sould be (num_seq, 174)
        set filtered output where not allowed positions are set to -1
        prosit output has the form:
        y1+1     y1+2 y1+3     b1+1     b1+2 b1+3     y2+1     y2+2 y2+3     b2+1     b2+2 b2+3
        if charge >= 3: all allowed
        if charge == 2: all +3 invalid
        if charge == 1: all +2 & +3 invalid
        """
        self.filtered_invalid_predictions = []
        for i in range(self.num_seq):
            charge = self.charges_list[i]
            preds = self.raw_predictions[i]
            if charge == 1:
                invalid_indexes = [(x * 3 + 1) for x in range((C.SEQ_LEN-1)*2)] + [(x * 3 + 2) for x in range((C.SEQ_LEN-1)*2)]
                preds[invalid_indexes] = -1
            elif charge == 2:
                invalid_indexes = [x * 3 + 2 for x in range((C.SEQ_LEN-1)*2)]
                preds[invalid_indexes] = -1
            else:
                if charge > C.MAX_CHARGE:
                    print("[ERROR] in charge greater than 6")
                    return False
            self.filtered_invalid_predictions.append(preds)
            len_seq = len(self.sequences_list[i])
            if len_seq < C.SEQ_LEN:
                self.filtered_invalid_predictions[i][(len_seq - 1) * 6:] = -1  # valid indexes are less than len_seq * 6

        return True


    def set_negative_to_zero(self):
        """
        assume reshaped and filtered output or prosit, shape should be (num_seq, 174)
        set output with positions <0 set to 0
        """
        for pred in self.filtered_invalid_predictions:
            pred[(pred != -1) & (pred < 0)] = 0

    def normalize_raw_predictions(self):
        """
        assume reshaped and filtered output of prosit, shape should be (num_seq, 174) normalized along first axis
        set normalized output between 0 and 1
        """
        self.predictions = normalize_intensities(self.filtered_invalid_predictions)
        self.predictions[self.predictions < 0] = -1

    def _predict_request(self, request):
        timeout = 5  # in seconds
        result_future = self.stub.Predict.future(request, timeout)  # asynchronous request
        return result_future.result()

    def predict(self):
        self.set_sequence_list_numeric()

        batch_start = 0

        if self.model_type == "intensity":
            # set charges to one hot
            self.set_charges_list_one_hot()

            # set numpy arrays
            self.set_charges_array_float32()
            self.set_collision_energies_array_float32()
            self.set_sequences_array_int32()

            requests = []
            while batch_start < self.num_seq:

                batch_end = batch_start+C.BATCH_SIZE-1
                batch_end = min(self.num_seq, batch_end)

                request = PredictPROSIT.create_request_intensity(
                    seq_array= self.sequences_array[batch_start:batch_end],
                    ce_array= self.collision_energies_array_float32[batch_start:batch_end],
                    charges_array= self.charges_array_float32[batch_start:batch_end],
                    model_name=self.model_name,
                    batchsize=(batch_end-batch_start))
                requests.append(request)

                batch_start = batch_end+1

        elif self.model_type == "iRT":
            self.set_sequences_array_int32()

            requests = []
            while batch_start < self.num_seq:
                batch_end = batch_start + C.BATCH_SIZE - 1
                batch_end = min(self.num_seq, batch_end)

                request = PredictPROSIT.create_request_irt(
                    seq_array=self.sequences_array[batch_start:batch_end],
                    model_name=self.model_name,
                    batchsize=(batch_end - batch_start))
                requests.append(request)
                batch_start = batch_end + 1

        elif self.model_type == "proteotypicity":
            self.set_sequences_array_float32()

            requests = []
            while batch_start < self.num_seq:
                batch_end = batch_start + C.BATCH_SIZE - 1
                batch_end = min(self.num_seq, batch_end)

                request = PredictPROSIT.create_request_proteotypicity(
                    seq_array=self.sequences_array[batch_start:batch_end],
                    model_name=self.model_name,
                    batchsize=(batch_end - batch_start))
                requests.append(request)
                batch_start = batch_end + 1



        self.raw_predictions = []
        for request in requests:
            self.raw_predictions.append(self.reshape_predict_response_to_raw_predictions(self._predict_request(request), model_type=self.model_type))

        self.raw_predictions = np.vstack(self.raw_predictions)

        if self.model_type == "intensity":
            self.filter_invalid()
            self.set_negative_to_zero()
            self.normalize_raw_predictions()

        if self.model_type == "proteotypicity" or self.model_type == "iRT":
            self.predictions = self.raw_predictions.flatten()

        self.predictions_done = True



    def get_raw_predictions(self):
        if self.predictions_done == False:
            self.predict()
        return self.raw_predictions

    def get_predictions(self):
        if self.predictions_done == False:
            self.predict()
        return self.predictions