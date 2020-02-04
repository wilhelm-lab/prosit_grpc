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
        self.num_batches = len(sequences_list)
        self.concurrency = 1
        self._condition = threading.Condition()
        self._done = 0
        self._active = 0


        # prediction input instructions
        self.sequences_list = sequences_list
        self.charges_list = charges_list
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

        # if model_name == "intensity":
        #     self.normalize = True
        # else:
        #     self.normalize = False

        # Create channel and stub
        self.channel = grpc.insecure_channel(self.server)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)

        self.outputs = {}  # for PROSIT OUTPUTS, key is the dataset number
        self.callback_output = None
        self.raw_predictions = []
        self.filtered_invalid_predictions = []

        # print("[INFO] Initialized predicter with {} batches".format(self.num_batches))

    @staticmethod
    def sequence_numbers_to_alpha(x):
        """
        :param seq: list of letters to be converted to numbers
        """
        return [C.AMINO_ACIDS_INT[n] for n in x]

    @staticmethod
    def sequence_alpha_to_numbers(x):
        return [C.AMINO_ACIDS_ALPH[n] for n in x]

    @staticmethod
    def _create_request(model_name="intensity_prosit_publication", signature_name="serving_default"):
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

    def throttle(self, dataset_index):
        with self._condition:
            while self._active == self.concurrency:
                self._condition.wait()
            self._active += 1
            # print("Activated BATCH",dataset_index)

    def dec_active(self, dataset_index):
        with self._condition:
            self._active -= 1
            self._condition.notify()
            # print("Deactivated for BATCH", dataset_index)

    def inc_done(self, dataset_index):
        with self._condition:
            self._done += 1
            self._condition.notify()
            print("DONE with BATCH", dataset_index)

    # def set_collision_energies(ce):
    #     self.collision_energies_list = ce
    #
    # def set_charges(c):
    #     self.charges_list = c
    #
    # def set_sequences(s):
    #     self.sequences_list = s


    def prosit_callback(self, result_future, sequences, charges, collision_energies,
                        scan_nums=None, ids=None, labels=None, verbose=False):
        """
        Wrapper for callback function needed to add parameters
        returns the callback function
        """

        def _callback(result_future):
            """
            Calculates the statistics for the prediction result.
            :param result_future:
            """

            exception = result_future.exception()
                # Get output
            response_outputs = result_future.result().outputs

        return _callback


    def set_sequence_list_numeric(self):
        """
        Function that converts the sequences saved in self.sequence_list to a numerical encoding
        saves the encoded sequence in self.sequence_list_numeric
        """
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

    def reshape_callback_output(self):
        if self.model_name == "intensity_prosit_publication":
            outputs_tensor_proto = self.callback_output.outputs["out/Reshape:0"]
            shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)
            self.raw_predictions = np.array(outputs_tensor_proto.float_val).reshape(shape.as_list())

        elif self.model_name == "proteotypicity":
            outputs_tensor_proto = self.callback_output.outputs["pep_dense4/BiasAdd:0"]
            shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)
            outputs = np.array(outputs_tensor_proto.float_val).reshape(shape.as_list())
            self.raw_predictions = outputs.flatten()

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
        # self.final_predictions = self.normalized_predictions.copy()

        # print("Number of sequences", self.num_seq)
        for i in range(self.num_seq):
            charge = self.charges_list[i]
            preds = self.raw_predictions[i]

            # print(preds, charge)

            if charge == 1:
                # if charge == 1: all +2 & +3 invalid i.e. indexes only valid are indexes 0,3,6,9,...
                # invalid are x mod 3 != 0
                invalid_indexes = [(x * 3 + 1) for x in range((C.SEQ_LEN-1)*2)] + [(x * 3 + 2) for x in range((C.SEQ_LEN-1)*2)]
                preds[invalid_indexes] = -1
            elif charge == 2:
                # if charge == 1: all +2 & +3 invalid i.e. indexes only valid are indexes 0,1,3,4,6,7,9,10...
                # invalid are x mod 3 == 2
                invalid_indexes = [x * 3 + 2 for x in range(C.SEQ_LEN)]
                preds[invalid_indexes] = -1
            else:
                if charge > C.MAX_CHARGE:
                    print("[ERROR] in charge greater than 6")
                    return False
            # charge >= 3 --> all valid
                # print("No filtering by charges")
            self.filtered_invalid_predictions.append(preds)

            # 2. Filter by length of input sequence
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

    def get_predictions(self, verbose=False):
        """
        Tests PredictionService with concurrent requests.

        :return: The classification error rate.
        :raises IOError: An error occurred processing test data set.
        """

        self.set_sequence_list_numeric()

        if self.model_name == "intensity_prosit_publication":
            # set charges to one hot
            self.set_charges_list_one_hot()

            # set numpy arrays
            self.set_charges_array_float32()
            self.set_collision_energies_array_float32()
            self.set_sequences_array_int32()

        elif self.model_name == "proteotypicity":
            self.set_sequences_array_float32()

        # Create request
        request = self._create_request(model_name=self.model_name)

        # set tensors
        if self.model_name == "intensity_prosit_publication":
            # Parse inputs to request
            request.inputs['peptides_in:0'].CopyFrom(
                tf.contrib.util.make_tensor_proto(self.sequences_array, shape=[self.num_seq, C.SEQ_LEN]))
            request.inputs['collision_energy_in:0'].CopyFrom(
                tf.contrib.util.make_tensor_proto(self.collision_energies_array_float32, shape=[self.num_seq, 1]))
            request.inputs['precursor_charge_in:0'].CopyFrom(
                tf.contrib.util.make_tensor_proto(self.charges_array_float32, shape=[self.num_seq, 6]))

        elif self.model_name == "proteotypicity":
            request.inputs['peptides_in_1:0'].CopyFrom(
                tf.contrib.util.make_tensor_proto(self.sequences_array, shape=[self.num_seq, C.SEQ_LEN]))

        self.throttle(0)  #
        timeout = 5  # in seconds
        result_future = self.stub.Predict.future(request, timeout)  # asynchronous request

        # Callback function
        result_future.add_done_callback(
            self.prosit_callback(
                result_future=result_future,
                sequences=self.sequences_array,
                charges=self.charges_array_float32,
                collision_energies=self.collision_energies_array_float32,
                verbose=verbose
            )
        )

        # Wait until all request finish
        with self._condition:
            while self._done != 0:
                self._condition.wait()

        self.callback_output = result_future.result()
        self.reshape_callback_output()


        if self.model_name == "intensity_prosit_publication":
            self.filter_invalid()
            self.set_negative_to_zero()
            self.normalize_raw_predictions()

        if self.model_name == "proteotypicity":
            self.predictions = self.raw_predictions

        return self.predictions

