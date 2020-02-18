#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predictROSIT.py is a client obtaining PROSIT predictions

__author__ = "Daniela Andrade Salazar"
__email__ = "daniela.andrade@tum.de"
"""
import numpy as np
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc
from typing import Iterable, Optional, Union

from . import __constants__ as C  # For constants
from . import __utils__ as U # Utility/Static functions


class PredictPROSIT:
    def __init__(self,
                 server: str,
                 model_name: str,
                 sequences_list: Optional[Union[np.ndarray, Iterable]] = None,
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

        self.set_model_name(model_name=model_name)
        self.predictions_done = False

        # sequences
        self.sequences_list = sequences_list
        self.sequences_list_numeric = None
        self.sequences_array = None

        # charges
        self.charges_list = charges_list
        self.charges_list_one_hot = None
        self.charges_array_float32 = None

        # ce
        self.collision_energies_list = collision_energies_list
        self.collision_energy_normed = None
        self.collision_energies_array_float32 = None

        # Create channel and stub
        self.channel = grpc.insecure_channel(self.server)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)


        # Output
        self.raw_predictions = []
        self.filtered_invalid_predictions = []

    def set_sequence_list_numeric(self, numeric_sequence_list = None):
        """
        Function that converts the sequences saved in self.sequence_list to a numerical encoding
        saves the encoded sequence in self.sequence_list_numeric
        """
        if numeric_sequence_list == None:
            self.sequences_list_numeric = []
            for sequence in self.sequences_list:
                numeric_sequence = list(U.map_peptide_to_numbers(sequence))
                while len(numeric_sequence)<30:
                    numeric_sequence.append(0)

                self.sequences_list_numeric.append(numeric_sequence)
        else:
            self.sequences_list_numeric = numeric_sequence_list

    def set_charges_list_one_hot(self, one_hot_charges_list = None):
        """
        convert charges to one hot encoding
        One hot encoding of every charge value --> 6 possible charges for every sequence for PROSIT
        """

        if one_hot_charges_list == None:
            self.charges_list_one_hot = [U.indices_to_one_hot(x, C.MAX_CHARGE) for x in self.charges_list]
        else:
            self.charges_list_one_hot = one_hot_charges_list

    def set_collision_energy_normed(self, collision_energy_normed = None):
        if collision_energy_normed == None:
            self.collision_energy_normed = [i / 100 for i in self.collision_energies_list]
        else:
            self.collision_energy_normed = collision_energy_normed

    def set_model_name(self, model_name):
        self.model_name = model_name
        self.model_type = model_name.split("_")[0]

        self.predictions_done = False

    def set_charges_array_float32(self):
        self.charges_array_float32 = np.array(self.charges_list_one_hot).astype(np.float32)

    def set_collision_energies_array_float32(self):
        self.collision_energies_array_float32 = np.array(self.collision_energy_normed).astype(np.float32)

    def set_sequences_array_int32(self):
        self.sequences_array = np.array(self.sequences_list_numeric).astype(np.int32)

    def set_sequences_array_float32(self):
        self.sequences_array = np.array(self.sequences_list_numeric).astype(np.float32)

    def set_fragment_masses(self):
        self.fragment_masses = []
        for i in range(self.num_seq):
            self.fragment_masses.append(U.compute_ion_masses(seq_int= self.sequences_list_numeric[i],
                                                           charge_onehot=self.charges_list_one_hot[i]))

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
            charge_one_hot = self.charges_list_one_hot[i]
            preds = self.raw_predictions[i]

            # filter invalid fragment charges
            if np.array_equal(charge_one_hot, [1, 0, 0, 0, 0, 0]):
                invalid_indexes = [(x * 3 + 1) for x in range((C.SEQ_LEN-1)*2)] + [(x * 3 + 2) for x in range((C.SEQ_LEN-1)*2)]
                preds[invalid_indexes] = -1
            elif np.array_equal(charge_one_hot, [0, 1, 0, 0, 0, 0]):
                invalid_indexes = [x * 3 + 2 for x in range((C.SEQ_LEN-1)*2)]
                preds[invalid_indexes] = -1

            self.filtered_invalid_predictions.append(preds)

            # filter invalid fragment numbers
            len_seq = 0
            for amino_acid in self.sequences_list_numeric[i]:
                if amino_acid != 0:
                    len_seq += 1

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
        self.predictions = U.normalize_intensities(self.filtered_invalid_predictions)
        self.predictions[self.predictions < 0] = -1

    def _predict_request(self, request):
        timeout = 5  # in seconds
        result_future = self.stub.Predict.future(request, timeout)  # asynchronous request
        return result_future.result()

    def predict(self):
        batch_start = 0

        if self.sequences_list_numeric == None:
            self.set_sequence_list_numeric()

        self.num_seq = len(self.sequences_list_numeric)

        if self.model_type == "intensity":
            if self.charges_list_one_hot == None:
                # set charges to one hot
                self.set_charges_list_one_hot()

            if self.collision_energy_normed == None:
                # norm ce
                self.set_collision_energy_normed()

            # set fragment masses
            self.set_fragment_masses()

            # set numpy arrays
            self.set_charges_array_float32()
            self.set_collision_energies_array_float32()
            self.set_sequences_array_int32()

            requests = []
            while batch_start < self.num_seq:

                batch_end = batch_start+C.BATCH_SIZE
                batch_end = min(self.num_seq, batch_end)

                request = U.create_request_intensity(
                    seq_array= self.sequences_array[batch_start:batch_end],
                    ce_array= self.collision_energies_array_float32[batch_start:batch_end],
                    charges_array= self.charges_array_float32[batch_start:batch_end],
                    model_name=self.model_name,
                    batchsize=(batch_end-batch_start))
                requests.append(request)

                batch_start = batch_end

        elif self.model_type == "iRT":
            self.set_sequences_array_int32()

            requests = []
            while batch_start < self.num_seq:
                batch_end = batch_start + C.BATCH_SIZE
                batch_end = min(self.num_seq, batch_end)

                request = U.create_request_irt(
                    seq_array=self.sequences_array[batch_start:batch_end],
                    model_name=self.model_name,
                    batchsize=(batch_end - batch_start))
                requests.append(request)
                batch_start = batch_end

        elif self.model_type == "proteotypicity":
            self.set_sequences_array_float32()

            requests = []
            while batch_start < self.num_seq:
                batch_end = batch_start + C.BATCH_SIZE
                batch_end = min(self.num_seq, batch_end)

                request = U.create_request_proteotypicity(
                    seq_array=self.sequences_array[batch_start:batch_end],
                    model_name=self.model_name,
                    batchsize=(batch_end - batch_start))
                requests.append(request)
                batch_start = batch_end

        self.raw_predictions = []
        for request in requests:
            self.raw_predictions.append(U.reshape_predict_response_to_raw_predictions(self._predict_request(request), model_type=self.model_type))

        self.raw_predictions = np.vstack(self.raw_predictions)

        if self.model_type == "intensity":
            self.filter_invalid()
            self.set_negative_to_zero()
            self.normalize_raw_predictions()

        elif self.model_type == "proteotypicity":
            self.predictions = self.raw_predictions.flatten()

        elif self.model_type == "iRT":
            self.raw_predictions = [i[0] for i in self.raw_predictions]
            self.predictions = [i * 43.39373 + 56.35363441 for i in self.raw_predictions]

        self.predictions_done = True

    def get_raw_predictions(self):
        if self.predictions_done == False:
            self.predict()
        return self.raw_predictions

    def get_predictions(self):
        if self.predictions_done == False:
            self.predict()
        # return self.predictions
        return [[int for int in el if int != -1] for el in self.predictions]

    def get_fragment_masses(self):
        if self.predictions_done == False:
            self.predict()
        # return self.fragment_masses
        return [[mass for mass in el if mass != -1] for el in self.fragment_masses]

    def get_fragment_annotation(self):
        if self.predictions_done == False:
            self.predict()
        # return self.fragment_masses

        annotation = []
        for masses in self.fragment_masses:
            valid_annotation = []
            for annotation_type in C.ANNOTATION:
                valid_annotation_types = []
                for mass, annotation_element in zip(masses, annotation_type):
                    if mass != -1:
                        valid_annotation_types.append(annotation_element)
                valid_annotation.append(valid_annotation_types)
                # convert annotation list to dictionary
            valid_annotation = {key: value for key, value in zip(["type", "charge", "number"], valid_annotation)}
            annotation.append(valid_annotation)

        return annotation
