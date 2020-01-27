#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predictROSIT.py is a client obtaining PROSIT predictions

__author__ = "Daniela Andrade Salazar"
__email__ = "daniela.andrade@tum.de"
"""

import pandas as pd
import os
import numpy as np
import time

import threading

import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

#import tensorflow.python.util.deprecation as deprecation
#deprecation._PRINT_DEPRECATION_WARNINGS = False
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import prosit_grpc_client._constants__ as C  # For constants
from prosit_grpc_client.__utils__ import compute_ion_masses, normalize_intensities  # For ion mass computation


class PredictPROSIT:

    def __init__(self, server: str, 
                 sequences_list: Union[np.ndarray, Iterable], 
                 charges_list: Union[np.ndarray, Iterable],
                 collision_energies_list: Union[np.ndarray, Iterable], 
                 concurrency: Optional[int] = 1, 
                 scan_nums_list: Optional[Union[np.ndarray, Iterable]] = None,
                 ids_list: Optional[Union[np.ndarray, Iterable]] = None,
                 normalize: Optional[bool] = False, 
                 labels_list: Optional[Union[np.ndarray, Iterable]] = None
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
        :param scanNums: List of Lists
                         Every list contains the scan numbers of the sequences, optional
        :param normalize: Flag for indicating whether to normalize the output or not
        """
        
        # Test if inputs are None
        if any(x is None for x in (sequences_list, charges_list, collision_energies_list)):
            raise ValueError("[ERROR] Define sequences, charges and collision energies first!")
        
        # Test types and shapes of inputs
        if not len({len(i) for i in (sequences_list, charges_list, collision_energies_list)}) == 1:
            raise ValueError("[ERROR] Unequal lengths of inputs")

        self.scan_nums_list = scan_nums_list
        self.ids_list = ids_list
        self.sequences_list = sequences_list
        self.charges_list = charges_list
        self.collision_energies_list = collision_energies_list
        self.num_batches = len(sequences_list)
        self.normalize = normalize
        self.labels_list = labels_list
        
        self.concurrency = concurrency
        self._condition = threading.Condition()
        self._done = 0
        self._active = 0
        
        self.server = server
        # Create channel and stub
        self.channel = grpc.insecure_channel(self.server)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        
        self.outputs = {}  # for PROSIT OUTPUTS, key is the dataset number
        print("[INFO] Initialized predicter with {} batches".format(self.num_batches))
        
    class OutputPROSIT:
        
        def __init__(self, response_outputs, sequences, charges, collision_energies, 
                     scan_nums=None, ids=None, labels=None):
            """
            :param response_outputs: output of PROSIT algorithm
            """
            
            self.sequences = sequences.reshape(-1, 30)  # reshaped to num_seq, 30  
            self.charges = charges.reshape(-1,6)  # reshaped to unum_seq, 6
            self.collision_energies = collision_energies
            self.scan_nums = scan_nums
            self.ids = ids
            self.labels = labels
            self.num_seq = self.sequences.shape[0] 
            
            self.predictions = np.array(response_outputs['out/Reshape:0'].float_val)

            #self.reshaped_output = None  # after reshape
            #self.normalized_predictions = None  # after normalizing
            #self.final_predictions = None  # after filtering invalid and setting negative to 0
            
        def reshape_output(self):
            """
            assume output array from prosit (one dimensional)
            set array to dimensions (num_seq, 174)
            """
            shape_output = 2 * 3 * (C.SEQ_LEN - 1) # 2 type of ions, 3 charges for each ion, 29 possibe frags --> 174
            self.predictions = self.predictions.reshape(self.num_seq, shape_output)  # -1 stands for num_seq

        def normalize_output(self):
            """
            assume reshaped and filtered output of prosit, shape should be (num_seq, 174) normalized along first axis 
            set normalized output between 0 and 1
            """            
            self.predictions = normalize_intensities(self.predictions)
            self.predictions[self.predictions<0] = -1
            
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
            #self.final_predictions = self.normalized_predictions.copy()
            
            #print("Number of sequences", self.num_seq)
            for i in range(self.num_seq):
                # 1. Filter by charges 
                charge, = np.where(self.charges[i] == 1)[0] + 1  # Charge sould be between 1 and 6                
                preds = self.predictions[i]
                #print("[INFO] Charge of sequence is {}".format(charge))
                if charge == 1:
                    # if charge == 1: all +2 & +3 invalid i.e. indexes only valid are indexes 0,3,6,9,...
                    # invalid are x mod 3 != 0
                    invalid_indexes = [(x*3 + 1) for x in range(C.SEQ_LEN)] + [(x*3 + 2) for x in range(C.SEQ_LEN)]
                    preds[invalid_indexes] = -1
                elif charge == 2:
                    # if charge == 1: all +2 & +3 invalid i.e. indexes only valid are indexes 0,1,3,4,6,7,9,10...
                    # invalid are x mod 3 == 2
                    invalid_indexes = [x*3 + 2 for x in range(C.SEQ_LEN)]
                    preds[invalid_indexes] = -1
                else:
                    if charge > 6:
                        print("[ERROR] in charge greater than 6")
                        return False
                    # charge >= 3 --> all valid
                    #print("No filtering by charges")
                
                self.predictions[i] = preds   
                
                # 2. Filter by input seq
                index_list = np.where(self.sequences[i] == 0)[0]
                len_seq =  index_list[0] if any(index_list) else C.SEQ_LEN  # e.g. seq is [1,4,1,1,0,...,0] then len is 4
                #print("[INFO] Sequence has length", len_seq)
                if len_seq < C.SEQ_LEN:
                    self.predictions[i, (len_seq-1)*6:] = -1 # valid indexes are less than len_seq * 6
            return True
        
            
        def set_negative_to_zero(self):
            """
            assume reshaped and filtered output or prosit, shape should be (num_seq, 174)
            set output with positions <0 set to 0
            """
            self.predictions[(self.predictions!=-1) & (self.predictions<0)] = 0

    @staticmethod
    def sequence_numbers_to_alpha(x):
        """
        :param seq: list of letters to be converted to numbers 
        """
        return [C.AMINO_ACIDS_INT[n] for n in x]

    @staticmethod
    def _create_request(model_name="intensity", signature_name="serving_default"):
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
            print("DONE with BATCH",dataset_index)
        
    def set_collision_energies(ce):
        self.collision_energies_list = ce
        
    def set_charges(c):
        self.charges_list = c

    def set_sequences(s):
        self.sequences_list = s

    def prosit_callback(self, result_future, dataset_index, sequences, charges, collision_energies, 
                        scan_nums=None, ids=None, labels=None, verbose=False):   
        """
        Wraper for callback funtion needed to add parameters
        returns the callback function
        """
        
        def _callback(result_future):
            """
            Calculates the statistics for the prediction result.
            :param result_future:
            """

            exception = result_future.exception()
            if exception:
                
                print("[EXCEPTION for BATCH {}] {}".format(dataset_index + 1,exception))
            else:
                print("[BATCH {}/{}] Start!".format(dataset_index + 1, self.num_batches))
                # Get output
                response_outputs = result_future.result().outputs

                # Create Output object
                output = PredictPROSIT.OutputPROSIT(response_outputs=response_outputs, sequences=sequences, 
                                                    charges=charges, collision_energies=collision_energies, 
                                                    scan_nums=scan_nums, ids=ids, labels=labels)
                output.reshape_output()  # Reshape Output
                print("[BATCH {}/{}] Reshaped result to shape {}".format(dataset_index + 1, self.num_batches,
                                                                         output.predictions.shape))
                
                filter_flag = output.filter_invalid()  # Set invalid positions to -1
                if not filter_flag:  
                    # Increase number of done and decrease number of active
                    self.inc_done(dataset_index)
                    self.dec_active(dataset_index)
                    return 
                
                output.set_negative_to_zero()  # Set negative positions to 0
                print("[BATCH {}/{}] Set invalid positions to -1 and negative positions to 0".format(dataset_index + 1,
                                                                                                    self.num_batches ))
                # Normalize Output
                if self.normalize:
                    output.normalize_output()
                    print("[BATCH {}/{}] Normalized predictions".format(dataset_index + 1, self.num_batches))
                
                if verbose:
                    print("[BATCH {}/{}] OUTPUT scan_nums, preds: ")
                    print(output.scan_nums)
                    print(output.predictions)
                    
                # Add output to outputs
                self.outputs[dataset_index] = output
                
            # Increase number of done and decrease number of active
            self.inc_done(dataset_index)
            self.dec_active(dataset_index)
                
        return _callback

    def get_predictions(self, verbose=False):
        """
        Tests PredictionService with concurrent requests.
        
        :return: The classification error rate.
        :raises IOError: An error occurred processing test data set.
        """
        # Start with predictions
        start = time.time()
        for i in range(self.num_batches):

            sequences = self.sequences_list[i]
            collision_energies = self.collision_energies_list[i]
            charges = self.charges_list[i]
            
            scan_nums = self.scan_nums_list[i] if self.scan_nums_list else None
            ids = self.ids_list[i] if self.ids_list else None
            labels = self.labels_list[i] if self.labels_list else None
            
            # Compute number of sequences, assume every sequence has length 30
            num_seq = int(len(sequences)/C.SEQ_LEN)
            # print("[INFO] Working with {} sequences".format(num_seq))

            # Convert inputs to numpy arrays
            sequences = np.array(sequences).astype(np.int32)  
            charges = np.array(charges).astype(np.float32)  
            collision_energies = np.array(collision_energies).astype(np.float32)  

            # Create request
            request = self._create_request()

            # Parse inputs to request
            request.inputs['peptides_in:0'].CopyFrom(tf.contrib.util.make_tensor_proto(sequences, shape=[num_seq, C.SEQ_LEN]))
            request.inputs['collision_energy_in:0'].CopyFrom(tf.contrib.util.make_tensor_proto(collision_energies, shape=[num_seq, 1]))
            request.inputs['precursor_charge_in:0'].CopyFrom(tf.contrib.util.make_tensor_proto(charges, shape=[num_seq, 6]))
            # print("[INFO] Finished parsing inputs")
            
            self.throttle(i)  #  
            timeout = 5  # in seconds
            result_future = self.stub.Predict.future(request, timeout)  # asynchronous request
            # Callback function
            result_future.add_done_callback(self.prosit_callback(result_future=result_future, dataset_index=i, sequences=sequences, 
                                                                 charges=charges, collision_energies=collision_energies, 
                                                                 scan_nums=scan_nums, ids=ids, labels=labels, verbose=verbose)) 
        # Wait untill all request finish
        with self._condition:
            while self._done != self.num_batches:
                self._condition.wait()
        end = time.time()
        print("[INFO] Done sending {} requests. Time needed: {}".format(self.num_batches, end-start))

    def predictions_to_dfs(self, add_scan_nums=False, add_ion_masses=False, add_ids=False, add_labels=False):
        """
        For every dataset in sequences_list write a csv file with the following columns:
        sequence, normalized_intensities, raw_intensities
        """
        print("[INFO] Saving outputs to dataframes. Number of BATCHES:", len(self.outputs))
        dfs = {}        
        col_names = ["peptide_str", "peptide_int", "charge", "intensity_array",]
                    
        for i, out in self.outputs.items():
            sequences = out.sequences.reshape(-1,30) # Assume 30 acids per sequence
            sequences_alpha = []
            for seq in sequences:
                    sequences_alpha.append("".join(self.sequence_numbers_to_alpha(seq)))
    
            l =  list(zip(sequences_alpha, sequences, out.charges, out.predictions))
            df = pd.DataFrame(l,columns=col_names)
            
            if add_scan_nums & (out.scan_nums is not None):
                print("[INFO] Added scan nums")
                df["scans"] = out.scan_nums
                
            if add_ids & (out.ids is not None):
                print("[INFO] Added ids")
                df["id"] = out.ids
            
            if add_labels & (out.labels is not None):
                print("[INFO] Added Labels!")
                df["Label"] = out.labels
            
            # Add ion masses
            if add_ion_masses:
                df["m/z array"] = df.apply(lambda x: ",".join(compute_ion_masses(x.peptide_int,
                                                                                 x.charge).astype(str)), axis=1)
            # Change charge from one-hot to num
            df["charge"] = df["charge"].apply(lambda x: list(x).index(1)+1)
        
            # Convert np arrays to strings
            df["peptide_int"] = df["peptide_int"].apply(lambda x: ",".join(x.astype(str)))
            df["intensity_array"] = df["intensity_array"].apply(lambda x: ",".join(x.astype(str)))
            dfs[i] = df
            print("\t Done with {}/{} batches!".format(i+1, self.num_batches))
        return dfs
        
        
           
