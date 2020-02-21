from . import __constants__ as C
import numpy as np

import itertools
import tensorflow as tf
from sklearn.preprocessing import normalize
from tensorflow_serving.apis import predict_pb2

def compute_ion_masses(seq_int,charge_onehot):
    """ 
    Collects an integer sequence e.g. [1,2,3] with charge 2 and returns array with 174 positions for ion masses. 
    Invalid masses are set to -1
    charge_one is a onehot representation of charge with 6 elems for charges 1 to 6
    """
    charge = list(charge_onehot).index(1) + 1
    if not (charge in (1,2,3,4,5,6) and len(charge_onehot)==6):
        print("[ERROR] One-hot-enconded Charge is not in valid range 1 to 6")
        return
    
    if not len(seq_int) == C.SEQ_LEN:
        print("[ERROR] Sequence length {} is not desired length of {}".format(len(seq_int), C.SEQ_LEN))
        return 
    
    l = list(seq_int).index(0) if 0 in seq_int else C.SEQ_LEN 
    masses = np.ones((C.SEQ_LEN-1)*2*3)*-1
    mass_b = 0
    mass_y = 0
    j = 0  # iterate over masses

    # Iterate over sequence, sequence should have length 30
    for i in range(l-1):  # only 29 possible ios
        j = i*6 # index for masses array at position 

        #### MASS FOR Y IONS
        # print("Addded", C.VEC_MZ[seq_int[l-1-i]])
        mass_y += C.VEC_MZ[seq_int[l-1-i]]
        

        # Compute charge +1
        masses[j] = (mass_y + 1*C.MASSES["PROTON"] + C.MASSES["C_TERMINUS"] + C.MASSES["H"])/1.0 
        # Compute charge +2
        masses[j+1] = (mass_y + 2*C.MASSES["PROTON"] + C.MASSES["C_TERMINUS"] + C.MASSES["H"])/2.0 if charge>=2 else -1.0
        # Compute charge +3
        masses[j+2] = (mass_y + 3*C.MASSES["PROTON"] + C.MASSES["C_TERMINUS"] + C.MASSES["H"])/3.0 if charge>= 3.0 else -1.0


        ### MASS FOR B IONS 
        mass_b += C.VEC_MZ[seq_int[i]]

        # Compute charge +1
        masses[j+3] = (mass_b + 1*C.MASSES["PROTON"] + C.MASSES["N_TERMINUS"] - C.MASSES["H"])/1.0 
        # Compute charge +2
        masses[j+4] = (mass_b + 2*C.MASSES["PROTON"] + C.MASSES["N_TERMINUS"] - C.MASSES["H"])/2.0 if charge>=2 else -1.0
        # Compute charge +3
        masses[j+5] = (mass_b + 3*C.MASSES["PROTON"] + C.MASSES["N_TERMINUS"] - C.MASSES["H"])/3.0 if charge>= 3.0 else -1.0
    
    return masses


def normalize_intensities(x, norm="max"):
    """
    This function normalizes the given intensity array of shape (num_seq, num_peaks)
    
    """
    return normalize(x, axis=1, norm=norm)

def map_peptide_to_numbers(seq):
    """
    Map string of peptide sequence to numeric list based on dictionary ALPHABET
    """
    nums = []
    i = 0        
    seq = seq.replace(" ", "")
    l = len(seq)
    while i<l:
        # Special Cases: CaC, OxM, M(ox), M(O), PhS, PhT, PhY, (Cam)
        if (i+3)<l and seq[i:i+3] == "Cac": 
            nums.append(C.ALPHABET["Cac"])
            i += 3
        elif (i+3)<l and seq[i:i+3] == "PhS": 
            nums.append(C.ALPHABET["PhS"])
            i += 3
        elif (i+3)<l and seq[i:i+3] == "PhT": 
            nums.append(C.ALPHABET["PhT"])
            i += 3
        elif (i+3)<l and seq[i:i+3] == "PhY": 
            nums.append(C.ALPHABET["PhY"])
            i += 3
        elif (i+3)<l and seq[i:i+3] == "OxM": 
            nums.append(C.ALPHABET["OxM"])
            i += 3
        elif (i+4)<l and seq[i:i+4] == "M(O)": 
            nums.append(C.ALPHABET["M(O)"])
            i += 4
        elif (i+5)<l and seq[i:i+5] == "M(ox)": 
            nums.append(C.ALPHABET["M(ox)"])
            i += 5
        elif (i+5)<l and seq[i:i+5] == "(Cam)": 
            nums.append(C.ALPHABET["(Cam)"])
            i += 5
        # Single char is in ALPHABET
        elif seq[i] in C.ALPHABET:
            nums.append(C.ALPHABET[seq[i]])
            i +=1
        else:
            print("Char {} not found in sequence {}".format(seq[i], seq))
            nums.append(-1)
            i += 1
    return nums
    
def flatten_list(l_2d):
    """ 
    Concatenate lists into one
    """
    return list(itertools.chain(*l_2d))

def indices_to_one_hot(data, nb_classes):
    """
    Convert an iterable of indices to one-hot encoded labels.
    :param data: charge, int between 1 and 6
    """
    targets = np.array([data-1])  # -1 for 0 indexing 
    return np.int_((np.eye(nb_classes)[targets])).tolist()[0]

def fill_zeros(x, fixed_length):
    """
    Fillzeros in an array to match desired fixed length
    """
    res = np.zeros(fixed_length)
    _l = min(fixed_length, len(x))
    res[:_l] = x[:_l]
    return list(np.int_(res))

def _create_request(model_name, signature_name="serving_default"):
    """
    :param model_name: Model name (taken from PROSIT)
    :param signature_name: Signature Name for the estimator (serving_default is by default set with custom tf estimator)
    :return created request
    """
    # Create Request
    request = predict_pb2.PredictRequest()
    # Model and Signature Name
    request.model_spec.name = model_name
    request.model_spec.signature_name = signature_name
    # print("[INFO] Set model and signature name")
    return request


def create_request_intensity(seq_array, ce_array, charges_array, batchsize, model_name):
    """
    seq_array
    ce_array
    charges_array
    batchsize       size of the created request
    model_name      specify the model that should be used to predict
    return:         request ready to be sent ther server
    """
    request = _create_request(model_name=model_name)
    request.inputs['peptides_in:0'].CopyFrom(
        tf.contrib.util.make_tensor_proto(seq_array, shape=[batchsize, C.SEQ_LEN]))
    request.inputs['collision_energy_in:0'].CopyFrom(
        tf.contrib.util.make_tensor_proto(ce_array, shape=[batchsize, 1]))
    request.inputs['precursor_charge_in:0'].CopyFrom(
        tf.contrib.util.make_tensor_proto(charges_array, shape=[batchsize, 6]))
    return request


def create_request_proteotypicity(seq_array, batchsize, model_name):
    """
    seq array
    batchsize
    model_name  specify the model used for prediction
    """
    request = _create_request(model_name=model_name)
    request.inputs['peptides_in_1:0'].CopyFrom(
            tf.contrib.util.make_tensor_proto(seq_array, shape=[batchsize, C.SEQ_LEN]))
    return request


def create_request_irt(seq_array, batchsize, model_name):
    """
    seq array
    batchsize
    model_name  specify the model used for prediction
    """
    request = _create_request(model_name=model_name)
    request.inputs['sequence_integer'].CopyFrom(
        tf.contrib.util.make_tensor_proto(seq_array, shape=[batchsize, C.SEQ_LEN]))
    return request


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