import numpy as np
from tqdm import tqdm
from . import __constants__ as C  # For constants
from . import __utils__ as U
from tensorflow_serving.apis import predict_pb2
import tensorflow as tf
from math import ceil

# surpresses tensorflow deprecation warning that would otherwise cause buggy behaviour of the tqdm progess bar
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class Base:
    """
    Base is the base class for all PredObject.
    The subclasses should each implement the methods:
        create_request
        unpack_response
        prepare_input
        prepare_output
    The remaining methods will most likely stay the same.
    """
    def __init__(self, stub, model_name, input):
        self.model_name = model_name
        self.stub = stub
        self.input = input
        self.predictions = None  # overwritten in predict function

    @staticmethod
    def create_request_scaffold(model_name, signature_name="serving_default"):
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

    # PredObjectType specific
    def create_request(self, model_name, inputs_batch, batchsize):
        """create_request is a function to create a single request of the PredObject that is implementing it

        :param model_name      specify the model that should be used to predict
        :param inputs_batch    inputs necessar for request
        :param batchsize       size of the created request

        :return                request ready to be sent to the server
        """
        pass

    def create_requests(self, model_name: str, inputs: dict):
        """create_requests is a generator to create all requests for a specific PredObject

        -- Non optional Parameters --
        :param model_name str equal to a valid Prosit model_name
        :param inputs dict of all inputs necessary for the chosen PredObject
        """

        # ensures that all inputs have the same number of rows
        for i in inputs.values():
            for j in inputs.values():
                assert len(i) == len(j)

        batch_start = 0
        num_seq = len(next(iter(inputs.values())))  # select the length of the first input parameter
        while batch_start < num_seq:
            batch_end: int = batch_start + C.BATCH_SIZE
            batch_end: int = min(num_seq, batch_end)
            batchsize: int = batch_end - batch_start

            inputs_batch = {}
            for key, value in inputs.items():
                inputs_batch[key] = value[batch_start:batch_end]
            batch_start: int = batch_end

            yield self.create_request(model_name, inputs_batch, batchsize)

    @staticmethod
    def unpack_response(response):
        """
        :return prediction formatted as numpy array
        """
        return None

    def send_request(self, request, timeout):
        response = self.stub.Predict.future(
            request, timeout).result()
        return self.unpack_response(response)

    def send_requests(self, requests, disable_progress_bar: bool):
        timeout = 10  # in seconds

        predictions = []
        n_batches = ceil(len(self.input.sequences.array)/C.BATCH_SIZE)
        for request in tqdm(requests,
                            disable=disable_progress_bar,
                            total=n_batches):
        # for request in requests:
            prediction = self.send_request(request=request, timeout=timeout)
            predictions.append(prediction)

        predictions = np.vstack(predictions)
        return predictions

    def prepare_input(self):
        pass

    def predict(self, disable_progress_bar):
        input_dict = self.prepare_input()
        requests = self.create_requests(model_name=self.model_name, inputs=input_dict)
        predictions = self.send_requests(requests=requests,
                                         disable_progress_bar=disable_progress_bar)
        self.predictions = predictions

    def prepare_output(self):
        pass


class Intensity(Base):
    def create_request(self, model_name, inputs_batch, batchsize):
        request = self.create_request_scaffold(model_name=model_name)
        request.inputs['peptides_in:0'].CopyFrom(
            tf.contrib.util.make_tensor_proto(inputs_batch["seq_array"],
                                              shape=[batchsize, C.SEQ_LEN],
                                              dtype=np.int32))

        request.inputs['collision_energy_in:0'].CopyFrom(
            tf.contrib.util.make_tensor_proto(inputs_batch["ce_array"],
                                              shape=[batchsize, 1],
                                              dtype=np.float32))

        request.inputs['precursor_charge_in:0'].CopyFrom(
            tf.contrib.util.make_tensor_proto(inputs_batch["charges_array"],
                                              shape=[batchsize, C.NUM_CHARGES_ONEHOT],
                                              dtype=np.float32))
        return request

    @staticmethod
    def unpack_response(response):
        """
        :return prediction formatted as numpy array
        """
        outputs_tensor_proto = response.outputs["out/Reshape:0"]
        shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)
        return np.array(outputs_tensor_proto.float_val, dtype=np.float32).reshape(shape.as_list())

    def prepare_input(self):
        in_dic = {
            "seq_array": self.input.sequences.array,
            "ce_array": self.input.collision_energies.array,
            "charges_array": self.input.charges.array
        }
        return in_dic

    @staticmethod
    def create_masking(charges_array, sequences_lengths):
        """
        assume reshaped output of prosit, shape sould be (num_seq, 174)
        set filtered output where not allowed positions are set to -1
        prosit output has the form:
        y1+1     y1+2 y1+3     b1+1     b1+2 b1+3     y2+1     y2+2 y2+3     b2+1     b2+2 b2+3
        if charge >= 3: all allowed
        if charge == 2: all +3 invalid
        if charge == 1: all +2 & +3 invalid
        """

        assert len(charges_array) == len(sequences_lengths)

        mask = np.ones(shape=(len(charges_array),
                              C.VEC_LENGTH), dtype=np.int32)

        for i in range(len(charges_array)):
            charge_one_hot = charges_array[i]
            len_seq = sequences_lengths[i]
            m = mask[i]

            # filter according to peptide charge
            if np.array_equal(charge_one_hot, [1, 0, 0, 0, 0, 0]):
                invalid_indexes = [(x * 3 + 1) for x in range((C.SEQ_LEN - 1) * 2)] + [(x * 3 + 2) for x in
                                                                                       range((C.SEQ_LEN - 1) * 2)]
                m[invalid_indexes] = -1

            elif np.array_equal(charge_one_hot, [0, 1, 0, 0, 0, 0]):
                invalid_indexes = [
                    x * 3 + 2 for x in range((C.SEQ_LEN - 1) * 2)]
                m[invalid_indexes] = -1

            if len_seq < C.SEQ_LEN:
                invalid_indexes = range((len_seq - 1) * 6, C.VEC_LENGTH)
                m[invalid_indexes] = -1

        return mask

    def apply_masking(self):

        invalid_indices = self.mask == -1

        self.data["intensity"][invalid_indices] = -1
        self.data["fragmentmz"][invalid_indices] = -1

        self.data["annotation"]["charge"][invalid_indices] = -1
        self.data["annotation"]["number"][invalid_indices] = -1
        self.data["annotation"]["type"][invalid_indices] = None

    def normalize_intensity(self):
        self.data["intensity"] = U.normalize_intensities(self.data["intensity"])

        self.data["intensity"][self.data["intensity"] < 0] = 0
        self.data["intensity"][self.mask == -1] = -1

    def prepare_output(self):

        n_seq = len(self.predictions)

        # prepare raw state of spectrum
        self.data = {
            "intensity": self.predictions,
            "fragmentmz": np.array([U.compute_ion_masses(self.input.sequences.array[i],
                                                         self.input.charges.array[i])
                                                         for i in range(n_seq)],
                                   dtype=np.float32),
            "annotation": {
                "charge": np.array([C.ANNOTATION[1] for _ in range(n_seq)], dtype=np.uint8),
                "number": np.array([C.ANNOTATION[2] for _ in range(n_seq)], dtype=np.uint8),
                # limited to single character unicode string
                "type": np.array([C.ANNOTATION[0] for _ in range(n_seq)], dtype=np.dtype('U1'))
            }
        }

        # create and apply masking
        self.mask = self.create_masking(charges_array=self.input.charges.array,
                                        sequences_lengths=self.input.sequences.lengths)
        self.apply_masking()

        # normalize intensities
        self.normalize_intensity()

        # # create and apply filter for masses with -1
        # self.create_filter()
        # self.apply_filter()

        self.output = self.data

class Irt(Base):
    def create_request(self, model_name, inputs_batch, batchsize):
        request = self.create_request_scaffold(model_name=model_name)
        request.inputs['sequence_integer'].CopyFrom(
            tf.contrib.util.make_tensor_proto(inputs_batch["seq_array"],
                                              shape=[batchsize, C.SEQ_LEN],
                                              dtype=np.int32))
        return request

    @staticmethod
    def unpack_response(response):
        """
        :return prediction formatted as numpy array
        """
        outputs_tensor_proto = response.outputs["prediction/BiasAdd:0"]
        shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)
        return np.array(outputs_tensor_proto.float_val, np.float32).reshape(shape.as_list())

    def prepare_input(self):
        in_dic = {
            "seq_array": self.input.sequences.array
        }
        return in_dic

    def prepare_output(self):
        self.output = (self.predictions*43.39373 + 56.35363441)

class Proteotypicity(Base):
    def create_request(self, model_name, inputs_batch, batchsize):
        request = self.create_request_scaffold(model_name=model_name)
        request.inputs['peptides_in_1:0'].CopyFrom(
            tf.contrib.util.make_tensor_proto(inputs_batch["seq_array"],
                                              shape=[batchsize, C.SEQ_LEN],
                                              dtype=np.float32))
        return request

    @staticmethod
    def unpack_response(response):
        """
        :return prediction formatted as numpy array
        """
        outputs_tensor_proto = response.outputs["pep_dense4/BiasAdd:0"]
        shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)
        return np.array(outputs_tensor_proto.float_val, dtype=np.float16).reshape(shape.as_list())

    def prepare_input(self):
        in_dic = {
            "seq_array": self.input.sequences.array
        }
        return in_dic

    def prepare_output(self):
        self.output = self.predictions

class Charge(Base):
    def create_request(self, model_name, inputs_batch, batchsize):
        request = self.create_request_scaffold(model_name=model_name)
        request.inputs['peptides_in_1:0'].CopyFrom(
            tf.contrib.util.make_tensor_proto(inputs_batch["seq_array"],
                                              shape=[batchsize, C.SEQ_LEN],
                                              dtype=np.float32))
        return request

    @staticmethod
    def unpack_response(response):
        """
        :return prediction formatted as numpy array
        """
        outputs_tensor_proto = response.outputs["softmax/Softmax:0"]
        shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)
        return np.array(outputs_tensor_proto.float_val, dtype=np.float16).reshape(shape.as_list())

    def prepare_input(self):
        in_dic = {
            "seq_array": self.input.sequences.array
        }
        return in_dic

    def prepare_output(self):
        self.output = self.predictions