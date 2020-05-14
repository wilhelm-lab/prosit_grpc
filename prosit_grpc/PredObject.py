import numpy as np
import tqdm
from . import __constants__ as C  # For constants
from tensorflow_serving.apis import predict_pb2
import tensorflow as tf


class PredObjectBase:
    """
    PredObjectBase is the base class for all PredObject.
    The subclasses should each implement the methods:
        create_request
        unpack_response
        prepare_input
        prepare_output
    The remaining methods will most likely stay the same.
    """
    def __init__(self, stub, model_name, PROSITinput):
        self.model_name = model_name
        self.stub = stub
        self.PROSITinput = PROSITinput
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
            batch_start: int = batch_end

            inputs_batch = {}
            for key, value in inputs.items():
                inputs_batch[key] = value[batch_start:batch_end]

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
        timeout = 5  # in seconds

        predictions = []
        for request in tqdm(requests, disable=disable_progress_bar):
            prediction = self.send_request(request=request,timeout=timeout)
            predictions.append(prediction)

        predictions = np.vstack(predictions)
        return predictions

    def prepare_input(self):
        pass

    def predict(self):
        input_dict = self.prepare_input()
        requests = self.create_requests(model_name=self.model_name, inputs=input_dict)
        predictions = self.send_requests(requests=requests,
                                         disable_progress_bar=False)
        self.predictions = predictions

    def prepare_output(self):
        pass


class PredObjectIntensity(PredObjectBase):
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
        return np.array(outputs_tensor_proto.float_val).reshape(shape.as_list())

    def prepare_input(self):
        in_dic = {
            "seq_array": self.PROSITinput.sequences.array,
            "ce_array": self.PROSITinput.collision_energies.array,
            "charge_array": self.PROSITinput.charges.array
        }
        return in_dic

    def prepare_output(self):
        pass


class PredObjectIrt(PredObjectBase):
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
        outputs_tensor_proto = response.outputs["prediction/BiasAdd:0"]
        shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)
        return np.array(outputs_tensor_proto.float_val).reshape(shape.as_list())

    def prepare_input(self):
        in_dic = {
            "seq_array": self.PROSITinput.sequences.array
        }
        return in_dic

    def prepare_output(self):
        pass


class PredObjectProteotypicity(PredObjectBase):
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
        return np.array(outputs_tensor_proto.float_val).reshape(shape.as_list())

    def prepare_input(self):
        in_dic = {
            "seq_array": self.PROSITinput.sequences.array
        }
        return in_dic

    def prepare_output(self):
        pass

class PredObjectCharge(PredObjectBase):
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
        return np.array(outputs_tensor_proto.float_val).reshape(shape.as_list())

    def prepare_input(self):
        in_dic = {
            "seq_array": self.PROSITinput.sequences.array
        }
        return in_dic

    def prepare_output(self):
        pass