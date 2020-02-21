import numpy as np
import prosit_grpc.predictPROSIT as prpc

def test_PROSITcollisionenergies_determine_type():

    # numeric
    ce = [10, 20, 30, 40, 50]
    ret = prpc.PROSITcollisionenergies.determine_type(ce)
    assert ret == "numeric"

    # procentual
    ce = [0.10, 0.20, 0.30, 0.40, 0.50]
    ret = prpc.PROSITcollisionenergies.determine_type(ce)
    assert ret == "procentual"

    # array
    ce = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype= np.float32)
    ret = prpc.PROSITcollisionenergies.determine_type(ce)
    assert ret == "array"

def