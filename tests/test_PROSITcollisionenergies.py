import numpy as np
import prosit_grpc.inputPROSIT as prpc

ce_numeric = [10, 20, 30, 40, 50]
ce_procentual = [0.10, 0.20, 0.30, 0.40, 0.50]
ce_array = np.array(ce_procentual, dtype=np.float32)
ce_array.shape = (5,1)

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
    ce = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    ret = prpc.PROSITcollisionenergies.determine_type(ce)
    assert ret == "array"


def test_PROSITcollisionenergies_init():
    # numeric
    inst = prpc.PROSITcollisionenergies(ce_numeric)
    assert inst.numeric == ce_numeric

    # procentual
    inst = prpc.PROSITcollisionenergies(ce_procentual)
    assert inst.procentual == ce_procentual

    # array
    inst = prpc.PROSITcollisionenergies(ce_array)
    assert np.array_equal(inst.array, ce_array)


def test_PROSITcollisionenergies_prepare_collisionenergies():
    inst = prpc.PROSITcollisionenergies(ce_numeric)
    inst.prepare_collisionenergies()

    assert inst.numeric == ce_numeric
    assert inst.procentual == ce_procentual
    assert np.array_equal(inst.array, ce_array)
