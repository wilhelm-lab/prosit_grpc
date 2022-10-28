import numpy as np

import prosit_grpc.inputPROSIT as prpc

# set test variables
charges_numeric = [1, 2, 3, 4, 5, 6]
charges_onehot = [
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
]
charges_array = np.array(charges_onehot, dtype=np.int32)


def test_prosit_charges_determine_type():
    """Test Prosit charges determine type."""
    # numeric
    ret = prpc.PROSITcharges.determine_type(charges_numeric)
    assert ret == "numeric"

    # one hot
    ret = prpc.PROSITcharges.determine_type(charges_onehot)
    assert ret == "one-hot"

    # array
    ret = prpc.PROSITcharges.determine_type(charges_array)
    assert ret == "array"


def test_prosit_charges_init():  # test instanciation
    """Test Prosit charges init."""
    # numeric
    inst = prpc.PROSITcharges(charges_numeric)
    assert inst.numeric == charges_numeric

    # one hot
    inst = prpc.PROSITcharges(charges_onehot)
    assert inst.onehot == charges_onehot

    # array
    inst = prpc.PROSITcharges(charges_array)
    assert np.array_equal(inst.array, charges_array)


def test_prosit_charges_prepare_charges():
    """Test Prosit charges predapre charges."""
    inst = prpc.PROSITcharges(charges_numeric)
    inst.prepare_charges()
    assert inst.numeric == charges_numeric
    assert inst.onehot == charges_onehot
    assert np.array_equal(inst.array, charges_array)
