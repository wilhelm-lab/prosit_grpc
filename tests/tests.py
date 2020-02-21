import numpy as np
import prosit_grpc.predictPROSIT as prpc

# set test variables
charges_numeric = [1,2,3,4,5,6]
charges_onehot = [[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]]
charges_array = np.array(charges_onehot, dtype=np.int32)

sequences_character = ["KKVCQGTSNKLTQLGTFE",
                       "GNMYYENSYALAVLSNYDANKTGLK",
                       "LTQLGTFEDHFLSLQR",
                       "FRDEATCKDTCPPLMLYNPTTQMDVNPEGK",
                       "RDEATCKDTCPPLM(ox)LNPTTYQMDVNPEGK",
                       "FRDEATKDTCPPLMM(O)YNPTTYQMDVNPEGK",
                       "ANKEILDEAYVMASVDNPHVCR",
                       "FRDEATCKDT(Cam)PPLLYNPTTYQMDVNPEGK",
                       "FRDEATCKDTCacPPLMYNPTTYQMDVNPEGK"]
sequences_numeric = [[9, 9, 18, 2, 14, 6, 17, 16, 12, 9, 10, 17, 14, 10, 6, 17, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [6, 12, 11, 20, 20, 4, 12, 16, 20, 1, 10, 1, 18, 10, 16, 12, 20, 3, 1, 12, 9, 17, 6, 10, 9, 0, 0, 0, 0, 0],
                     [10, 17, 14, 10, 6, 17, 5, 4, 3, 7, 5, 10, 16, 10, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [5, 15, 3, 4, 1, 17, 2, 9, 3, 17, 2, 13, 13, 10, 11, 10, 20, 12, 13, 17, 17, 14, 11, 3, 18, 12, 13, 4, 6, 9],
                     [15, 3, 4, 1, 17, 2, 9, 3, 17, 2, 13, 13, 10, 21, 10, 12, 13, 17, 17, 20, 14, 11, 3, 18, 12, 13, 4, 6, 9, 0],
                     [5, 15, 3, 4, 1, 17, 9, 3, 17, 2, 13, 13, 10, 11, 21, 20, 12, 13, 17, 17, 20, 14, 11, 3, 18, 12, 13, 4, 6, 9],
                     [1, 12, 9, 4, 8, 10, 3, 4, 1, 20, 18, 11, 1, 16, 18, 3, 12, 13, 7, 18, 2, 15, 0, 0, 0, 0, 0, 0, 0, 0],
                     [5, 15, 3, 4, 1, 17, 2, 9, 3, 17, 2, 13, 13, 10, 10, 20, 12, 13, 17, 17, 20, 14, 11, 3, 18, 12, 13, 4, 6, 9],
                     [5, 15, 3, 4, 1, 17, 2, 9, 3, 17, 2, 13, 13, 10, 11, 20, 12, 13, 17, 17, 20, 14, 11, 3, 18, 12, 13, 4, 6, 9]]

sequences_array = np.array(sequences_numeric, dtype=np.int32)

ce_numeric = [10,20,30,40,50]
ce_procentual = [0.10,0.20,0.30,0.40,0.50]
ce_array = np.array(ce_procentual, dtype=np.float32)

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

def test_PROSITsequences_determine_type():
    # character
    seq = ["AAAAAAAA", "AAAAAAAA", "AAAAAAAA", "AAAAAAAA"]
    ret = prpc.PROSITsequences.determine_type(seq)
    assert ret == "character"

    # numeric
    seq = [[1,1,1,1,1,1,1], [1,1,1,1,1,1,1], [1,1,1,1,1,1,1], [1,1,1,1,1,1,1]]
    ret = prpc.PROSITsequences.determine_type(seq)
    assert ret == "numeric"

    # array
    seq = np.array([[1,1,1,1,1,1,1], [1,1,1,1,1,1,1], [1,1,1,1,1,1,1], [1,1,1,1,1,1,1]], dtype=np.int32)
    ret = prpc.PROSITsequences.determine_type(seq)
    assert ret == "array"

def test_PROSITcharges_determine_type():
    # numeric
    ret = prpc.PROSITcharges.determine_type(charges_numeric)
    assert ret == "numeric"

    # one hot
    ret = prpc.PROSITcharges.determine_type(charges_onehot)
    assert ret == "one-hot"

    # array
    ret = prpc.PROSITcharges.determine_type(charges_array)
    assert ret == "array"

def test_PROSITcharges(): # test instanciation
    #numeric
    inst = prpc.PROSITcharges(charges_numeric)
    assert inst.numeric == charges_numeric

    # one hot
    inst = prpc.PROSITcharges(charges_onehot)
    assert inst.onehot == charges_onehot

    # array
    inst = prpc.PROSITcharges(charges_array)
    assert np.array_equal(inst.array, charges_array)

def test_PROSITcharges_prepare_charges():
    inst = prpc.PROSITcharges(charges_numeric)
    inst.prepare_charges()
    assert inst.numeric == charges_numeric
    assert inst.onehot == charges_onehot
    assert np.array_equal(inst.array, charges_array)

def test_PROSITsequences(): # test instanciation
    # character
    inst = prpc.PROSITsequences(sequences_character)
    assert inst.character == sequences_character

    # numeric
    inst = prpc.PROSITsequences(sequences_numeric)
    assert inst.numeric == sequences_numeric

    # array
    inst = prpc.PROSITsequences(sequences_array)
    assert np.array_equal(inst.array, sequences_array)

def test_PROSITsequences_prepare_sequences():
    inst = prpc.PROSITsequences(sequences_character)
    inst.prepare_sequences()

    assert inst.character == sequences_character
    assert inst.numeric == sequences_numeric
    assert np.array_equal(inst.array, sequences_array)

def test_PROSITcollisionenergies():
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
