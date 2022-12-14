import numpy as np

import prosit_grpc.inputPROSIT as prpc

# set sequences that should be checked
sequences_character = [
    "KKVCQGTSNKLTQLGTFE",
    "GNMYYENSYALAVLSNYDANKTGLK",
    "LTQLGTFEDHFLSLQR",
    "FRDEATCKDTCPPLMLYNPTTQMDVNPEGK",
    "RDEATCKDTCPPLM[UNIMOD:35]LNPTTYQMDVNPEGK",
    "FRDEATKDTCPPLMM[UNIMOD:35]YNPTTYQMDVNPEGK",
    "ANKEILDEAYVMASVDNPHVCR",
    "FRDEATCKDTC[UNIMOD:4]PPLLYNPTTYQMDVNPEGK",
    "FRDEATCKDTC[UNIMOD:4]PPLMYNPTTYQMDVNPEGK",
]

sequences_numeric = [
    [9, 9, 18, 24, 14, 6, 17, 16, 12, 9, 10, 17, 14, 10, 6, 17, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [6, 12, 11, 20, 20, 4, 12, 16, 20, 1, 10, 1, 18, 10, 16, 12, 20, 3, 1, 12, 9, 17, 6, 10, 9, 0, 0, 0, 0, 0],
    [10, 17, 14, 10, 6, 17, 5, 4, 3, 7, 5, 10, 16, 10, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 15, 3, 4, 1, 17, 24, 9, 3, 17, 24, 13, 13, 10, 11, 10, 20, 12, 13, 17, 17, 14, 11, 3, 18, 12, 13, 4, 6, 9],
    [15, 3, 4, 1, 17, 24, 9, 3, 17, 24, 13, 13, 10, 21, 10, 12, 13, 17, 17, 20, 14, 11, 3, 18, 12, 13, 4, 6, 9, 0],
    [5, 15, 3, 4, 1, 17, 9, 3, 17, 24, 13, 13, 10, 11, 21, 20, 12, 13, 17, 17, 20, 14, 11, 3, 18, 12, 13, 4, 6, 9],
    [1, 12, 9, 4, 8, 10, 3, 4, 1, 20, 18, 11, 1, 16, 18, 3, 12, 13, 7, 18, 24, 15, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 15, 3, 4, 1, 17, 24, 9, 3, 17, 2, 13, 13, 10, 10, 20, 12, 13, 17, 17, 20, 14, 11, 3, 18, 12, 13, 4, 6, 9],
    [5, 15, 3, 4, 1, 17, 24, 9, 3, 17, 2, 13, 13, 10, 11, 20, 12, 13, 17, 17, 20, 14, 11, 3, 18, 12, 13, 4, 6, 9],
]

sequences_array = np.array(sequences_numeric, dtype=np.int32)


def test_prosit_sequences_determine_type():
    """Test PROSITsequences.determine_type."""
    # character
    seq = ["AAAAAAAA", "AAAAAAAA", "AAAAAAAA", "AAAAAAAA"]
    ret = prpc.PROSITsequences.determine_type(seq)
    assert ret == "character"

    # numeric
    seq = [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]]
    ret = prpc.PROSITsequences.determine_type(seq)
    assert ret == "numeric"

    # array
    seq = np.array(
        [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]], dtype=np.int32
    )
    ret = prpc.PROSITsequences.determine_type(seq)
    assert ret == "array"


def test_prosit_sequences_init():
    """Test instanciation of PROSITsequences."""
    # character
    inst = prpc.PROSITsequences(sequences_character)
    assert inst.character == sequences_character

    # numeric
    inst = prpc.PROSITsequences(sequences_numeric)
    assert inst.numeric == sequences_numeric

    # array
    inst = prpc.PROSITsequences(sequences_array)
    assert np.array_equal(inst.array, sequences_array)


def test_prosit_sequences_prepare_sequences():
    """Test PROSITsequences and prepare_sequences."""
    inst = prpc.PROSITsequences(sequences_character)
    inst.prepare_sequences(flag_disable_progress_bar=False)

    assert inst.character == sequences_character
    assert inst.numeric is None
    assert np.array_equal(inst.array, sequences_array)

    inst = prpc.PROSITsequences(sequences_numeric)
    inst.prepare_sequences(flag_disable_progress_bar=False)

    assert inst.character is None
    assert inst.numeric == sequences_numeric
    assert np.array_equal(inst.array, sequences_array)


def test_prosit_sequence_filtermode():
    """Test PROSITsequence with filtermode."""
    inst = prpc.PROSITsequences(["TESTING", "TESTING94", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "TESTING"])
    inst.prepare_sequences(flag_disable_progress_bar=True, filter=True)
    inst.calculate_lengths()

    arr = np.array(
        [
            [17, 4, 16, 17, 8, 12, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [17, 4, 16, 17, 8, 12, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    arr_len = np.array([7, 0, 0, 7])

    assert np.array_equal(inst.array, arr)
    assert np.array_equal(inst.lengths, arr_len)
