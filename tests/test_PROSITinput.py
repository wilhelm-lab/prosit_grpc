import numpy as np
import prosit_grpc.inputPROSIT as prpc
import pytest

sequences_character = ["KKVCQGTSNKLTQLGTFE",
                       "GNMYYENSYALAVLSNYDANKTGLK",
                       "LTQLGTFEDHFLSLQR",
                       "FRDEATCKDTCPPLMLYNPTTQMDVNPEGK",
                       "RDEATCKDTCPPLM(ox)LNPTTYQMDVNPEGK",
                       "FRDEATKDTCPPLMM(O)YNPTTYQMDVNPEGK",
                       "ANKEILDEAYVMASVDNPHVCR",
                       "FRDEATCKDT(Cam)PPLLYNPTTYQMDVNPEGK",
                       "FRDEATCKDTCacPPLMYNPTTYQMDVNPEGK"]
ce_numeric = [10, 20, 30, 40, 50, 10, 20, 30, 40]
charges_numeric = [1, 2, 3, 4, 5, 6, 1, 2, 3]


def test_PROSITinput_init():
    inst = prpc.PROSITinput(sequences=sequences_character,
                            charges=charges_numeric,
                            collision_energies=ce_numeric)

    assert inst.sequences.character == sequences_character
    assert inst.charges.numeric == charges_numeric
    assert inst.collision_energies.numeric == ce_numeric

    # with pytest.raises(Exception):
    #     inst = prpc.PROSITinput(sequences=sequences_character[:8],
    #                             charges=ce_numeric,
    #                             collision_energies=charges_numeric)


def test_PROSITinput_prepare_input():
    inst = prpc.PROSITinput(sequences=sequences_character,
                            charges=charges_numeric,
                            collision_energies=ce_numeric)

    inst.prepare_input(flag_disable_progress_bar = False)

    assert type(inst.sequences.array) == np.ndarray
    assert type(inst.charges.array) == np.ndarray
    assert type(inst.collision_energies.array) == np.ndarray


def test_expand_matrices():
    inst = prpc.PROSITinput(sequences=[[1, 1, 1, 1], [1, 1, 2, 1], [1, 2, 2, 1], [2, 2, 2, 1], [2, 2, 2, 2]],
                            charges=[1, 2, 3, 4, 5],
                            collision_energies=[10, 20, 30, 40, 50])

    inst.prepare_input(flag_disable_progress_bar=False)
    inst.expand_matrices(
        param={'AA_to_permutate': 'C', 'into': 'M(ox)', 'max_in_parallel': 2})

    assert len(inst.sequences.array) == len(inst.charges.array)
    assert len(inst.sequences.array) == len(inst.collision_energies.array)

    assert np.array_equal(inst.sequences.array, [[1,  1,  1,  1],
                                                 [1,  1,  2,  1],
                                                 [1,  2,  2,  1],
                                                 [2,  2,  2,  1],
                                                 [2,  2,  2,  2],
                                                 [1,  1, 21,  1],
                                                 [1, 21,  2,  1],
                                                 [1,  2, 21,  1],
                                                 [1, 21, 21,  1],
                                                 [21,  2,  2,  1],
                                                 [2, 21,  2,  1],
                                                 [2,  2, 21,  1],
                                                 [21, 21,  2,  1],
                                                 [21,  2, 21,  1],
                                                 [2, 21, 21,  1],
                                                 [21,  2,  2,  2],
                                                 [2, 21,  2,  2],
                                                 [2,  2, 21,  2],
                                                 [2,  2,  2, 21],
                                                 [21, 21,  2,  2],
                                                 [21,  2, 21,  2],
                                                 [21,  2,  2, 21],
                                                 [2, 21, 21,  2],
                                                 [2, 21,  2, 21],
                                                 [2,  2, 21, 21]])
    # copies created = [0, 1, 3, 6, 10]
    assert np.array_equal(inst.charges.array, [[1, 0, 0, 0, 0, 0],
                                               [0, 1, 0, 0, 0, 0],
                                               [0, 0, 1, 0, 0, 0],
                                               [0, 0, 0, 1, 0, 0],
                                               [0, 0, 0, 0, 1, 0],
                                               [0, 1, 0, 0, 0, 0],
                                               [0, 0, 1, 0, 0, 0],
                                               [0, 0, 1, 0, 0, 0],
                                               [0, 0, 1, 0, 0, 0],
                                               [0, 0, 0, 1, 0, 0],
                                               [0, 0, 0, 1, 0, 0],
                                               [0, 0, 0, 1, 0, 0],
                                               [0, 0, 0, 1, 0, 0],
                                               [0, 0, 0, 1, 0, 0],
                                               [0, 0, 0, 1, 0, 0],
                                               [0, 0, 0, 0, 1, 0],
                                               [0, 0, 0, 0, 1, 0],
                                               [0, 0, 0, 0, 1, 0],
                                               [0, 0, 0, 0, 1, 0],
                                               [0, 0, 0, 0, 1, 0],
                                               [0, 0, 0, 0, 1, 0],
                                               [0, 0, 0, 0, 1, 0],
                                               [0, 0, 0, 0, 1, 0],
                                               [0, 0, 0, 0, 1, 0],
                                               [0, 0, 0, 0, 1, 0]])

    collision_energies_arry = np.round(inst.collision_energies.array, 1)
    truth = np.array([0.1, 0.2, 0.3, 0.4, 0.5,
                      0.2,
                      0.3, 0.3, 0.3,
                      0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
                      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                     dtype=np.float32)
    assert np.array_equal(collision_energies_arry, truth)