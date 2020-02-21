import numpy as np
import prosit_grpc.predictPROSIT as prpc
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
ce_numeric = [10,20,30,40,50,10,20,30,40]
charges_numeric = [1,2,3,4,5,6,1,2,3]

def test_PROSITinput_init():
    inst = prpc.PROSITinput(sequences=sequences_character,
                            charges=charges_numeric,
                            collision_energies=ce_numeric)

    assert inst.sequences.character == sequences_character
    assert inst.charges.numeric == charges_numeric
    assert inst.collision_energies.numeric == ce_numeric

    with pytest.raises(Exception):
        inst = prpc.PROSITinput(sequences=sequences_character[:8],
                                charges=ce_numeric,
                                collision_energies=charges_numeric)

def test_PROSITinput_prepare_input():
    inst = prpc.PROSITinput(sequences=sequences_character,
                            charges=charges_numeric,
                            collision_energies=ce_numeric)

    inst.prepare_input()

    assert type(inst.sequences.array) == np.ndarray
    assert type(inst.charges.array) == np.ndarray
    assert type(inst.collision_energies.array) == np.ndarray