import numpy as np
import prosit_grpc.predictPROSIT as prpc

def test_PROSITcollisionenergies_determine_type:

    # numeric
    ce = [10, 20, 30, 40, 50]
    prpc.PROSITcollisionenergies.determine_type(ce)


    # procentual
    ce = [0.10, 0.20, 0.30, 0.40, 0.50]
    prpc.PROSITcollisionenergies.determine_type(ce)


    # array
    ce = [0.10, 0.20, 0.30, 0.40, 0.50]
    prpc.PROSITcollisionenergies.determine_type(ce)

