import prosit_grpc.predictPROSIT as prpc

import h5py
import csv
import numpy as np
import os

# constants
test_server = "proteomicsdb.org:8500"
ca_cert = "cert/Proteomicsdb-Prosit.crt"
cert = "cert/ci-pipeline.crt"
key = "cert/ci-pipeline.key"

with h5py.File("tests/data.hdf5", 'r') as f:
    intensities = list(f["intensities_pred"])
    irt = list(f["iRT"])
    irt = [i[0] for i in irt]
    masses = list(f["masses_pred"])

with open("tests/input_test.csv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    header = next(reader, None)
    sequences = []
    ce = []
    charge = []
    for line in reader:
        sequences.append(line[0])
        ce.append(int(line[1]))
        charge.append(int(line[2]))


def test_prediction():
    predictor = prpc.PROSITpredictor(server=test_server,
                                     path_to_ca_certificate=ca_cert,
                                     path_to_certificate=cert,
                                     path_to_key_certificate=key,
                                     )

    output_dict = predictor.predict(sequences=sequences,
                                    charges=charge,
                                    collision_energies=ce,
                                    intensity_model="intensity_prosit_publication",
                                    irt_model="iRT",
                                    proteotypicity_model="proteotypicity")

    # test spectrum prediction
    my_int = predictor.output.spectrum.intensity.normalized
    my_masses = predictor.output.spectrum.mz.masked
    assert len(intensities) == len(my_int)
    assert len(masses) == len(my_masses)
    assert len(my_int) == len(my_masses)
    for i in range(len(my_int)):
        pearson_correlation_int = np.corrcoef(intensities[i], my_int[i])[0, 1]
        assert round(pearson_correlation_int, 11) == 1

        pearson_correlation_masses = np.corrcoef(masses[1], my_masses[1])[0, 1]
        assert round(pearson_correlation_masses, 15) == 1

    # test irt prediction
    my_irt = predictor.output.irt.normalized
    assert len(my_irt) == len(irt)
    for i in range(len(irt)):
        # converting them to float because:
        # the prediction returns numpy float 64 while the hdf5 from the website has numpy float 32
        assert round(float(my_irt[i]), 3) == round(float(irt[i]), 3)


def test_seperate_prediction():
    predictor = prpc.PROSITpredictor(server=test_server,
                                     path_to_ca_certificate=ca_cert,
                                     path_to_certificate=cert,
                                     path_to_key_certificate=key,
                                     )

    dict_intensity = predictor.predict(sequences=sequences,
                                       charges=charge,
                                       collision_energies=ce,
                                       intensity_model="intensity_prosit_publication")

    assert len(dict_intensity) == 5

    # test spectrum prediction
    my_int = predictor.output.spectrum.intensity.normalized
    my_masses = predictor.output.spectrum.mz.masked
    assert len(intensities) == len(my_int)
    assert len(masses) == len(my_masses)
    assert len(my_int) == len(my_masses)
    for i in range(len(my_int)):
        pearson_correlation_int = np.corrcoef(intensities[i], my_int[i])[0, 1]
        assert round(pearson_correlation_int, 11) == 1

        pearson_correlation_masses = np.corrcoef(masses[1], my_masses[1])[0, 1]
        assert round(pearson_correlation_masses, 15) == 1

    dict_irt = predictor.predict(sequences=sequences,
                                 irt_model="iRT")

    assert len(dict_irt) == 1

    # test irt prediction
    my_irt = predictor.output.irt.normalized
    assert len(my_irt) == len(irt)
    for i in range(len(irt)):
        # converting them to float because:
        # the prediction returns numpy float 64 while the hdf5 from the website has numpy float 32
        assert round(float(my_irt[i]), 3) == round(float(irt[i]), 3)

    dict_proteotyp = predictor.predict(sequences=sequences,
                                       proteotypicity_model="proteotypicity")
    assert len(dict_proteotyp) == 1


def test_batching():
    predictor = prpc.PROSITpredictor(server=test_server,
                                     path_to_ca_certificate=ca_cert,
                                     path_to_certificate=cert,
                                     path_to_key_certificate=key,
                                     )

    output_dict = predictor.predict(sequences=[sequences[1] for _ in range(10000)],
                                    charges=[charge[1] for _ in range(10000)],
                                    collision_energies=[ce[1]
                                                        for _ in range(10000)],
                                    intensity_model="intensity_prosit_publication",
                                    irt_model="iRT",
                                    proteotypicity_model="proteotypicity")

    for output in output_dict.values():
        assert len(output) == 10000


def test_predict_to_hdf5():
    predictor = prpc.PROSITpredictor(server=test_server,
                                     path_to_ca_certificate=ca_cert,
                                     path_to_certificate=cert,
                                     path_to_key_certificate=key,
                                     )
    predictor.predict_to_hdf5(sequences=sequences,
                              charges=charge,
                              collision_energies=ce,
                              intensity_model="intensity_prosit_publication",
                              irt_model="iRT",
                              path_hdf5="tests/output.hdf5")

    os.remove("tests/output.hdf5")


def test_predict_with_matrix_expansion():

    predictor = prpc.PROSITpredictor(server=test_server,
                                     path_to_ca_certificate=ca_cert,
                                     path_to_certificate=cert,
                                     path_to_key_certificate=key,
                                     )

    pred_dict = predictor.predict(sequences=["AAAAMK", "AAAAMK"],
                                  charges=[2, 3],
                                  collision_energies=[20, 30],
                                  intensity_model="intensity_prosit_publication",
                                  irt_model="iRT",
                                  proteotypicity_model="proteotypicity",
                                  matrix_expansion_param=[
                                      {'AA_to_permutate': 'M', 'into': 'M(ox)', 'max_in_parallel': 2}]
                                  )

    assert len(pred_dict["intensity"]) == 4


def test_predict_with_repeated_matrix_expansion():

    predictor = prpc.PROSITpredictor(server=test_server,
                                     path_to_ca_certificate=ca_cert,
                                     path_to_certificate=cert,
                                     path_to_key_certificate=key,
                                     )

    mexp = [{'AA_to_permutate': 'C', 'into': 'M(ox)', 'max_in_parallel': 1},
            {'AA_to_permutate': 'A', 'into': 'PhS', 'max_in_parallel': 1},
            {'AA_to_permutate': 'D', 'into': 'PhT', 'max_in_parallel': 1},
            {'AA_to_permutate': 'E', 'into': 'PhY', 'max_in_parallel': 1}]

    pred_dict = predictor.predict(sequences=["ACDEFGH"],
                                  charges=[2],
                                  collision_energies=[20],
                                  intensity_model="intensity_prosit_publication",
                                  irt_model="iRT",
                                  proteotypicity_model="proteotypicity",
                                  matrix_expansion_param=mexp
                                  )

    assert len(pred_dict["intensity"]) == 16
