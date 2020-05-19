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

test_int_model = "Prosit_2019_intensity"
test_irt_model = "Prosit_2019_irt"
test_prot_model = "Prosit_2020_proteotypicity"
test_charge_model = "Prosit_2020_charge"

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


def test_bundled_prediction():
    predictor = prpc.PROSITpredictor(server=test_server,
                                     path_to_ca_certificate=ca_cert,
                                     path_to_certificate=cert,
                                     path_to_key_certificate=key,
                                     )

    output_dict = predictor.predict(sequences=sequences,
                                    charges=charge,
                                    collision_energies=ce,
                                    models=[test_int_model, test_irt_model, test_prot_model]
                                    )

    # test spectrum prediction
    my_int = output_dict[test_int_model]["normalized"]["intensity"]
    my_masses = output_dict[test_int_model]["masked"]["fragmentmz"]
    assert len(intensities) == len(my_int)
    assert len(masses) == len(my_masses)
    assert len(my_int) == len(my_masses)
    for i in range(len(my_int)):
        pearson_correlation_int = np.corrcoef(intensities[i], my_int[i])[0, 1]
        assert round(pearson_correlation_int, 11) == 1

        pearson_correlation_masses = np.corrcoef(masses[1], my_masses[1])[0, 1]
        assert round(pearson_correlation_masses, 15) == 1

    # test irt prediction
    my_irt = output_dict[test_irt_model]["normalized"]
    assert len(my_irt) == len(irt)
    for i in range(len(irt)):
        # converting them to float because:
        # the prediction returns numpy float 64 while the hdf5 from the website has numpy float 32
        assert round(float(my_irt[i]), 3) == round(float(irt[i]), 3)


def test_intensity_prediction():
    predictor = prpc.PROSITpredictor(server=test_server,
                                     path_to_ca_certificate=ca_cert,
                                     path_to_certificate=cert,
                                     path_to_key_certificate=key,
                                     )

    dict_intensity = predictor.predict(sequences=sequences,
                                       charges=charge,
                                       collision_energies=ce,
                                       models=[test_int_model])

    assert len(dict_intensity[test_int_model]) == 4

    # test spectrum prediction
    my_int = dict_intensity[test_int_model]["normalized"]["intensity"]
    my_masses = dict_intensity[test_int_model]["masked"]["fragmentmz"]
    assert len(intensities) == len(my_int)
    assert len(masses) == len(my_masses)
    assert len(my_int) == len(my_masses)
    for i in range(len(my_int)):
        pearson_correlation_int = np.corrcoef(intensities[i], my_int[i])[0, 1]
        assert round(pearson_correlation_int, 11) == 1

        pearson_correlation_masses = np.corrcoef(masses[1], my_masses[1])[0, 1]
        assert round(pearson_correlation_masses, 15) == 1


def test_irt_prediction():
    predictor = prpc.PROSITpredictor(server=test_server,
                                     path_to_ca_certificate=ca_cert,
                                     path_to_certificate=cert,
                                     path_to_key_certificate=key,
                                     )

    dict_irt = predictor.predict(sequences=sequences,
                                 models=[test_irt_model])

    assert len(dict_irt) == 1

    # test irt prediction
    my_irt = dict_irt[test_irt_model]["normalized"]
    assert len(my_irt) == len(irt)
    for i in range(len(irt)):
        # converting them to float because:
        # the prediction returns numpy float 64 while the hdf5 from the website has numpy float 32
        assert round(float(my_irt[i]), 3) == round(float(irt[i]), 3)

    dict_proteotyp = predictor.predict(sequences=sequences,
                                       models=[test_prot_model])
    assert len(dict_proteotyp) == 1

def test_charge_prediction():
    predictor = prpc.PROSITpredictor(server=test_server,
                                     path_to_ca_certificate=ca_cert,
                                     path_to_certificate=cert,
                                     path_to_key_certificate=key,
                                     )
    dict_out = predictor.predict(sequences=sequences,
                                       charges=charge,
                                       collision_energies=ce,
                                       models=[test_charge_model])
    assert dict_out[test_charge_model].shape == (120, 6)

def test_batching():
    l = 100

    predictor = prpc.PROSITpredictor(server=test_server,
                                     path_to_ca_certificate=ca_cert,
                                     path_to_certificate=cert,
                                     path_to_key_certificate=key
                                     )

    output_dict = predictor.predict(sequences=[x for x in sequences for _ in range(l)],
                                    charges=[x for x in charge for _ in range(l)],
                                    collision_energies=[x for x in ce for _ in range(l)],
                                    models=[test_int_model, test_irt_model, test_prot_model, test_charge_model])

    assert len(output_dict[test_int_model]["raw"]["intensity"]) == l*len(sequences)
    assert len(output_dict[test_irt_model]["raw"]) == l*len(sequences)
    assert len(output_dict[test_prot_model]) == l*len(sequences)
    assert len(output_dict[test_charge_model]) == l*len(sequences)

    # ensure the predictions are not shuffeled
    truth_irt = [x for x in irt for _ in range(l)]
    pred_irt = output_dict[test_irt_model]["normalized"]
    assert len(truth_irt) == len(pred_irt)
    for x, y in zip(truth_irt, pred_irt):
        # converting them to float because:
        # the prediction returns numpy float 64 while the hdf5 from the website has numpy float 32
        assert round(float(x), 3) == round(float(y), 3)

def test_predict_to_hdf5():
    predictor = prpc.PROSITpredictor(server=test_server,
                                     path_to_ca_certificate=ca_cert,
                                     path_to_certificate=cert,
                                     path_to_key_certificate=key,
                                     )
    predictor.predict_to_hdf5(sequences=sequences,
                              charges=charge,
                              collision_energies=ce,
                              intensity_model=test_int_model,
                              irt_model=test_irt_model,
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
                                  models=[test_int_model, test_irt_model, test_prot_model],
                                  matrix_expansion_param=[
                                      {'AA_to_permutate': 'M', 'into': 'M(ox)', 'max_in_parallel': 2}]
                                  )

    assert len(pred_dict[test_int_model]) == 4


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
                                  models=[test_irt_model],
                                  matrix_expansion_param=mexp
                                  )

    for model_dict in pred_dict.values():
        assert len(model_dict["raw"]) == 16
