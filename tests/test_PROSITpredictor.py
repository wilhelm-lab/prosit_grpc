import prosit_grpc.predictPROSIT as prpc

import h5py
import csv
import numpy as np
import os

# constants
test_server = "131.159.152.7:8500"
ca_cert = "cert/Proteomicsdb-Prosit.crt"
cert = "cert/ci-pipeline.crt"
key = "cert/ci-pipeline.key"

test_int_model = "Prosit_2019_intensity"
test_irt_model = "Prosit_2019_irt"
test_prot_model = "Prosit_2020_proteotypicity"
test_charge_model = "Prosit_2020_charge"

with h5py.File("tests/predict_to_hdf5.hdf5", 'r') as f:
    irt = list(f["iRT"])
    irt = [i[0] for i in irt]

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

    assert len(output_dict[test_int_model]["intensity"]) == l*len(sequences)
    assert len(output_dict[test_irt_model]) == l*len(sequences)
    assert len(output_dict[test_prot_model]) == l*len(sequences)
    assert len(output_dict[test_charge_model]) == l*len(sequences)

    # ensure the predictions are not shuffeled
    truth_irt = [x for x in irt for _ in range(l)]
    pred_irt = output_dict[test_irt_model]
    assert len(truth_irt) == len(pred_irt)
    for x, y in zip(truth_irt, pred_irt):
        # converting them to float because:
        # the prediction returns numpy float 64 while the hdf5 from the website has numpy float 32
        assert round(float(x), 3) == round(float(y), 3)

def test_predict_to_hdf5():

    pred_hdf5 = "tests/pred.hdf5"

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
                              proteotypicicty_model=test_prot_model,
                              path_hdf5=pred_hdf5)

    with h5py.File("tests/predict_to_hdf5.hdf5", 'r') as truth:
        with h5py.File(pred_hdf5, 'r') as pred:
            for i in ["intensities_pred", "iRT", "masses_pred", "collision_energy_aligned_normed", "precursor_charge_onehot", "sequence_integer", "proteotypicity"]:
                assert np.array(truth[i]).shape == np.array(pred[i]).shape
    os.remove(pred_hdf5)


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
                                      {'AA_to_permutate': 'M', 'into': 'M(U:35)', 'max_in_parallel': 2}]
                                  )

    assert len(pred_dict[test_int_model]) == 3


def test_predict_with_repeated_matrix_expansion():

    predictor = prpc.PROSITpredictor(server=test_server,
                                     path_to_ca_certificate=ca_cert,
                                     path_to_certificate=cert,
                                     path_to_key_certificate=key,
                                     )

    mexp = [{'AA_to_permutate': 'C', 'into': 'M(U:35)', 'max_in_parallel': 1},
            {'AA_to_permutate': 'A', 'into': 'S', 'max_in_parallel': 1},
            {'AA_to_permutate': 'D', 'into': 'V', 'max_in_parallel': 1},
            {'AA_to_permutate': 'E', 'into': 'K', 'max_in_parallel': 1}]

    pred_dict = predictor.predict(sequences=["ACDEFGH"],
                                  charges=[2],
                                  collision_energies=[20],
                                  models=[test_irt_model, test_prot_model, test_charge_model],
                                  matrix_expansion_param=mexp
                                  )

    for model_dict in pred_dict.values():
        assert len(model_dict) == 16


# def test_prediction_consistency():
#     predictor = prpc.PROSITpredictor(server=test_server,
#                                      path_to_ca_certificate=ca_cert,
#                                      path_to_certificate=cert,
#                                      path_to_key_certificate=key,
#                                      )
#
#     output_dict = predictor.predict(sequences=sequences,
#                                     charges=charge,
#                                     collision_energies=ce,
#                                     models=["Prosit_2019_intensity",
#                                             "Prosit_2019_irt",
#                                             "Prosit_2019_irt_supplement",
#                                             "Prosit_2020_charge",
#                                             "Prosit_2020_intensity_preview",
#                                             "Prosit_2020_proteotypicity"])
#
#     intensity_models = ["Prosit_2019_intensity",
#                         "Prosit_2020_intensity_preview"]
#
#     irt_models = ["Prosit_2019_irt",
#                   "Prosit_2019_irt_supplement"]
#
#     proteotypicity_models = ["Prosit_2020_proteotypicity"]
#
#     charge_models = ["Prosit_2020_charge"]
#
#     with h5py.File("tests/prediction_consistency.hdf5", 'r') as f:
#         for model in intensity_models:
#             print(output_dict.keys())
#             print("Testing model consistency:", model)
#             nrow = len(output_dict[model]["intensity"])
#
#             for rowid in range(nrow):
#                 pred = np.round_(output_dict[model]["intensity"][rowid], 10)
#                 true = np.round_(np.array(f[model]["intensity"][rowid]), 10)
#                 assert 1 == round(np.corrcoef(pred, true)[0][1], 10)
#                 pred = np.round_(output_dict[model]["fragmentmz"][rowid], 16)
#                 true = np.round_(np.array(f[model]["fragmentmz"][rowid]), 16)
#                 assert 1 == round(np.corrcoef(pred, true)[0][1], 16)
#
#             assert np.array_equal(output_dict[model]["annotation"]["charge"], np.array(f[model]["annotation"]["charge"]))
#             assert np.array_equal(output_dict[model]["annotation"]["number"], np.array(f[model]["annotation"]["number"]))
#             tmp = np.array(f[model]["annotation"]["type"])
#             tmp.dtype = np.dtype("U1")
#             assert np.array_equal(output_dict[model]["annotation"]["type"], tmp)
#
#         for model in irt_models:
#             print("Testing model consistency:", model)
#             pred = np.round_(output_dict[model], 30)
#             true = np.round_(np.array(f[model]), 30)
#             assert np.array_equal(pred, true)
#
#         for model in proteotypicity_models:
#             print("Testing model consistency:", model)
#             pred = np.around(output_dict[model], 30)
#             true = np.around(np.array(f[model]), 30)
#             assert np.array_equal(pred, true)
#
#         for model in charge_models:
#             print("Testing model consistency:", model)
#             pred = np.round_(output_dict[model], 30)
#             true = np.round_(np.array(f[model]), 30)
#             assert np.array_equal(pred, true)
