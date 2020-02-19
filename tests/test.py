from prosit_grpc.predictPROSIT import PredictPROSIT

import h5py
import csv
import numpy as np

with h5py.File("data.hdf5", 'r') as f:
    intensities = list(f["intensities_pred"])
    irt = list(f["iRT"])
    irt = [i[0] for i in irt]
    masses = list(f["masses_pred"])


with open("input_test.csv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    header = next(reader, None)

    sequences = []
    ce = []
    charge = []

    for line in reader:
        sequences.append(line[0])
        ce.append(int(line[1]))
        charge.append(int(line[2]))

predictor = PredictPROSIT(server="131.159.152.7:8500",
                          sequences_list=sequences,
                          charges_list=charge,
                          collision_energies_list=ce,
                          model_name="intensity_prosit_publication"
                          )
predictor.predict()
pred = predictor.predictions

def test_grpc_call():
    assert type(pred) == np.ndarray

def test_array_size():
    assert len(pred) == 120
    for i in range(len(pred)):
        assert len(pred[i]) == 174

def test_intensity_prediction():
    predictor = PredictPROSIT(server="131.159.152.7:8500",
                              sequences_list=sequences,
                              charges_list=charge,
                              collision_energies_list=ce,
                              model_name="intensity_prosit_publication"
                              )
    predictor.predict()
    pred = predictor.predictions
    mymasses = predictor.fragment_masses

    assert len(intensities) == len(pred)

    for i in range(len(pred)):
        pearson_correlation = np.corrcoef(intensities[i], pred[i])[0,1]
        print(pearson_correlation)
        assert round(pearson_correlation, 11) == 1

    for i in range(len(pred)):
        print(masses[i])
        print(mymasses[i])
        print("*"*10)

        pearson_correlation = np.corrcoef(masses[i], mymasses[i])[0,1]
        assert round(pearson_correlation, 15) == 1


def test_proteotypicity_prediction():
    predictor = PredictPROSIT(server="131.159.152.7:8500",
                              sequences_list= ["THDLGKW", "VLQKQFFYCTMEKWNGRT", "QMQCNWNVMQGAPSMTCEHRVEYSMEWIID"],
                              model_name="proteotypicity"
                              )
    predictor.predict()
    pred = predictor.predictions
    target = [-0.762, -6.674, -4.055]
    assert len(pred) == len(target)
    for i in range(3):
        assert round(pred[i], 3) == target[i]

def test_batched_prediction():
    predictor = PredictPROSIT(server="131.159.152.7:8500",
                              sequences_list=["THDLGKW" for i in range(10000)],
                              charges_list= [2 for i in range(10000)],
                              collision_energies_list= [20 for i in range(10000)],
                              model_name="intensity_prosit_publication"
                              )
    predictor.predict()
    pred = predictor.get_predictions()
    assert len(pred) == 10000

    predictor = PredictPROSIT(server="131.159.152.7:8500",
                              sequences_list=["THDLGKW" for i in range(10000)],
                              model_name="proteotypicity"
                              )
    predictor.predict()
    pred = predictor.predictions
    assert len(pred) == 10000

    predictor = PredictPROSIT(server="131.159.152.7:8500",
                              sequences_list=["THDLGKW" for i in range(10000)],
                              model_name="iRT"
                              )
    predictor.predict()
    pred = predictor.predictions
    assert len(pred) == 10000

def test_irt_prediction():
    predictor = PredictPROSIT(server="131.159.152.7:8500",
                              sequences_list=sequences,
                              model_name="iRT"
                              )
    predictor.predict()
    pred = predictor.predictions

    assert len(pred) == len(irt)

    for i in range(len(irt)):
        # converting them to float because:
        # the prediction returns numpy float 64 while the hdf5 from the website has numpy float 32
        assert round(float(pred[i]), 1) == round(float(irt[i]), 1)

def test_get_functions():
    predictor = PredictPROSIT(server="131.159.152.7:8500",
                              sequences_list=sequences,
                              charges_list=charge,
                              collision_energies_list=ce,
                              model_name="intensity_prosit_publication"
                              )
    pred = predictor.get_predictions()
    masses = predictor.get_fragment_masses()
    annotations = predictor.get_fragment_annotation()

    # assert that the number of masses matches the number of predictions
    assert len(pred) == len(masses)

    # assert that the number of annotations matches the number of predictions
    assert len(pred) == len(annotations)

    # assert that the number of number of peaks for each spectrum matches in intensities and masses
    for i in range(len(pred)):
        assert len(pred[i]) == len(masses[i])

    # assert that the number of number of peaks for each spectrum matches in intensities and annotations
    for i in range(len(pred)):
        for annotation_type in annotations[i].values():
            assert len(annotation_type) == len(pred[i])

    # assert that the lowest intensity is at least 0
    assert min((min(pred))) >= 0 # min of min because pred is a nested list

# def test_hdf5_input_output():
#     with h5py.File("data.hdf5", 'r') as f:
#
#         print(f.keys())
#         predictor = PredictPROSIT(server="131.159.152.7:8500",
#                                   model_name="intensity_prosit_publication")
#
#         predictor.set_sequence_list_numeric(numeric_sequence_list=list(f["sequence_integer"]))
#         predictor.set_charges_list_one_hot(list(f["precursor_charge_onehot"]))
#         predictor.set_collision_energy_normed(list(f["collision_energy_aligned_normed"]))
#         predictor.predict()
#
#         output_dict={
#             "sequence_integer": f["sequence_integer"],
#             "precursor_charge_onehot": f["precursor_charge_onehot"],
#             "collision_energy_aligned_normed": f["collision_energy_aligned_normed"],
#             'intensities_pred': np.array(predictor.predictions).astype(np.float64),
#             'masses_pred': np.array(predictor.fragment_masses).astype(np.float64)}
#
#
#
#         predictor.set_model_name(model_name="iRT")
#         predictor.predict()
#         output_dict["iRT"] = np.array([np.array(el).astype(np.float32) for el in predictor.predictions]).astype(np.float32)
#
#         output_dict["iRT"].shape = (120,1)
#
#         with h5py.File("output.hdf5", "w") as data_file:
#             for key, data in output_dict.items():
#                 data_file.create_dataset(key, data=data, dtype=data.dtype, compression="gzip")
#
#         # pred_web = list(f["intensities_pred"])
#         # pred_grpc = output_dict["intensities_pred"]
#         # # assert len(pred_web) == len(pred_grpc)
#         # # for i in range(len(pred_web)):
#         # #     assert len(pred_web[i]) == len(pred_grpc[i])
#         # print(np.corrcoef(pred_web[i], pred_grpc[i])[0, 1])
#         #
#         # for j in range(len(pred_web[i])):
#         #     assert pred_web[i][j] == pred_grpc[i][j]
#
#
#
#     with h5py.File("data.hdf5", 'r') as input_file:
#         with h5py.File("output.hdf5", 'r') as output_file:
#
#             assert input_file.keys() == output_file.keys()
#
#             print(input_file["iRT"].shape)
#             print(output_file["iRT"].shape)
#
#             for key in input_file.keys():
#                 print(key)
#                 print(list(input_file[key]))
#                 print(list(output_file[key]))
#
#                 assert list(input_file[key]) == list(output_file[key])
