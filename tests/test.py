from prosit_grpc.predictPROSIT import PredictPROSIT

import h5py
import csv
import numpy as np

with h5py.File("data.hdf5", 'r') as f:
    intensities = list(f["intensities_pred"])
    irt = list(f["iRT"])
    irt = [i[0] for i in irt]

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
pred = predictor.get_predictions()

def test_grpc_call():
    assert type(pred) == np.ndarray

def test_array_size():
    assert len(pred) == 27
    for i in range(len(pred)):
        assert len(pred[i]) == 174

def test_sequence_alpha_to_numbers():
    result=predictor.sequence_alpha_to_numbers("ACDEFGHIKLMNPQRSTVWYUO")
    target= [x+1 for x in range(22)]
    assert result == target

def test_intensity():
    predictor = PredictPROSIT(server="131.159.152.7:8500",
                              sequences_list=sequences,
                              charges_list=charge,
                              collision_energies_list=ce,
                              model_name="intensity_prosit_publication"
                              )
    pred = predictor.get_predictions()

    assert len(intensities) == len(pred)
    for i in range(len(pred)):
        pearson_correlation = np.corrcoef(intensities[i], pred[i])[0,1]
        assert round(pearson_correlation, 12) == 1


def test_proteotypicity():
    predictor = PredictPROSIT(server="131.159.152.7:8500",
                              sequences_list= ["THDLGKW", "VLQKQFFYCTMEKWNGRT", "QMQCNWNVMQGAPSMTCEHRVEYSMEWIID"],
                              model_name="proteotypicity"
                              )
    pred = predictor.get_predictions()
    target = [-0.762, -6.674, -4.055]

    print(pred)

    assert len(pred) == len(target)
    for i in range(3):
        assert round(pred[i], 3) == target[i]

def test_batching():
    predictor = PredictPROSIT(server="131.159.152.7:8500",
                              sequences_list=["THDLGKW" for i in range(10000)],
                              charges_list= [2 for i in range(10000)],
                              collision_energies_list= [20 for i in range(10000)],
                              model_name="intensity_prosit_publication"
                              )
    pred = predictor.get_predictions()
    assert len(pred) == 10000

    predictor = PredictPROSIT(server="131.159.152.7:8500",
                              sequences_list=["THDLGKW" for i in range(10000)],
                              model_name="proteotypicity"
                              )
    pred = predictor.get_predictions()
    assert len(pred) == 10000

    predictor = PredictPROSIT(server="131.159.152.7:8500",
                              sequences_list=["THDLGKW" for i in range(10000)],
                              model_name="iRT"
                              )
    pred = predictor.get_predictions()
    assert len(pred) == 10000

def test_irt():
    predictor = PredictPROSIT(server="131.159.152.7:8500",
                              sequences_list=sequences,
                              model_name="iRT"
                              )
    pred = predictor.get_predictions()

    assert len(pred) == len(irt)

    for i in range(len(irt)):
        # converting them to float because:
        # the prediction returns numpy float 64 while the hdf5 from the website has numpy float 32
        assert round(float(pred[i]), 1) == round(float(irt[i]), 1)