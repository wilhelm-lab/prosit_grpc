from prosit_grpc.predictPROSIT_2 import PredictPROSIT

import h5py
import csv
import numpy as np

with h5py.File("data.hdf5", 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    # a_group_key = list(f.keys())[0]

    # Get the data
    # data = list(f[a_group_key])
    data = list(f["intensities_pred"])

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
                          model_name="intensity"
                          )
pred = predictor.get_predictions()

def test_grpc_call():
    assert type(pred) == np.ndarray

def test_array_size():
    # test number of predictions made
    assert len(pred) == 27
    # test number of returned intensities
    for i in range(len(pred)):
        assert len(pred[i]) == 174

def test_predictions():
    for i in range(len(pred)):
        print(sequences[i])
        print(pred[i])
        print(data[i])
        # for y in range(len(data[i])):
        #     assert data[i][y] == pred[i][y]
    assert 1 == 2