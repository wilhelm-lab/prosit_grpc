import h5py
import csv
from prosit_grpc.predictPROSIT_2 import PredictPROSIT

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

    seq = []
    ce = []
    charge = []

    for line in reader:
        seq.append(line[0])
        ce.append(int(line[1]))
        charge.append(int(line[2]))


def test_pytest():
    assert 1 == 1


# def test_raw_predictions():
predictor = PredictPROSIT(server="131.159.152.7:8500",
                      sequences_list=seq,
                      charges_list=charge,
                      collision_energies_list=ce,
                      model_name="intensity"
                      )

pred = predictor.get_predictions()

print("="*100)
print(pred[1])
print("="*100)
print(data[1])
print(max(data[1]))


