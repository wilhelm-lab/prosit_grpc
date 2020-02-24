import prosit_grpc.predictPROSIT as prpc


import h5py
import csv
import numpy as np

# constants
test_server = "proteomicsdb.org:8500"
ca_cert = "../cert/Proteomicsdb-Prosit.crt"
cert = "../cert/ci-pipeline.crt"
key = "../cert/ci-pipeline.key"

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



def test_prediction():
    predictor = prpc.PROSITpredictor(server=test_server,
                              path_to_ca_certificate=ca_cert,
                              path_to_certificate=cert,
                              path_to_key_certificate=key,
                              )

    x = predictor.predict(sequences=sequences,
                      charges=charge,
                      collision_energies=ce,
                      intensity_model="intensity_prosit_publication",
                      irt_model="iRT",
                      proteotypicity_model="proteotypicity",)

def test_temp():
    charges_onehot = [[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]]
    charges_array = np.array(charges_onehot, dtype=np.int32)


    import prosit_grpc.outputPROSIT as opr

    temp = opr.PROSITspectrum()
    temp.create_masking(sequences_lengths=[7, 10, 15, 30, 13, 26],
                        charges_array=charges_array)

    print(temp.mask)

