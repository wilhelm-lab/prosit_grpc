#!/usr/bin/env python

# In[6]:

import numpy as np

from prosit_grpc.predictPROSIT import PROSITpredictor

np.set_printoptions(threshold=100)
import csv

import h5py

# In[2]:


predictor = PROSITpredictor(
    server="proteomicsdb.org:8500",
    path_to_ca_certificate="../ssl_certificates_grpc/out/Proteomicsdb-Prosit.crt",
    path_to_certificate="../ssl_certificates_grpc/out/llautenbacher.crt",
    path_to_key_certificate="../ssl_certificates_grpc/out/llautenbacher.key",
)


# In[3]:


with open("tests/input_test.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    header = next(reader, None)
    sequences = []
    ce = []
    charge = []
    for line in reader:
        sequences.append(line[0])
        ce.append(int(line[1]))
        charge.append(int(line[2]))


# In[4]:


output_dict = predictor.predict(
    sequences=sequences,
    charges=charge,
    collision_energies=ce,
    models=[
        "Prosit_2019_intensity",
        "Prosit_2019_irt",
        "Prosit_2019_irt_supplement",
        "Prosit_2020_charge",
        "Prosit_2021_intensity_cid",
        "Prosit_2021_intensity_hcd",
        "Prosit_2020_intensity_preview",
        "Prosit_2020_proteotypicity",
    ],
)


# In[5]:


def dict_to_hdf5(dic, hdf5_path, prefix=None):
    """Convert dictionary to hdf5."""
    for key, data in dic.items():
        if prefix is not None:
            dataset_name = "/".join([prefix, key])
        else:
            dataset_name = key

        if type(data) is np.ndarray:
            print("WRITING:\t", dataset_name)
            with h5py.File(hdf5_path, "a") as data_file:
                try:
                    data_file.create_dataset(dataset_name, data=data, compression="gzip")
                except TypeError:
                    data.dtype = np.dtype("S1")
                    data_file.create_dataset(dataset_name, data=data, compression="gzip")

        elif type(data) is dict:
            dict_to_hdf5(data, hdf5_path, prefix=dataset_name)
        else:
            raise ValueError


# In[6]:


dict_to_hdf5(output_dict, "true_state_forever.hdf5")


# In[ ]:


# In[26]:


models = ["A", "B", ""]


# In[27]:


[m for m in models if m != ""]


# In[ ]:
