# Status of code
## Master
[![pipeline status](https://gitlab.lrz.de/proteomics/prosit_tools/prosit_grpc/badges/master/pipeline.svg)](https://gitlab.lrz.de/proteomics/prosit_tools/prosit_grpc/commits/master)
[![coverage report](https://gitlab.lrz.de/proteomics/prosit_tools/prosit_grpc/badges/master/coverage.svg)](https://gitlab.lrz.de/proteomics/prosit_tools/prosit_grpc/commits/master)
## Develop
[![pipeline status](https://gitlab.lrz.de/proteomics/prosit_tools/prosit_grpc/badges/develop/pipeline.svg)](https://gitlab.lrz.de/proteomics/prosit_tools/prosit_grpc/commits/develop)
[![coverage report](https://gitlab.lrz.de/proteomics/prosit_tools/prosit_grpc/badges/develop/coverage.svg)](https://gitlab.lrz.de/proteomics/prosit_tools/prosit_grpc/commits/develop)

# Prosit gRPC Client as a poetry package

## 1. Simple installation by pip
We provide **tagged** versions of our package pointing to the latest tested master releases but can give you also direct access to new features if requested. The correctness is indicated by the green=good or red=wrong flags above. 
```bash
pip install -e git+https://gitlab.lrz.de/proteomics/prosit_tools/prosit_grpc.git@v1.2.0#egg=prosit_grpc
```
### If you deployed an ssh key
```bash
pip install -e git+git@gitlab.lrz.de:proteomics/prosit_tools/prosit_grpc.git@v1.2.0#egg=prosit_grpc
```
## 2. Ask Ludwig for certificates
You are using a special access to our GPUs and are therefore required to identify against our server. Ludwig Lautenbacher will provide you with certificates that are valid for a limited time but can be renewed. Do not share those certificates with people outside of our group.

## 3. How to use the gRPC Client using Python?

### Write HDF5 == Get everything as file -> Input for any kind of converter

```python
from prosit_grpc.predictPROSIT import PROSITpredictor
predictor = PROSITpredictor(server="proteomicsdb.org:8500",
                            path_to_ca_certificate= "path/to/certificate/Proteomicsdb-Prosit.crt",
                            path_to_certificate= "path/to/certificate/individual_certificate_name.crt",
                            path_to_key_certificate= "path/to/certificate/individual_certificate_name.key",
                            )
predictor.predict_to_hdf5(sequences=["AAAAAKAK","AAAAAA"],
                          charges=[1,2],
                          collision_energies=[25,25],
                          intensity_model="Prosit_2019_intensity",
                          irt_model="Prosit_2019_irt",
                          path_hdf5="output.hdf5")
```

### Alternative: get predictions as dictionary in python

Predictions are generated for the model you specify.

```python
from prosit_grpc.predictPROSIT import PROSITpredictor
predictor = PROSITpredictor(server="proteomicsdb.org:8500",
                            path_to_ca_certificate= "path/to/certificate/Proteomicsdb-Prosit.crt",
                            path_to_certificate= "path/to/certificate/individual_certificate_name.crt",
                            path_to_key_certificate= "path/to/certificate/individual_certificate_name.key",
                            )
# predicts intensity, proteotypicity, iRT
output_dict = predictor.predict(sequences=["AAAAAKAK","AAAAAA"],
                                charges=[1,2],
                                collision_energies=[25,25],
                                intensity_model="Prosit_2019_intensity",
                                irt_model="Prosit_2019_irt",
                                proteotypicity_model="Prosit_2020_proteotypicity"
                                )
```

If you only want specific predictions you can ignore the other models.

```python
from prosit_grpc.predictPROSIT import PROSITpredictor
predictor = PROSITpredictor(server="proteomicsdb.org:8500",
                            path_to_ca_certificate= "path/to/certificate/Proteomicsdb-Prosit.crt",
                            path_to_certificate= "path/to/certificate/individual_certificate_name.crt",
                            path_to_key_certificate= "path/to/certificate/individual_certificate_name.key",
                            )

# predicts ONLY iRT
output_dict_irt = predictor.predict(sequences=["AAAAAKAK","AAAAAA"],
                                    irt_model="Prosit_2019_irt",
                                    )

# predicts ONLY proteotypicity
output_dict_proteotypicity = predictor.predict(sequences=["AAAAAKAK","AAAAAA"],
                                              proteotypicity_model="Prosit_2020_proteotypicity",
                                              )

# predicts ONLY intensity/spectra
output_dict_intensity = predictor.predict(sequences=["AAAAAKAK","AAAAAA"],
                                          charges=[1,2],
                                          collision_energies=[25,25],
                                          intensity_model="Prosit_2020_intensity",
                                          )
```

## Sequence Restrictions

The peptide Sequence can only contain the following AA abbreviations:
(Cysteine is expected to be alkylated as such all three representations are treated the same)

Amino acid|accepted abbreviation
:-----:|:-----:
Alanine|A
Cysteine|C, Cac
Aspartic acid|D
Glutamic acid|E
Phenylalanine|F
Glycine|G
Histidine|H
Isoleucine|I
Lysine|K
Leucine|L
Methionine|M
Asparagine|N
Pyrrolysine|O
Proline|P
Glutamine|Q
Arginine|R
Serine|S
Threonine|T
Selenocysteine|U
Valine|V
Tryptophan|W
Tyrosine|Y



Modified AA can be specified with:

Modified Amino Acid|accepted abbreviation
:-----:|:-----:
Alkylated Cystein |C, Cac
Oxidized Methionine|M(ox), OxM
Phosphorylated Serine|Phs
Phosphorylated Threonine|PhT
Phosphorylated Tyrosine|PhY

# Developer information
## How to use the gRPC client in my own project using the poetry package managment system?

Specify the "**SSH** clone link" of the `prosit_grpc` repository in your `pyproject.toml`.

```
prosit_grpc = {git = "git@gitlab.lrz.de:proteomics/prosit_tools/prosit_grpc.git"}
```
