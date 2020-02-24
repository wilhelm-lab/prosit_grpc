# Status of code
## Master
[![pipeline status](https://gitlab.lrz.de/proteomics/prosit_tools/prosit_grpc/badges/master/pipeline.svg)](https://gitlab.lrz.de/proteomics/prosit_tools/prosit_grpc/commits/master)
## Develop
[![pipeline status](https://gitlab.lrz.de/proteomics/prosit_tools/prosit_grpc/badges/develop/pipeline.svg)](https://gitlab.lrz.de/proteomics/prosit_tools/prosit_grpc/commits/develop)

# Prosit gRPC Client as a poetry package

## 1. Simple installation by pip
We provide **tagged** versions of our package pointing to the latest tested master releases but can give you also direct access to new features if requested. The correctness is indicated by the green=good or red=wrong flags above. 
```bash
pip install -e git+https://gitlab.lrz.de/proteomics/prosit_tools/prosit_grpc.git@v1.0.1#egg=prosit_grpc
```
### If you deployed an ssh key
```bash
pip install -e git+git@gitlab.lrz.de:proteomics/prosit_tools/prosit_grpc.git@v1.0.1#egg=prosit_grpc
```
## 2. Ask Ludwig for certificates
You are using a special access to our GPUs and are therefore required to identify against our server. Ludwig Lautenbacher will provide you with certificates that are valid for a limited time but can be renewed. Do not share those certificates with people outside of our group.

## 3. How to use the gRPC Client using Python?

### Write HDF5 == Get everything as file -> Input for any kind of converter

```python
predictor = PredictPROSIT(server="proteomicsdb.org:8500",
                          sequences_list=["AAAAAKAK","AAAAAA"],
                          charges_list=[1,2],
                          collision_energies_list=[25,25],
                          model_name="LOREM_IPSUM",
                          path_to_ca_certificate= "path/to/certificate/Proteomicsdb-Prosit.crt",
                          path_to_certificate= "path/to/certificate/individual_certificate_name.crt",
                          path_to_key_certificate= "path/to/certificate/individual_certificate_name.key",
                          )

predictor.write_hdf5(intensity_model="intensity_prosit_publication",
                         irt_model="iRT",
                         output_file="output.hdf5",)
```

### Alternative: get predictions seperately

```python
from prosit_grpc.predictPROSIT import PredictPROSIT

# Predict Intensities
predictor = PredictPROSIT(server="proteomicsdb.org:8500",
                          sequences_list=["AAAAAKAK","AAAAAA"],
                          charges_list=[1,2],
                          collision_energies_list=[25,25],
                          model_name="intensity_prosit_publication",
                          path_to_ca_certificate= "path/to/certificate/Proteomicsdb-Prosit.crt",
                          path_to_certificate= "path/to/certificate/individual_certificate_name.crt",
                          path_to_key_certificate= "path/to/certificate/individual_certificate_name.key",
                          )
# Get predictions
predictions = predictor.get_predictions()
fragment_annotations = predictor.get_fragment_annotation()
fragment_masses = predictor.get_fragment_masses()

```

```python
# Predict Proteotypicity
predictor = PredictPROSIT(server="proteomicsdb.org:8500",
                          sequences_list= ["THDLGKW", "VLQKQFFYCTMEKWNGRT", "QMQCNWNVMQGAPSMTCEHRVEYSMEWIID"],
                          model_name="proteotypicity",
                          path_to_ca_certificate= "path/to/certificate/Proteomicsdb-Prosit.crt",
                          path_to_certificate= "path/to/certificate/individual_certificate_name.crt",
                          path_to_key_certificate= "path/to/certificate/individual_certificate_name.key",
                          )
pred = predictor.get_predictions()
```

```python
# Predict iRT
predictor = PredictPROSIT(server="proteomicsdb.org:8500",
                              sequences_list=["THDLGKW", "VLQKQFFYCTMEKWNGRT", "QMQCNWNVMQGAPSMTCEHRVEYSMEWIID"],
                              model_name="iRT",
                              path_to_ca_certificate= "path/to/certificate/Proteomicsdb-Prosit.crt",
                              path_to_certificate= "path/to/certificate/individual_certificate_name.crt",
                              path_to_key_certificate= "path/to/certificate/individual_certificate_name.key",
                              )
pred = predictor.get_predictions()
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
# How to use the gRPC client in my own project using the poetry package managment system?

Specify the "**SSH** clone link" of the `prosit_grpc` repsitory in your `pyproject.toml`.

```
prosit_grpc = {git = "git@gitlab.lrz.de:proteomics/prosit_tools/prosit_grpc.git"}
```

