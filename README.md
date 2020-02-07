# Prosit gRPC Client as a poetry package

## How to use the gRPC client in my own project using the poetry package managment system?

Specify the "**SSH** clone link" of the `prosit_grpc` repsitory in your `pyproject.toml`.

```
prosit_grpc = {git = "git@gitlab.lrz.de:proteomics/prosit_tools/prosit_grpc.git"}
```

## How to use the gRPC Client using Python?
```
from prosit_grpc.predictPROSIT import PredictPROSIT

# Predict Intensities
predictor = PredictPROSIT(server="131.159.152.7:8500",
                          sequences_list=["AAAAAKAK","AAAAAA"],
                          charges_list=[1,2],
                          collision_energies_list=[25,25],
                          model_name="intensity"
                          )
# Get predictions
predictions = predictor.get_predictions()

```

```
# Predict Proteotypicity
predictor = PredictPROSIT(server="131.159.152.7:8500",
                          sequences_list= ["THDLGKW", "VLQKQFFYCTMEKWNGRT", "QMQCNWNVMQGAPSMTCEHRVEYSMEWIID"],
                          model_name="proteotypicity"
                          )
pred = predictor.get_predictions()
```

```
# Predict iRT
predictor = PredictPROSIT(server="131.159.152.7:8500",
                              sequences_list=["THDLGKW", "VLQKQFFYCTMEKWNGRT", "QMQCNWNVMQGAPSMTCEHRVEYSMEWIID"],
                              model_name="iRT"
                              )
pred = predictor.get_predictions()
```

## Sequence Restrictions

The peptide Sequence can only contain the following AA abbreviations:
(Cysteine is expected to be alkylated as such all three representations are treated the same)

Amino acid|accepted abbreviation
:-----:|:-----:
Alanine|A
Cysteine|C, Cac, Cam
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
Alkylated Cystein |C, Cac, Cam
Oxidized Methionine|M(ox), M(O), OxM
Phosphorylated Serine|Phs, S(ph)
Phosphorylated Threonine|PhT, T(ph)
Phosphorylated Tyrosine|PhY, Y(ph)