# Prosit gRPC Client as a poetry package

## How to use the gRPC client in my own project using the poetry package managment system?

Specify the "**SSH** clone link" of the `prosit_grpc` repsitory in your `pyproject.toml`.

```
prosit_grpc = {git = "git@gitlab.lrz.de:proteomics/prosit_tools/prosit_grpc.git"}
```

## How to use the gRPC Client using Python?
```
from prosit_grpc.predictPROSIT_2 import PredictPROSIT

#Initialize Predictor
predictor = PredictPROSIT(server="131.159.152.7:8500",
                          sequences_list=["AAAAAKAK","AAAAAA"],
                          charges_list=[1,2],
                          collision_energies_list=[25,25],
                          model_name="intensity"
                          )

# Get predictions
predictions = predictor.get_predictions()

```