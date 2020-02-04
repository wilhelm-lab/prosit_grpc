
from prosit_grpc.predictPROSIT_2 import PredictPROSIT

predictor = PredictPROSIT(server="131.159.152.7:8500",
                          sequences_list=["AAAAAKAK","AAAAAA"],
                          charges_list=[1,2],
                          collision_energies_list=[25,25],
                          model_name="intensity_prosit_publication"
                          )
# Get predictions

print("=============================================")
output = predictor.get_predictions()
print("=============================================")
print(output)
