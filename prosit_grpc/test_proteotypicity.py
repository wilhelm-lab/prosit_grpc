
from prosit_grpc.predictPROSIT_2 import PredictPROSIT

predictor = PredictPROSIT(server="131.159.152.7:8500",
                          sequences_list=["AAAAAKAK","AAAAAA"],
                          model_name="proteotypicity"
                          )
# Get predictions

print("=============================================")
output = predictor.get_predictions()
print("=============================================")
print(output)


