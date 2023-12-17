from model.lenet5 import LeNet5

def get_model(model_name):
  if (model_name == "lenet5"):
    return LeNet5
  else:
    print("unknown model")
