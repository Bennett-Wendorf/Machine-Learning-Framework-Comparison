import torch
import torchvision.models as models

model = models.vgg16(weights="IMAGENET1K_V1")
torch.save(model.state_dict(), "model_weights.pth")

model = models.vgg16() # we don't specify the weights, i.e. create untrained model
model.load_state_dict(torch.load("model_weights.pth"))
model.eval() # set the model to evaluation mode
print(model)

torch.save(model, "model.pth")
model = torch.load("model.pth")
model.eval()
print(model)