import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

import torch.nn as nn
from torchvision import models

def load_resnet():
    model = models.resnet18(pretrained=True)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer (2 classes: real/fake)
    model.fc = nn.Linear(model.fc.in_features, 2)

    return model

def predict(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        # Fix for GoogLeNet
        if hasattr(outputs, "logits"):
            outputs = outputs.logits
        _, pred = torch.max(outputs, 1)

    return pred.item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def modify_last_layer(model, in_features):
    return torch.nn.Linear(in_features, 2)

def load_resnet18():
    model = models.resnet18(weights=None)
    model.fc = modify_last_layer(model, model.fc.in_features)
    return model.to(device)

def load_resnet34():
    model = models.resnet34(weights=None)
    model.fc = modify_last_layer(model, model.fc.in_features)
    return model.to(device)

def load_mobilenet():
    model = models.mobilenet_v2(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(model.last_channel, 2)

    return model

def load_efficientnet():
    model = models.efficientnet_b0(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    return model

def load_vgg():
    model = models.vgg16(weights=None)
    model.classifier[6] = modify_last_layer(model, model.classifier[6].in_features)
    return model.to(device)

def load_densenet():
    model = models.densenet121(weights=None)
    model.classifier = modify_last_layer(model, model.classifier.in_features)
    return model.to(device)

def load_alexnet():
    model = models.alexnet(weights=None)
    model.classifier[6] = modify_last_layer(model, model.classifier[6].in_features)
    return model.to(device)

def load_squeezenet():
    model = models.squeezenet1_0(weights=None)
    model.classifier[1] = torch.nn.Conv2d(512, 2, kernel_size=1)
    return model.to(device)

def load_shufflenet():
    model = models.shufflenet_v2_x1_0(weights=None)
    model.fc = modify_last_layer(model, model.fc.in_features)
    return model.to(device)

def load_googlenet():
    model = models.googlenet(weights=None)
    model.fc = modify_last_layer(model, model.fc.in_features)
    return model.to(device)

