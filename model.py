import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=4):
    model = models.efficientnet_b0(pretrained=True)

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )

    return model