#model.py
import torch
import torchvision.models as models
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class ImageSearchModel(torch.nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        self.model = efficientnet_b3(weights=weights)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = torch.nn.Identity()
        self.model.to(device)
        self.preprocess = weights.transforms()

    def forward(self, x):
        return self.model(x)

    def get_embeddings(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def get_preprocess(self):
        return self.preprocess
