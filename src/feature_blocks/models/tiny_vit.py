import timm
import torch
from torch import nn
from torchvision import transforms
import numpy


class TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        # A very small ViT model — downloads <10MB weights
        self.model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
        self.n_features = self.model.num_features
        # Remove the ImageNet classification head
        self.model.head = nn.Identity()
        self.output_shape = (self.n_features, 1, 1, 1)

    def forward(self, x):
        transform = transforms.Resize((224, 224))
        if isinstance(x, numpy.ndarray):
            x = torch.from_numpy(x)
            x = x[:, 0, ...]  # Drop Z-dim
            x = x.unsqueeze(0).to(torch.float)
            pre_batched = False
        
        elif isinstance(x, torch.Tensor):
            # Data has been batched already
            pre_batched = True

        with torch.no_grad():
            features = self.model(transform(x))

        if not pre_batched:
            features = features.reshape(self.output_shape)

        return features.cpu().numpy()