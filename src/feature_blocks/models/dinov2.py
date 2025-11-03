import timm
import torch
from torch import nn
from torchvision import transforms


class DINOv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("vit_large_patch14_dinov2", pretrained=True)
        self.n_features = self.model.num_features
        self.output_shape = (self.n_features, 1, 1, 1)  # (C, Z, H, W)

    def forward(self, x):
        
        self.model.eval()

        transform = transforms.Compose(
            [
                transforms.Resize((518, 518)),
            ]
        )

        x = torch.from_numpy(x)
        x = x[:, 0, ...]  # Drop the Z-dim
        x = x.unsqueeze(0).to(torch.float)

        with torch.no_grad():
            features = self.model(transform(x))

        features = features.reshape(self.output_shape)

        return features.cpu().numpy()
