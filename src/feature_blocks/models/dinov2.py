from copy import deepcopy

import timm
import torch
from torch import nn


class DINOv2(nn.Module):
    def __init__(self):
        super().__init__()

        # self.model = AutoModel.from_pretrained('facebook/dinov2-base')
        # self.n_features = model.config.hidden_size

        self.model = timm.create_model("vit_large_patch14_dinov2", pretrained=True)
        self.model.eval()
        self.n_features = self.model.num_features
        self.output_shape = (self.n_features, 1, 1, 1)  # (C, Z, H, W)

    def forward(self, x):
        from torchvision import transforms

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
            features = deepcopy(features)

        features = features.squeeze()

        features = features.view(len(features), *([1] * 3))

        return features.cpu().numpy()
