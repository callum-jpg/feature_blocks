import timm
import torch
from torch import nn
from torchvision import transforms


class GigaPathTile(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", pretrained=True
        )
        self.model.eval()
        self.n_features = self.model.num_features
        self.output_shape = (self.n_features, 1, 1, 1)  # (C, Z, H, W)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def forward(self, x):
        x = torch.from_numpy(x)
        x = x[:, 0, ...]  # Drop the Z-dim
        x = x.unsqueeze(0).to(torch.float)

        with torch.no_grad():
            features = self.model(self.transform(x))

        features = features.squeeze()

        features = features.view(len(features), *([1] * 3))

        return features.cpu().numpy()
