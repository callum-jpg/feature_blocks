import timm
import torch
from torch import nn


# from memory_profiler import profile
class GigaPathTile(nn.Module):
    def __init__(self):
        super().__init__()

        self.n_features = 1536
        self.output_shape = (self.n_features, 1, 1, 1)  # (C, Z, H, W)

    # @profile
    def forward(self, x):
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        model.eval()

        x = x[:, 0, ...]  # Drop the Z-dim
        x = x.transpose(1, 2, 0)  # To (YXC) for transform
        x = transform(x).unsqueeze(0).to(torch.float)

        with torch.no_grad():
            features = model(x)

        features = features.reshape(self.output_shape)

        return features.cpu().numpy()
