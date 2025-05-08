import timm
import torch
from torch import nn

class GigaPathTilePatch(nn.Module):
    def __init__(self):
        super().__init__()

        self.n_features = 1536
        self.output_shape = (self.n_features, 1, 14, 14)  # (C, Z, H, W)

    def forward(self, x):
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    224, 
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        model = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", pretrained=True
        )
        model.eval()

        x = x[:, 0, ...]  # Drop the Z-dim
        x = x.transpose(1, 2, 0) # To (YXC) for transform
        x = transform(x).unsqueeze(0).to(torch.float)

        with torch.no_grad():
            features = model.forward_intermediates(x)[0][:, 1:, :]

        # We don't need to squeeze the batch dimension out
        # since it can be used as the z-dim
        # features = features.view(self.output_shape)
        features = einops.rearrange(
            features, "B (P_H P_W) D -> D B P_H P_W", P_H=self.output_shape[2], P_W=self.output_shape[3]
        )

        return features.cpu().numpy()
