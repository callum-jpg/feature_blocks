import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch import nn


class UNI(nn.Module):
    def __init__(self):
        super().__init__()

        self.n_features = 1536
        self.output_shape = (self.n_features, 1, 1, 1)  # (C, Z, H, W)

    def forward(self, x):
        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        model = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs
        )
        transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
        model.eval()

        x = x[:, 0, ...]  # Drop the Z-dim
        x = torch.from_numpy(x).to(torch.float)
        x = transform(x).unsqueeze(0)

        with torch.no_grad():
            features = model(x)

        features = features.reshape(self.output_shape)

        return features.cpu().numpy()
