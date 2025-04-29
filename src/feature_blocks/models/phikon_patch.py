import einops
import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModel


class PhikonV2Patch(nn.Module):
    def __init__(self):
        super().__init__()

        # self.model = AutoModel.from_pretrained('facebook/dinov2-base')
        # self.n_features = model.config.hidden_size

        self.processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
        self.model = AutoModel.from_pretrained("owkin/phikon-v2")
        self.model.eval()

        self.n_features = 1024

        self.output_shape = (self.n_features, 1, 14, 14)  # (C, Z, H, W)

    def forward(self, x):

        x = x[:, 0, ...]  # Subset to the first z-plane
        x = self.processor(x, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**x)
            features = outputs.last_hidden_state[:, 1:, :]

        # We don't need to squeeze the batch dimension out
        # since it can be used as the z-dim
        # features = features.view(self.output_shape)
        features = einops.rearrange(
            features, "B (P_H P_W) D -> D B P_H P_W", P_H=14, P_W=14
        )

        return features.cpu().numpy()
