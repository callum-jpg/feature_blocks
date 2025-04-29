import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModel


class PhikonV2(nn.Module):
    def __init__(self):
        super().__init__()

        # self.model = AutoModel.from_pretrained('facebook/dinov2-base')
        # self.n_features = model.config.hidden_size

        self.processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
        self.model = AutoModel.from_pretrained("owkin/phikon-v2")
        self.model.eval()

        self.n_features = 1024
        self.output_shape = (self.n_features, 1, 1, 1)  # (C, Z, H, W)

    def forward(self, x):

        x = x[:, 0, ...]  # Subset to the first z-plane
        x = self.processor(x, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**x)
            features = outputs.last_hidden_state[:, 0, :]

        features = features.squeeze()

        features = features.view(len(features), *([1] * 3))

        return features.cpu().numpy()
