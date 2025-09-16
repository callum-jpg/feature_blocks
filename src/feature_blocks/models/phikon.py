
import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModel


class PhikonV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.n_features = 1024
        self.output_shape = (self.n_features, 1, 1, 1)  # (C, Z, H, W)

    def forward(self, x):
        # It seems odd to define the model on each forward call, but this
        # avoids dask.delayed serialisation errors on workers. As far as
        # dask is concerned, this is a single function that can be serialized
        # once. If the model had a local scope (ie. self.model), dask seems to
        # have issues in serialisation. This is due to the class having been
        # instantiated outside of the delayed call.
        # Regardless, by structuring as a class, we can define configurations
        # in the __init__.
        processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
        model = AutoModel.from_pretrained("owkin/phikon-v2")

        x = x[:, 0, ...]  # Subset to the first z-plane
        x = processor(x, return_tensors="pt")

        with torch.no_grad():
            features = model(**x).last_hidden_state[:, 0, :]

        features = features.reshape(self.output_shape)

        return features.detach().cpu().numpy()
