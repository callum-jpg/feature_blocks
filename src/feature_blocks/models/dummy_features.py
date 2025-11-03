import uuid

import torch
from torch import nn


def random_torch_tensor(shape):
    """
    When spawning dask workers, if they start at the same time
    they can have the same seed. This can lead to a randomly initialized
    tensor to look the same across an image, which doesn't feel right.

    This function forces randomness by setting the seed based on a uuid hash.
    """
    unique_seed = (2**32) ^ hash(uuid.uuid4())
    torch.manual_seed(unique_seed)

    tensor = torch.randint(0, 2**16 - 1, shape)

    return tensor


class DummyModel(nn.Module):
    """
    An example feature extraction class
    """
    def __init__(self):
        super().__init__()

        # Define the model
        self.model = nn.Identity()
        # Define how many features the model returns (ie. the hidden dim)
        # This allows up to build the zarr store before any inference has been
        # run, which therefore allows for on the fly writing results to disk.
        self.n_features = 42
        self.output_shape = (self.n_features, 1, 1, 1)  # (C, Z, H, W)

    def forward(self, x):
        """
        We don't actually compute from x
        """
        features = random_torch_tensor(self.output_shape)

        return features.cpu().numpy()
