import dask
import dask.array
import torch
from torch import nn

# from zarr_back import tensor_to_dask


def tensor_to_dask(tensor):
    dask_array = dask.array.from_array(
        tensor.numpy(), chunks=tuple(tensor.shape)
    )  # .astype(numpy.uint16)
    return dask_array


class ConvNet(nn.Module):
    def __init__(self, in_channels=3, feature_dim=128):
        super(ConvNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels, 32, kernel_size=3, stride=1, padding=1
            ),  # -> [B, 32, H, W]
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [B, 32, H/2, W/2]
            nn.Conv2d(
                32, 64, kernel_size=3, stride=1, padding=1
            ),  # -> [B, 64, H/2, W/2]
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [B, 64, H/4, W/4]
            nn.Conv2d(
                64, 128, kernel_size=3, stride=1, padding=1
            ),  # -> [B, 128, H/4, W/4]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> [B, 128, 1, 1]
        )
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten -> [B, 128]
        x = self.fc(x)  # -> [B, feature_dim]
        return x


class ConvFeatures(nn.Module):
    """
    An example feature extraction class
    """

    def __init__(self, n_channels):
        super().__init__()

        # Define how many features the model returns (ie. the hidden dim)
        # This allows up to build the zarr store before any inference has been
        # run, which therefore allows for on the fly writing results to disk.
        self.n_features = 42
        self.output_shape = (self.n_features, 1, 1, 1)  # (C, Z, H, W)

        # Define the model
        self.model = ConvNet(n_channels, self.n_features)
        self.model.eval()

    def forward(self, x):
        with torch.no_grad():
            x = x[:, 0, ...]  # Drop the Z-dim
            # Convert to tensor
            x = torch.from_numpy(x)
            # Add a batch dim
            x = x.unsqueeze(0).to(torch.float)

            features = self.model(x)
            # Model outputs cannot be easily garbage collected,
            # so we create copy that can.
            # features = deepcopy(features)
            # Remove batch dim
            features = features.squeeze()
            # Make shape (n_features, 1, 1, 1)
            features = features.view(len(features), *([1] * 3))

        return features.cpu().numpy()
