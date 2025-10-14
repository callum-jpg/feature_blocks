import torch
import timm 
from torchvision import transforms


class H_optimus_0(nn.Module):
    def __init__(self):

        self.n_features = 1536
        self.output_shape = (self.n_features, 1, 1, 1)  # (C, Z, H, W)

    def forward(self, x):
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False
        )
        model.eval()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), 
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])
        
        x = x[:, 0, ...]  # Drop the Z-dim
        x = torch.from_numpy(x).to(torch.float)
        x = transform(x).unsqueeze(0)

        with torch.no_grad():
            features = model(x)

        features = features.reshape(self.output_shape)

        return features.cpu().numpy()
