import torch
import torch.nn as nn
from mobilevig import Stem, Grapher, FFN

"""Custom Unet with mobiblevig as backbone
    for segmentation of images"""

class UNetMobileVig(nn.Module):
    def __init__(self, local_channels) -> None:
        """Initialize the model
        Args: local_channels (list): list of channels for each layer
        """
        super(UNetMobileVig, self).__init__()

        self.stem = Stem(input_dim=3, output_dim=local_channels[0])
        self.backbone = nn.ModuleList([])
        self.backbone += [nn.Sequential(
                        Grapher(input_dim=local_channels[0], output_dim=local_channels[1]),
                        FFN(input_dim=local_channels[1], output_dim=local_channels[1], drop_path=0.1),
                        )]

    def forward(self, x):
        x = self.stem(x)
        return x
    

if __name__ == "__main__":
    # random input to test model
    x = torch.randn(1, 3, 224, 224)
    model = UNetMobileVig(local_channels=[56])
    y = model(x)
    print(y.shape)