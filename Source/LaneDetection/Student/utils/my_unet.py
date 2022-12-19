import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from typing import List
from torchvision.models.segmentation._utils import _SimpleSegmentationModel
import my_regnet

__all__ = ['Unet']


class Unet(_SimpleSegmentationModel):
    pass


class UnetHead(nn.Sequential):
    # def __init__(self, in_channels: List[int], num_classes: int, bilinear: bool = False) -> None:
    def __init__(self, in_channels: List[int], backbone: str, bilinear: bool = False, pretrained: bool = True) -> None:
        super(UnetHead, self).__init__(
            _MyUnet(in_channels, backbone=backbone, bilinear=bilinear, pretrained=pretrained),
            # nn.Conv2d(in_channels[-1], in_channels[-1], 3, padding=1, bias=False),
            # nn.BatchNorm2d(in_channels[-1]),
            # nn.ReLU(),
            # nn.Conv2d(in_channels[-1], num_classes, 1)
        )


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: bool, mid_channels: int = None) -> None:
        if not mid_channels:
            mid_channels = out_channels
        if not dilation:
            modules = [
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                # nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
                # nn.BatchNorm2d(out_channels),
                # nn.ReLU(inplace=True)
            ]
        else:
            modules = [
                nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                # nn.Conv2d(mid_channels, out_channels, 3, padding=24, dilation=24, bias=False),
                # nn.BatchNorm2d(out_channels),
                # nn.ReLU(inplace=True)
            ]
        super(DoubleConv, self).__init__(*modules)


class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, biliner: bool = True) -> None:
        super(Decoder, self).__init__()
        if biliner:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            self.conv = DoubleConv(in_channels, out_channels, dilation=False, mid_channels=(in_channels // 2))
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dilation=False)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class _MyUnet(nn.Module):
    def __init__(self, in_channels: List[int], backbone: str, bilinear: bool = False, pretrained: bool = True):
        super(_MyUnet, self).__init__()

        backbone = models.regnet.__dict__[backbone](pretrained=True)
        # backbone = my_regnet.__dict__[backbone](pretrained=pretrained)

        self.base_layers = list(backbone.children())
        # self.layer0 = self.base_layers[0]
        # self.layer1 = self.base_layers[1]
        # self.layer2 = self.base_layers[2]
        # self.layer3 = self.base_layers[3]
        # self.layer4 = self.base_layers[4]

        self.layer0 = self.base_layers[0]  # 32

        self.layer1 = self.base_layers[1].block1  # 48
        self.layer2 = self.base_layers[1].block2  # 104
        self.layer3 = self.base_layers[1].block3  # 208
        self.layer4 = self.base_layers[1].block4  # 440

        # modules = []
        factor = 2 if bilinear else 1

        self.conv1 = nn.Conv2d(440, 416, kernel_size=1)
        self.conv2 = nn.Conv2d(104, 96, kernel_size=1)

        self.layer5 = Decoder(416, 208 // factor, bilinear)
        self.layer6 = Decoder(208, 104 // factor, bilinear)
        self.layer7 = Decoder(96, 48 // factor, bilinear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.layer0(x)  # 32,144,400

        x1 = self.layer1(x0)  # 48,72,200
        x2 = self.layer2(x1)  # 104,36,100
        x3 = self.layer3(x2)  # 208,18,50
        x4 = self.layer4(x3)  # 440,9,25

        x4 = self.conv1(x4)  # 416,9,25
        x5 = self.layer5(x4, x3)  # 208,18,50

        x6 = self.layer6(x5, x2)  # 104,36,100

        x6 = self.conv2(x6)  # 96,36,100
        x7 = self.layer7(x6, x1)  # 48,72,200

        return x7
# test = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True, num_classes=21, aux_loss=True)
# test2 = models.segmentation.DeepLabV3
# print('----------------------------')
