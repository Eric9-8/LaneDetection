import torchvision.models as models
from torchvision._internally_replaced_utils import load_state_dict_from_url
from typing import Any, Optional
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.fcn import FCNHead
from my_unet import UnetHead, Unet

model_urls = {
    "regnet_y_400mf": "https://download.pytorch.org/models/regnet_y_400mf-c65dace8.pth",
    "regnet_y_3_2gf": "https://download.pytorch.org/models/regnet_y_3_2gf-b5a9779c.pth",
    "regnet_y_8gf": "https://download.pytorch.org/models/regnet_y_8gf-d0d0e4a8.pth",
}


def seg_model(
        backbone: str,
        # num_classes: int,
        # aux: Optional[bool],
        pretrained_backbone: bool = True,
        dilation: bool = False,
) -> nn.Module:
    if dilation:
        # backbone = my_regnet()
        cc = []
    else:
        backbone = models.regnet.__dict__[backbone](
            pretrained=pretrained_backbone)

        base_layers = list(backbone.children())
        stem = base_layers[0]

        layer1 = base_layers[1].block1
        layer2 = base_layers[1].block2
        layer3 = base_layers[1].block3
        layer4 = base_layers[1].block4

        out1 = list(layer1.children())[0].proj.out_channels
        out2 = list(layer2.children())[0].proj.out_channels
        out3 = list(layer3.children())[0].proj.out_channels
        out4 = list(layer4.children())[0].proj.out_channels
        out_channels = [out4, out3, out2, out1]
        #
        # out_layer = 'trunk_output'
        # aux_layer = 'stem'
        # aux_inplanes = stem.out_channels

    # return_layers = {out_layer: 'out'}
    # if aux:
    #     return_layers[aux_layer] = 'aux'

    # backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    aux_classifier = None
    # if aux:
    #     aux_classifier = FCNHead(aux_inplanes, num_classes)
    model_map = {
        'Unet': (UnetHead, Unet)
    }
    # classifier = model_map['Unet'][0](out_channels, num_classes)
    up_sample = model_map['Unet'][0](out_channels, backbone=backbone)

    base_model = model_map['Unet'][1]
    model = base_model(backbone, up_sample, aux_classifier)
    return model


def load_model(
        backbone: str,
        pretrained: bool,
        progress: bool,
        # num_classes: int,
        # aux_loss: Optional[bool],
        **kwargs: Any
) -> nn.Module:
    if pretrained:
        #     aux_loss = True
        kwargs['pretrained_backbone'] = False
    # model = seg_model(backbone, num_classes, aux_loss, **kwargs)
    model = seg_model(backbone, **kwargs)

    if pretrained:
        load_weight(model, backbone, progress)
    return model


def load_weight(
        model: nn.Module,
        backbone: str,
        progress: bool,
) -> None:
    arch = backbone
    model_url = model_urls.get(arch, None)
    state_dict = load_state_dict_from_url(model_url, progress=progress)
    model.load_state_dict(state_dict, strict=False)


def unet_regnet_y_400mf(
        pretrained: bool = False,
        progress: bool = True,
        # num_classes: int = 2,
        # aux_loss: Optional[bool] = None,
        **kwargs: Any
) -> nn.Module:
    # return load_model('regnet_y_400mf', pretrained, progress, num_classes, aux_loss, **kwargs)
    return load_model('regnet_y_400mf', pretrained, progress, **kwargs)


def unet_regnet_y_3_2gf(
        pretrained: bool = False,
        progress: bool = True,
        # num_classes: int = 2,
        # aux_loss: Optional[bool] = None,
        **kwargs: Any
) -> nn.Module:
    # return load_model('regnet_y_3_2gf', pretrained, progress, num_classes, aux_loss, **kwargs)
    return load_model('regnet_y_3_2gf', pretrained, progress, **kwargs)


def unet_regnet_y_8gf(
        pretrained: bool = False,
        progress: bool = True,
        # num_classes: int = 2,
        # aux_loss: Optional[bool] = None,
        **kwargs: Any
) -> nn.Module:
    # return load_model('regnet_y_8gf', pretrained, progress, num_classes, aux_loss, **kwargs)
    return load_model('regnet_y_8gf', pretrained, progress, **kwargs)
