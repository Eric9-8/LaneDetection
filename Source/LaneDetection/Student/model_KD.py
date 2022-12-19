from Student.utils.my_unet import UnetHead
import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
import my_regnet


class _KDNET(nn.Module):
    def __init__(self, pretrained=True, backbone=None):
        super(_KDNET, self).__init__()
        self.pretrained = pretrained
        if not self.pretrained:
            self.weight_init()
        backbones = models.regnet.__dict__[backbone](pretrained=pretrained)
        # backbones = my_regnet.__dict__[backbone](pretrained=pretrained)
        base_layers = list(backbones.children())

        layer1 = base_layers[1].block1
        layer2 = base_layers[1].block2
        layer3 = base_layers[1].block3
        layer4 = base_layers[1].block4

        out1 = list(layer1.children())[0].proj.out_channels
        out2 = list(layer2.children())[0].proj.out_channels
        out3 = list(layer3.children())[0].proj.out_channels
        out4 = list(layer4.children())[0].proj.out_channels
        out_channels = [out4, out3, out2, out1]

        self.model = UnetHead(in_channels=out_channels, backbone=backbone, pretrained=pretrained)

        self.scale_background = 0.4
        self.scale_seg = 1.0
        # self.scale_exist = 0.1
        self.kd_weight = 1.0

        # self.fc_input_feature = 72000
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([self.scale_background, 1]))
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.layer1 = nn.Sequential(nn.Dropout(0.2), nn.Conv2d(48, 2, 1))
        # self.layer2 = nn.Sequential(nn.Softmax(dim=1), nn.AvgPool2d(2, 2))
        # self.fc = nn.Sequential(nn.Linear(self.fc_input_feature, 128), nn.ReLU(), nn.Linear(128, 4))

    # def forward(self, img, seg_gt=None, exist_gt=None, seg_kd=None, exist_kd=None):
    def forward(self, img, seg_gt=None, seg_kd=None):
        x = self.model(img)
        x = self.layer1(x)
        seg_pre = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

        # x = self.layer2(x)
        # x = x.view(-1, self.fc_input_feature)
        # exist_pre = self.fc(x)

        if seg_gt is not None and seg_kd is not None:
            # loss_seg_gt = self.ce_loss(seg_pre, seg_gt)
            # loss_exist_gt = self.bce_loss(exist_pre, exist_gt)
            # loss_seg_kd = self.ce_loss(seg_pre, seg_kd)
            # loss_exist_kd = self.bce_loss(exist_pre, exist_kd)
            # loss = (loss_seg_gt * self.scale_seg + loss_exist_gt * self.scale_exist) + self.kd_weight * (
            #         loss_seg_kd * self.scale_seg + loss_exist_kd * self.scale_exist)
            loss = self.ce_loss(seg_pre, seg_gt) * self.scale_seg + self.kd_weight * (
                    self.ce_loss(seg_pre, seg_kd) * self.scale_seg)
        else:
            # loss_seg_gt = torch.tensor(0, dtype=img.dtype, device=img.device)
            # loss_exist_gt = torch.tensor(0, dtype=img.dtype, device=img.device)
            # loss_seg_kd = torch.tensor(0, dtype=img.dtype, device=img.device)
            # loss_exist_kd = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss = torch.tensor(0, dtype=img.dtype, device=img.device)
        # return seg_pre, exist_pre, loss_seg_gt, loss_exist_gt, loss_seg_kd, loss_exist_kd, loss
        return seg_pre, loss

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data[:] = 1
                m.bias.data.zero_()
