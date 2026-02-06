import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.GTCM import GTCM
from src.model.FSCM import FSCM
from src.model.GACS import GACS
from src.utils.utils import *
import math
import timm

class PVT_V2_b2(nn.Module):
    def __init__(self, pretrained=True, ch_in=3):
        super().__init__()

        # 从 timm 创建模型
        model = timm.create_model("pvt_v2_b2", pretrained=pretrained, in_chans=ch_in)

        # Stage blocks
        self.patch_embed = model.patch_embed    # OverlapPatchEmbed
        self.stage1 = model.stages[0]           # Stage 1
        self.stage2 = model.stages[1]           # Stage 2
        self.stage3 = model.stages[2]           # Stage 3
        self.stage4 = model.stages[3]           # Stage 4

        # Out channels
        self.out_channels = [64, 128, 320, 512]

class GTFS_Net(nn.Module):
    """
    Geometry–Texture and Frequency–Spatial Collaborative Learning
    for Cross-Modality Remote Sensing Segmentation
    """
    def __init__(self, num_classes=6):
        super(GTFS_Net, self).__init__()
        self.encoder_opt = PVT_V2_b2(ch_in=3)
        self.encoder_dsm = PVT_V2_b2(ch_in=1)
        # [64, 128, 320, 512]
        out_channels = self.encoder_opt.out_channels

        self.corr1 = GTCM(out_channels[0])
        self.corr2 = GTCM(out_channels[1])
        self.corr3 = GTCM(out_channels[2])
        self.corr4 = GTCM(out_channels[3])

        self.fusion1 = FSCM(out_channels[0])
        self.fusion2 = FSCM(out_channels[1])
        self.fusion3 = FSCM(out_channels[2])
        self.fusion4 = FSCM(out_channels[3])


        self.decoder3 = GACS(out_channels[3], out_channels[2], out_channels[2])
        self.decoder2 = GACS(out_channels[2], out_channels[1], out_channels[1])
        self.decoder1 = GACS(out_channels[1], out_channels[0], out_channels[0])

        self.out = nn.Sequential(
            nn.Conv2d(out_channels[0], out_channels[0] // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels[0] // 2, num_classes, kernel_size=1),
        )

    def forward(self, x_opt, x_dsm):
        orsize = x_opt.shape[2:]  # H W
        x0_opt = self.encoder_opt.patch_embed(x_opt)
        x0_dsm = self.encoder_dsm.patch_embed(x_dsm)

        # stage 1
        x1_opt = self.encoder_opt.stage1(x0_opt)
        x1_dsm = self.encoder_dsm.stage1(x0_dsm)
        xr1_opt, xr1_dsm = self.corr1(x1_opt, x1_dsm)  # 1/4
        f1_dsm = xr1_dsm + x1_dsm

        # stage 2
        x2_opt = self.encoder_opt.stage2(xr1_opt + x1_opt)
        x2_dsm = self.encoder_dsm.stage2(xr1_dsm + x1_dsm)
        xr2_opt, xr2_dsm = self.corr2(x2_opt, x2_dsm)  # 1/8
        f2_dsm = xr2_dsm + x2_dsm

        # stage 3
        x3_opt = self.encoder_opt.stage3(xr2_opt + x2_opt)
        x3_dsm = self.encoder_dsm.stage3(xr2_dsm + x2_dsm)
        xr3_opt, xr3_dsm = self.corr3(x3_opt, x3_dsm)  # 1/16
        f3_dsm = xr3_dsm + x3_dsm

        # stage 4
        x4_opt = self.encoder_opt.stage4(xr3_opt + x3_opt)
        x4_dsm = self.encoder_dsm.stage4(xr3_dsm + x3_dsm)
        xr4_opt, xr4_dsm = self.corr4(x4_opt, x4_dsm)  # 1/32

        # Decoder
        f4 = self.fusion4(xr4_opt, xr4_dsm)
        f4 = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=False)

        f3 = self.fusion3(xr3_opt, xr3_dsm)
        d3_4 = self.decoder3(f4, f3, f3_dsm)

        f2 = self.fusion2(xr2_opt, xr2_dsm)
        d2_3 = self.decoder2(d3_4, f2, f2_dsm )

        f1 = self.fusion1(xr1_opt, xr1_dsm)
        d1_2 = self.decoder1(d2_3, f1, f1_dsm)

        out = self.out(d1_2)
        out = F.interpolate(out, size=orsize, mode='bilinear', align_corners=False)
        return out

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = GTFS_Net()

    opt_backbone = model.encoder_opt
    dsm_backbone = model.encoder_dsm

    print("OPT backbone params:", count_params(opt_backbone) / 1e6, "M")
    print("DSM backbone params:", count_params(dsm_backbone) / 1e6, "M")
    print("OPT backbone trainable params:", count_trainable_params(opt_backbone) / 1e6, "M")
    compute_model_complexity(model, [(3, 256, 256), (1, 256, 256)])



