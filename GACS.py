import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from src.model.GTCM import SGAM, GTSP
from src.model.FSCM import LCCG

class GACS(nn.Module):
    """
    Geometry-Aware Cross-Scale Refinement Module
    """
    def __init__(self, in_high:int, in_low:int, out_ch:Optional[int]=None,
                 reduction:int=8, use_geom:bool=True):
        super().__init__()
        if out_ch is None:
            out_ch = in_low
        self.out_ch = out_ch
        self.use_geom = use_geom

        self.proj_high = nn.Conv2d(in_high, out_ch, 1, bias=False)
        self.proj_low  = nn.Conv2d(in_low,  out_ch, 1, bias=False)

        # fusion (channel mix + local interaction)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # gating (unchanged)
        self.gate = LCCG(out_ch)

        # geom guidance
        if self.use_geom:
            self.sagm = SGAM(out_ch)                   # unchanged
            self.geom_proj = nn.Conv2d(1, out_ch, 1, bias=False)
        else:
            self.sagm = None
            self.geom_proj = None

        # output refine
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, high_feat: torch.Tensor, low_feat: torch.Tensor,
                geom_guide: Optional[torch.Tensor] = None):
        high_up = F.interpolate(high_feat, size=low_feat.shape[2:], mode='bilinear', align_corners=False)
        H = self.proj_high(high_up)
        L = self.proj_low(low_feat)

        x = self.fuse(torch.cat([H, L], dim=1))
        x = self.gate(x)

        if self.sagm is not None and geom_guide is not None:
            if geom_guide.shape[2:] != x.shape[2:]:
                geom_guide = F.interpolate(geom_guide, size=x.shape[2:], mode='bilinear', align_corners=False)
            geom_prior = GTSP(geom_guide)         # [B,1,H,W]
            geom = self.geom_proj(geom_prior)     # [B,out_ch,H,W] for SGAM
            x = self.sagm(x, geom)

        out = self.out_conv(x) + L
        return out
