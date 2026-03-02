import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# --------- 小工具：频域变换（complex ops） ----------
def fft2_real(x):
    """
    2D傅里叶变换 空间域（时域）-->频域
    :param x:
    :return:complex(复数) 编码了幅度和相位
    """
    # x: [B,C,H,W] -> returns complex tensor [B,C, H, W//2+1] as torch.complex64
    return torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')

def ifft2_real(X, s):
    """
    2D傅里叶逆变换
    :param X:
    :param s:
    :return:
    """
    # X: complex [B,C,H, Wf] -> return real spatial [B,C,H,W] using irfft2
    # s = (H,W)
    return torch.fft.irfft2(X, s=s, dim=(-2, -1), norm='ortho')

def mag_phase(X):
    """
    将复数频域张量分解为两个实数分量：幅度谱和相位谱
    :param X:
    :return:（幅度，相位）
    """
    # X: complex -> return (mag, phase) real tensors
    mag = torch.abs(X)
    phase = torch.angle(X)
    return mag, phase

def recompose_from_mag_phase(mag, phase):
    """
    将分离的幅度谱和相位谱重新组合成一个复数频域张量。
    :param mag:
    :param phase:
    :return:complex
    """
    # mag, phase -> complex
    return torch.polar(mag, phase)  # mag * exp(i*phase)

# --------- 频谱掩码网络（对幅值作可学习门控） ----------
class SFG(nn.Module):
    """
    Spectral Filtering Gate
    Learnable spectral mask applied per-channel or channel-shared.
    We parameterize in log-domain for stability: mask = sigmoid(alpha * log(mag + eps) + beta)
    Or simpler: mask = sigmoid(conv_on_mag)
    可学习频域掩码模块
    根据输入特征的幅度谱，动态生成一个自适应的频域滤波掩码,用于在频域中有选择地增强或抑制特定的频率成分。
    """
    def __init__(self, in_ch, hidden=32, per_channel=True):
        super().__init__()
        self.per_channel = per_channel
        # We'll operate on magnitude map spatial dims (Hf x Wf). Use 1x1 conv across channels in mag-space
        # Use a small conv block to predict a mask in freq domain (real). Using real conv on magnitude.
        ch = in_ch if per_channel else 1
        self.net = nn.Sequential(
            nn.Conv2d(ch, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, ch, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, mag):
        # mag: [B, C, Hf, Wf] (real)
        # 通道独立：每个通道学习独立的频域掩码，适用于不同通道承载不同频率信息的场景
        if not self.per_channel:
            mag_in = mag.mean(dim=1, keepdim=True)  # [B,1,Hf,Wf]
        # 通道共享：所有通道共享同一个频域掩码，适用于通道间频率特性相似的场景
        else:
            mag_in = mag  # [B,C,Hf,Wf]
        mask = self.net(mag_in)  # [B, ch, Hf, Wf]
        if not self.per_channel:
            mask = mask.repeat(1, mag.shape[1], 1, 1)
        return mask  # [B,C,Hf,Wf] 0-1

# --------- 空间域引导模块（轻量） ----------
class SSRB(nn.Module):
    """
    Spatial Structure Refinement Block
    空间域特征精炼模块
    """
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class CSMI(nn.Module):
    """
    Cross-Spectral Mask Injection(CSMI)
    """
    def __init__(self, channels, hidden=32, per_channel=False):
        super().__init__()
        self.mask_net = SFG(
            channels, hidden=hidden, per_channel=per_channel
        )

    def forward(self, src_mag, tgt_mag):
        """
        src_mag: source modality magnitude (used to generate mask)
        tgt_mag: target modality magnitude (to be modulated)
        """
        mask = self.mask_net(src_mag)
        tgt_mag_fused = tgt_mag * (1.0 + torch.tanh(mask))
        return tgt_mag_fused

class LCCG(nn.Module):
    """
    Lightweight Cross-Coupled Global–Local Gating
    """
    def __init__(self, channels, reduction=8, alpha=0.5):
        super().__init__()
        mid = max(4, channels // reduction)
        self.alpha = alpha

        self.ch_global = nn.Sequential(
            nn.Conv2d(channels + 1, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid()
        )

        self.sp_global = nn.Sequential(
            nn.Conv2d(channels * 2, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1),
            nn.Sigmoid()
        )

        # local structural refinement
        self.struct = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        sp_ctx = torch.mean(x, dim=1, keepdim=True)  # [B,1,H,W]
        ch_ctx = F.adaptive_avg_pool2d(x, 1).expand_as(x)  # [B,C,H,W]

        ch_w = self.ch_global(torch.cat([x, sp_ctx], dim=1))  # channel gate
        sp_w = self.sp_global(torch.cat([x, ch_ctx], dim=1))  # spatial gate

        x_gated = x * (1 + self.alpha * ch_w) * (1 + self.alpha * sp_w)
        x_ref   = self.struct(x_gated)

        out = x_gated + x_ref
        return out

# --------- FS-CMFM 主体 ----------
class FSCM(nn.Module):
    """
    Frequency-Spatial Cross-Modality Fusion Module.
    """
    def __init__(self, channels, token_ds=1, spectral_per_channel=False, fusion_hidden=None):
        super().__init__()
        self.channels = channels
        self.token_ds = token_ds
        self.spectral_per_channel = spectral_per_channel
        if fusion_hidden is None:
            fusion_hidden = max(32, channels // 2)

        # small proj before FFT to reduce channels (for efficiency)
        self.pre_proj = nn.Conv2d(channels, fusion_hidden, 1, bias=False)
        self.pre_proj_back = nn.Conv2d(fusion_hidden, channels, 1, bias=False)

        # spectral masks for each direction
        self.csmi_d2o = CSMI(fusion_hidden, hidden=32, per_channel=spectral_per_channel)
        self.csmi_o2d = CSMI(fusion_hidden, hidden=32, per_channel=spectral_per_channel)

        # cross-spectral linear mixing (1x1 conv in freq-magnitude domain)
        self.spec_mix = nn.Conv2d(fusion_hidden * 2, fusion_hidden, 1, bias=False)

        # spatial refinement after inverse FFT
        self.refine = SSRB(channels)

        # fusion gates (spatial & channel) for final merging
        self.asg = LCCG(channels)

        # final fusion conv
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, F_opt, F_dsm):
        """
        Inputs: F_opt, F_dsm: [B,C,H,W]
        Outputs: F_fuse: [B,C,H,W] (fused)
        """
        B, C, H, W = F_opt.shape
        # optionally downsample for FFT cost reduction
        if self.token_ds > 1:
            opt_ds = F.avg_pool2d(F_opt, kernel_size=self.token_ds, stride=self.token_ds)
            dsm_ds = F.avg_pool2d(F_dsm, kernel_size=self.token_ds, stride=self.token_ds)
        else:
            opt_ds = F_opt
            dsm_ds = F_dsm

        # channel reduction for spectral ops
        opt_red = self.pre_proj(opt_ds)  # [B, F, Hs, Ws]
        dsm_red = self.pre_proj(dsm_ds)

        # spectral transform (complex)
        opt_spec = fft2_real(opt_red)  # complex [B, F, Hf, Wf]
        dsm_spec = fft2_real(dsm_red)

        # magnitude & phase
        opt_mag, opt_phase = mag_phase(opt_spec)
        dsm_mag, dsm_phase = mag_phase(dsm_spec)

        opt_mag_fused = self.csmi_d2o(dsm_mag, opt_mag)
        dsm_mag_fused = self.csmi_o2d(opt_mag, dsm_mag)

        # optionally combine cross-spectral info (concat mags then mix)
        # we'll fuse mags of both to obtain a cross-spectral representation
        combined_mag = torch.cat([opt_mag_fused, dsm_mag_fused], dim=1)  # channel dim doubled (F)
        combined_mag_proj = self.spec_mix(combined_mag)  # [B, F, Hf, Wf]

        # reconstruct complex spectra (use phases from opt for opt-target and dsm for dsm-target)
        opt_spec_new = recompose_from_mag_phase(combined_mag_proj, opt_phase)
        dsm_spec_new = recompose_from_mag_phase(combined_mag_proj, dsm_phase)

        # inverse transform to spatial
        # compute original ds spatial size
        Hs, Ws = opt_red.shape[2], opt_red.shape[3]
        opt_spatial = ifft2_real(opt_spec_new, s=(Hs, Ws)).real  # [B, F, Hs, Ws]
        dsm_spatial = ifft2_real(dsm_spec_new, s=(Hs, Ws)).real

        # project back to original channel dim
        opt_back = self.pre_proj_back(opt_spatial)  # [B, C, Hs, Ws]
        dsm_back = self.pre_proj_back(dsm_spatial)

        # if downsampled, upsample back to original resolution
        if self.token_ds > 1:
            opt_back = F.interpolate(opt_back, size=(H, W), mode='bilinear', align_corners=False)
            dsm_back = F.interpolate(dsm_back, size=(H, W), mode='bilinear', align_corners=False)

        # spatial refine
        opt_ref = self.refine(opt_back)
        dsm_ref = self.refine(dsm_back)

        # final merge: [opt_ref, dsm_ref, avg(originals)]
        avg_orig = 0.5 * (F_opt + F_dsm)
        merged = torch.cat([opt_ref, dsm_ref, avg_orig], dim=1)
        fused = self.final_conv(merged)

        # gating
        out = self.asg(fused)

        return out
