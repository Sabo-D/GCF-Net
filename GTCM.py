import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sobel_kernel(channel=1, device=None, dtype=None):
    # returns conv2d weight for sobel magnitude (applied per-channel)
    kx = torch.tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=dtype, device=device)
    ky = torch.tensor([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=dtype, device=device)
    kx = kx.view(1, 1, 3, 3).repeat(channel, 1, 1, 1)
    ky = ky.view(1, 1, 3, 3).repeat(channel, 1, 1, 1)
    return kx, ky

def GTSP(x):
    """
    Geometry–Texture Structural Prior Extraction
    :param x:
    :return:
    """
    # x: [B, C, H, W] -> compute per-channel sobel magnitude then mean over channel
    B, C, H, W = x.shape
    device = x.device
    dtype = x.dtype
    kx, ky = get_sobel_kernel(channel=C, device=device, dtype=dtype)
    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
    # collapse channels:
    return mag.mean(dim=1, keepdim=True)  # [B,1,H,W]

class GTCS(nn.Module):
    """
    Geometry–Texture Aware Channel Selection (GTCS)
    Soft Top-k Guided Channel Gating
    """
    def __init__(self, channels, reduction=8, keep_ratio=0.7, tau=0.1):
        super().__init__()
        self.channels = channels
        self.keep_ratio = keep_ratio
        self.tau = tau

        mid = max(4, channels // reduction)

        # ----- feature channel score -----
        self.fc1 = nn.Conv2d(channels, mid, 1, bias=True)
        self.fc2 = nn.Conv2d(mid, channels, 1, bias=True)

        # ----- prior channel score -----
        self.prior_fc1 = nn.Conv2d(1, mid, 1, bias=True)
        self.prior_fc2 = nn.Conv2d(mid, channels, 1, bias=True)

        # ----- local refinement -----
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def soft_topk_gate(self, score):
        """
        score: [B, C] 第 b 个样本第 c 个通道的重要性
        return: [B, C, 1, 1]
        """
        B, C = score.shape
        k = max(1, int(self.keep_ratio * C))   # 保留通道数

        # top-k threshold
        topk_val, _ = torch.topk(score, k, dim=1)  # 获取score前k个最大值的具体数值
        thr = topk_val[:, -1].unsqueeze(1)         # 取第k大的值作为阈值门槛 [B, 1]

        # soft top-k gating
        gate = torch.sigmoid((score - thr) / self.tau)
        return gate.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x, prior_map=None):
        B, C, H, W = x.shape

        # ======= main channel score =======
        z = F.adaptive_avg_pool2d(x, 1)          # [B,C,1,1]
        z = F.relu(self.fc1(z), inplace=True)
        z = self.fc2(z)       # [B C 1 1]
        score = z.view(B, C)  # [B,C]

        gate_main = self.soft_topk_gate(score)   # [B,C,1,1]

        # ======= prior-guided modulation =======
        if prior_map is not None:
            p = F.adaptive_avg_pool2d(prior_map, 1)   # [B,1,1,1]
            p = F.relu(self.prior_fc1(p), inplace=True)
            p = self.prior_fc2(p)
            gate_prior = torch.sigmoid(p)             # [B,C,1,1]

            # prior confidence re-weighting
            gate = gate_main * (1.0 + gate_prior)
        else:
            gate = gate_main

        x_weighted = x * gate
        out = self.local(x_weighted)
        return out

class PGCA(nn.Module):
    """
    Prior-Guided Cross Attention (PGCA)
    """
    def __init__(self, embed_dim, num_heads=4, token_ds=4):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.token_ds = token_ds
        # use PyTorch MultiheadAttention with batch_first=True
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # small proj to convert geometry scalar -> bias vector per head
        self.geom_proj = nn.Linear(1, embed_dim)

    def spatial_downsample(self, x):
        # x: [B, C, H, W]
        if self.token_ds == 1:
            return x
        # use avg pooling to reduce tokens while preserving numeric stability
        return F.avg_pool2d(x, kernel_size=self.token_ds, stride=self.token_ds, ceil_mode=False)

    def flatten_for_attn(self, x):
        # x: [B,C,H,W] -> [B, N, C]
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2)  # [B, H*W, C]

    def forward(self, q_feat, kv_feat, kv_geom_prior=None):
        """
        q_feat: [B,C,H,W]  (query from modality A)
        kv_feat: [B,C,H,W] (key/value from modality B, possibly selected/chained)
        kv_geom_prior: [B,1,H,W] (e.g., sobel magnitude of DSM) optional
        """
        # downsample both to same token grid
        q_ds = self.spatial_downsample(q_feat)   # [B,C,H1,W1]
        kv_ds = self.spatial_downsample(kv_feat) # [B,C,H1,W1]
        if kv_geom_prior is not None and self.token_ds != 1:
            geom_ds = F.avg_pool2d(kv_geom_prior, kernel_size=self.token_ds, stride=self.token_ds)
        else:
            geom_ds = kv_geom_prior  # can be None or same size

        Q = self.flatten_for_attn(q_ds)   # [B, N, C]
        K = self.flatten_for_attn(kv_ds)
        V = K

        # geometry bias: project per-token scalar to embedding and add to K
        if geom_ds is not None:
            # geom_ds: [B,1,H1,W1] -> [B,N,1]
            geom_flat = geom_ds.flatten(2).transpose(1, 2)  # [B, N, 1]
            geom_vec = self.geom_proj(geom_flat)           # [B, N, C]
            K = K + geom_vec  # bias keys (lightweight)
        # MultiheadAttention expects inputs: (B, L, E)
        attn_out, attn_weights = self.mha(Q, K, V, need_weights=False)
        # reshape back to spatial shape of q_ds
        B, N, C = attn_out.shape
        # compute H1,W1 from q_ds
        H1 = q_ds.shape[2]
        W1 = q_ds.shape[3]
        out = attn_out.transpose(1, 2).view(B, C, H1, W1)
        # optionally upsample back to original q_feat spatial resolution
        if self.token_ds != 1:
            out = F.interpolate(out, size=(q_feat.shape[2], q_feat.shape[3]), mode='bilinear', align_corners=False)
        return out

class SGAM(nn.Module):
    """
    Spatially-Guided Affine Modulation
    """
    def __init__(self, channels, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = max(16, channels // 4)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, kernel_size=3, padding=1)
        )

    def forward(self, fused_feat, orig_feat):
        # orig_feat: [B,C,H,W] -> produce gamma,beta of same spatial size
        param = self.net(orig_feat)  # [B,2C,H,W]
        gamma, beta = torch.chunk(param, 2, dim=1)
        gamma = torch.tanh(gamma)
        return fused_feat * (1.0 + gamma) + beta

class GTCM(nn.Module):
    """
    Geometry–Texture Guided Cross-Modality Correction Module
    """
    def __init__(self, channels, reduction=8, keep_ratio=0.7, num_heads=4, token_ds=4, use_geom_prior=True):
        super().__init__()
        self.channels = channels
        self.keep_ratio = keep_ratio
        self.token_ds = token_ds
        self.use_geom_prior = use_geom_prior

        # soft channel select + expand for both modalities
        self.select_opt = GTCS(channels, reduction=reduction, keep_ratio=keep_ratio)
        self.select_dsm = GTCS(channels, reduction=reduction, keep_ratio=keep_ratio)

        # cross attention modules (bidirectional)
        self.cross_opt = PGCA(channels, num_heads=num_heads, token_ds=token_ds)
        self.cross_dsm = PGCA(channels, num_heads=num_heads, token_ds=token_ds)

        # spatial geometric residual blocks
        self.gam_opt = SGAM(channels)
        self.gam_dsm = SGAM(channels)

    def forward(self, F_opt, F_dsm):
        """
        Input:
            F_opt: [B, C, H, W] optical features (stage k)
            F_dsm: [B, C, H, W] DSM features (stage k)
        Output:
            F_opt_corr, F_dsm_corr: corrected features (same shapes)
        """
        # 1) compute lightweight priors
        if self.use_geom_prior:
            # DSM geometry prior: sobel magnitude
            dsm_geom = GTSP(F_dsm)  # [B,1,H,W]
            # optical texture prior: laplacian-like (use sobel on optical)
            opt_tex = GTSP(F_opt)   # [B,1,H,W]
        else:
            dsm_geom = None
            opt_tex = None

        # 2) channel soft select (use priors to modulate gates)
        F_opt_sel = self.select_opt(F_opt, prior_map=dsm_geom)   # optical selected to help DSM
        F_dsm_sel = self.select_dsm(F_dsm, prior_map=opt_tex)    # dsm selected to help optical

        # 3) cross attention (reduced token count, geom bias in keys)
        F_opt_cross = self.cross_opt(F_opt, F_dsm_sel, kv_geom_prior=dsm_geom)
        F_dsm_cross = self.cross_dsm(F_dsm, F_opt_sel, kv_geom_prior=opt_tex)

        # 4) spatial geometric residual correction
        F_opt_corr = self.gam_opt(F_opt_cross, F_opt)
        F_dsm_corr = self.gam_dsm(F_dsm_cross, F_dsm)

        return F_opt_corr, F_dsm_corr
