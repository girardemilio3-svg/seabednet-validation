#!/usr/bin/env python3
"""v5 backbone — hierarchical ConvNeXt/attention UNet for seabed completion.

Inputs (P x P): [depth_vis, known, gravity] — all depth-like channels are
normalized by the GRAVITY patch statistics (physics-anchored norm: continuous
everywhere -> no window seams, and mu-prediction becomes residual-over-physics).
Conditioning: learned resolution embedding (10 m / 100 m), FiLM-injected.
Heads: mu, logvar (heteroscedastic NLL, temperature-calibrated post-hoc).
Ladder configs: tiny ~9M / small ~40M / base ~150M.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

class Block(nn.Module):
    """ConvNeXt-ish block with FiLM conditioning."""
    def __init__(s, c, cond=128):
        super().__init__()
        s.dw = nn.Conv2d(c, c, 7, 1, 3, groups=c)
        s.n = nn.GroupNorm(1, c)
        s.p1 = nn.Conv2d(c, 4*c, 1); s.p2 = nn.Conv2d(4*c, c, 1)
        s.film = nn.Linear(cond, 2*c)
    def forward(s, x, e):
        g, b = s.film(e)[:, :, None, None].chunk(2, 1)
        h = s.n(s.dw(x))*(1+g)+b
        return x + s.p2(F.gelu(s.p1(h)))

class Attn(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.n = nn.GroupNorm(1, c); s.qkv = nn.Conv2d(c, 3*c, 1); s.o = nn.Conv2d(c, c, 1)
        s.heads = max(1, c//64)
    def forward(s, x):
        B, C, H, W = x.shape
        q, k, v = s.qkv(s.n(x)).reshape(B, 3, s.heads, C//s.heads, H*W).unbind(1)
        h = F.scaled_dot_product_attention(q.transpose(-2, -1), k.transpose(-2, -1),
                                           v.transpose(-2, -1))
        return x + s.o(h.transpose(-2, -1).reshape(B, C, H, W))

CONFIGS = {
    "tiny":  dict(widths=[48, 96, 192, 384],  depths=[2, 2, 4, 2]),
    "small": dict(widths=[96, 192, 384, 768], depths=[2, 3, 6, 3]),
    "base":  dict(widths=[160, 320, 640, 1280], depths=[3, 4, 12, 4]),
}

class V5(nn.Module):
    def __init__(s, size="small", cond=128):
        super().__init__()
        cfg = CONFIGS[size]; W, D = cfg["widths"], cfg["depths"]
        s.res_emb = nn.Embedding(2, cond)              # 0: 100 m, 1: 10 m
        s.cond_mlp = nn.Sequential(nn.Linear(cond, cond), nn.SiLU(), nn.Linear(cond, cond))
        s.stem = nn.Conv2d(3, W[0], 3, 1, 1)
        s.downs, s.stages, s.attns = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for li, (w, d) in enumerate(zip(W, D)):
            s.stages.append(nn.ModuleList([Block(w, cond) for _ in range(d)]))
            s.attns.append(Attn(w) if li >= 2 else nn.Identity())
            s.downs.append(nn.Conv2d(w, W[li+1], 2, 2) if li < len(W)-1 else nn.Identity())
        s.ups, s.ustages = nn.ModuleList(), nn.ModuleList()
        for li in range(len(W)-1, 0, -1):
            s.ups.append(nn.ConvTranspose2d(W[li], W[li-1], 2, 2))
            s.ustages.append(nn.ModuleList([Block(W[li-1], cond)
                                            for _ in range(max(1, D[li-1]//2))]))
        s.fuse = nn.ModuleList([nn.Conv2d(2*W[li-1], W[li-1], 1)
                                for li in range(len(W)-1, 0, -1)])
        s.head = nn.Conv2d(W[0], 2, 3, 1, 1)
    def forward(s, x, res_idx):
        e = s.cond_mlp(s.res_emb(res_idx))
        h = s.stem(x); skips = []
        for blocks, attn, down in zip(s.stages, s.attns, s.downs):
            for b in blocks: h = b(h, e)
            h = attn(h); skips.append(h); h = down(h)
        h = skips.pop()
        for up, blocks, fuse in zip(s.ups, s.ustages, s.fuse):
            h = up(h)
            h = fuse(torch.cat([h, skips.pop()], 1))
            for b in blocks: h = b(h, e)
        o = s.head(h)
        return o[:, :1], o[:, 1:].clamp(-8, 4)

def normalize(depth_vis, known, gravity):
    """Physics-anchored norm: stats from the gravity channel (present everywhere)."""
    mu = gravity.mean(dim=(-2, -1), keepdim=True)
    sd = gravity.std(dim=(-2, -1), keepdim=True) + 5.0     # floor: 5 m
    return (depth_vis - mu*known)/sd, (gravity - mu)/sd, mu, sd

if __name__ == "__main__":
    for size in CONFIGS:
        m = V5(size)
        n = sum(p.numel() for p in m.parameters())/1e6
        x = torch.randn(2, 3, 256, 256)
        mu, lv = m(x, torch.tensor([0, 1]))
        print(f"{size:6s} {n:7.1f}M params  out {tuple(mu.shape)} {tuple(lv.shape)}")
