"""Unbiased linear sketches for inner-product estimation: E<Sa, Sb> = <a, b>."""
import torch


class CountSketch:
    """CountSketch S: R^d -> R^k with hash h and signs s (memory O(d)); Var<Sa,Sb> <= 2|a|^2|b|^2/k."""

    def __init__(self, d, k, seed=0, device="cpu"):
        g = torch.Generator().manual_seed(seed)
        self.h = torch.randint(0, k, (d,), generator=g).to(device)
        self.s = (torch.randint(0, 2, (d,), generator=g) * 2 - 1).to(torch.float32).to(device)
        self.k, self.d = k, d

    def __call__(self, v):
        out = torch.zeros(self.k, device=v.device, dtype=v.dtype)
        return out.index_add_(0, self.h, v * self.s)


class GaussianSketch:
    """Dense JL sketch with N(0, 1/k) entries (small models only)."""

    def __init__(self, d, k, seed=0, device="cpu"):
        g = torch.Generator().manual_seed(seed)
        self.S = (torch.randn(k, d, generator=g) / k ** 0.5).to(device)

    def __call__(self, v):
        return self.S @ v


def make_sketch(kind, d, k, seed, device):
    if kind == "gaussian":
        return GaussianSketch(d, k, seed, device)
    return CountSketch(d, k, seed, device)
