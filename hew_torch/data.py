"""Datasets (downloaded automatically via torchvision), Dirichlet label partitions and systems heterogeneity.
"""
import os
import numpy as np
import torch

_STATS = {"mnist": ((0.1307,), (0.3081,)), "fmnist": ((0.2860,), (0.3530,)),
          "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
          "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))}


class TensorData:
    """Images X (N, C, H, W) float32 and labels y (N,) int64 on `device`."""

    def __init__(self, X, y, device, augment=False):
        self.X, self.y, self.augment = X.to(device), y.to(device), augment
        self.device = device

    def __len__(self):
        return len(self.y)

    def batch(self, idx, train=True):
        xb, yb = self.X[idx], self.y[idx]
        if train and self.augment:
            xb = _augment(xb)
        return xb, yb


def _augment(x):
    """Random crop with 4-pixel zero padding and random horizontal flip, batched on device."""
    n, c, h, w = x.shape
    xp = torch.nn.functional.pad(x, (4, 4, 4, 4))
    dx = torch.randint(0, 9, (n,), device=x.device); dy = torch.randint(0, 9, (n,), device=x.device)
    ar_h = torch.arange(h, device=x.device); ar_w = torch.arange(w, device=x.device)
    rows = (dy[:, None] + ar_h[None, :])                       # n x h
    cols = (dx[:, None] + ar_w[None, :])                       # n x w
    out = xp[torch.arange(n, device=x.device)[:, None, None, None], torch.arange(c, device=x.device)[None, :, None, None],
             rows[:, None, :, None], cols[:, None, None, :]]
    flip = torch.rand(n, device=x.device) < 0.5
    out[flip] = out[flip].flip(-1)
    return out


def _to_tensor_images(data, name):
    """torchvision stores MNIST-type data as uint8 (N,28,28) tensors and CIFAR as uint8 (N,32,32,3) arrays."""
    x = torch.as_tensor(np.asarray(data))
    if x.dim() == 3:
        x = x[:, None]
    elif x.dim() == 4 and x.shape[-1] in (1, 3):
        x = x.permute(0, 3, 1, 2)
    x = x.float() / 255.0
    mean, std = _STATS[name]
    mean = torch.tensor(mean).view(1, -1, 1, 1); std = torch.tensor(std).view(1, -1, 1, 1)
    return (x - mean) / std


def load(name, root="./data", device="cpu"):
    """Returns (train, test, num_classes). name in {mnist, fmnist, cifar10, cifar100} (downloaded on first use)
    or a path to an .npz file with arrays X (N x ...) and y (N,)."""
    if name.endswith(".npz"):
        d = np.load(name); X = torch.tensor(d["X"], dtype=torch.float32); y = torch.tensor(d["y"]).long()
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        if X.dim() == 2 and int(round(X.shape[1] ** 0.5)) ** 2 == X.shape[1]:
            s = int(round(X.shape[1] ** 0.5)); X = X.view(-1, 1, s, s)
        perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(0)); nte = len(y) // 7
        te, tr = perm[:nte], perm[nte:]
        return TensorData(X[tr], y[tr], device), TensorData(X[te], y[te], device), int(y.max()) + 1
    import torchvision.datasets as D
    os.makedirs(root, exist_ok=True)
    cls = {"mnist": D.MNIST, "fmnist": D.FashionMNIST, "cifar10": D.CIFAR10, "cifar100": D.CIFAR100}[name]
    tr = cls(root, train=True, download=True); te = cls(root, train=False, download=True)
    Xtr, Xte = _to_tensor_images(tr.data, name), _to_tensor_images(te.data, name)
    ytr, yte = torch.as_tensor(np.asarray(tr.targets)).long(), torch.as_tensor(np.asarray(te.targets)).long()
    aug = name.startswith("cifar")
    C = int(max(ytr.max(), yte.max())) + 1
    return TensorData(Xtr, ytr, device, augment=aug), TensorData(Xte, yte, device), C


def labels_of(ds):
    return ds.y.cpu().numpy()


def dirichlet_partition(y, n, alpha, seed=0, min_size=20):
    rng = np.random.default_rng(seed)
    if not np.isfinite(alpha):
        return [np.sort(a) for a in np.array_split(rng.permutation(len(y)), n)]
    C = int(y.max()) + 1
    for _ in range(1000):
        idx = [[] for _ in range(n)]
        for c in range(C):
            ic = rng.permutation(np.where(y == c)[0]); prop = rng.dirichlet(alpha * np.ones(n))
            for i, part in enumerate(np.split(ic, (np.cumsum(prop) * len(ic)).astype(int)[:-1])):
                idx[i].extend(part.tolist())
        if min(len(a) for a in idx) >= min_size:
            return [np.sort(np.array(a)) for a in idx]
    raise RuntimeError("partition failed; increase alpha or decrease n")


class Systems:
    """Per-client horizons and batch sizes; optional per-round stragglers and partial participation."""

    def __init__(self, n, H_set=(1, 2, 4, 8, 16), b_set=(8, 16, 32, 64, 128), participation=1.0,
                 stragglers=False, seed=0):
        self.rng = np.random.default_rng(seed); self.n = n
        self.H = self.rng.choice(H_set, n); self.b = self.rng.choice(b_set, n)
        self.participation, self.stragglers = participation, stragglers

    def draw(self):
        H = self.H.copy()
        if self.stragglers:
            H = np.maximum(1, (H * self.rng.choice([0.5, 1.0, 1.5], self.n)).astype(int))
        k = max(1, int(round(self.participation * self.n)))
        act = np.sort(self.rng.choice(self.n, k, replace=False)) if k < self.n else np.arange(self.n)
        return act, H, self.b.copy()


class ClientSampler:
    """Infinite sampling with replacement from a client's shard (as in the analysis), on the device."""

    def __init__(self, ds: TensorData, indices, batch, seed):
        self.ds, self.batch = ds, batch
        self.idx = torch.as_tensor(indices, device=ds.device)
        self.g = torch.Generator(device="cpu").manual_seed(seed)

    def __iter__(self):
        return self

    def __next__(self):
        j = torch.randint(0, len(self.idx), (self.batch,), generator=self.g).to(self.idx.device)
        return self.ds.batch(self.idx[j], train=True)
