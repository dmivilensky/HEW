"""Datasets, federated partitions and systems-heterogeneity profiles."""
from __future__ import annotations
import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def ensure_npz(name: str) -> str:
    """Return the path of data/<name>.npz, creating it on first use by downloading the dataset with
    torchvision (train and test splits concatenated; X in [0,1], shape N x 784)."""
    path = os.path.join(DATA_DIR, f"{name}.npz")
    if os.path.exists(path):
        return path
    os.makedirs(DATA_DIR, exist_ok=True)
    import torchvision.datasets as D
    cls = {"fmnist": D.FashionMNIST, "mnist": D.MNIST}[name]
    raw = os.path.join(DATA_DIR, "torchvision")
    tr, te = cls(raw, train=True, download=True), cls(raw, train=False, download=True)
    X = np.concatenate([tr.data.numpy(), te.data.numpy()]).reshape(-1, 784).astype(np.float32) / 255.0
    y = np.concatenate([tr.targets.numpy(), te.targets.numpy()]).astype(np.int64)
    np.savez_compressed(path, X=X, y=y)
    return path


def load_dataset(name: str, pca: int | None = None, test_frac: float = 1 / 7, seed: int = 0):
    """Return (Xtr, ytr, Xte, yte). Features standardized; bias column appended."""
    d = np.load(ensure_npz(name))
    X, y = d["X"].astype(np.float64), d["y"].astype(np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    nte = int(len(y) * test_frac)
    Xte, yte, Xtr, ytr = X[:nte], y[:nte], X[nte:], y[nte:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    if pca:
        U, S, Vt = np.linalg.svd(Xtr[:20000], full_matrices=False)
        P = Vt[:pca].T
        Xtr, Xte = Xtr @ P, Xte @ P
        s = Xtr.std(0) + 1e-6
        Xtr, Xte = Xtr / s, Xte / s
    # normalize rows to unit average squared norm (keeps L ~ O(1))
    scale = np.sqrt((Xtr ** 2).sum(1).mean())
    Xtr, Xte = Xtr / scale, Xte / scale
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
    return Xtr, ytr, Xte, yte


def dirichlet_partition(y: np.ndarray, n: int, alpha: float, rng, min_size: int = 50):
    """Label-skew partition. alpha=inf -> iid equal split."""
    N = len(y)
    if not np.isfinite(alpha):
        perm = rng.permutation(N)
        return [np.sort(p) for p in np.array_split(perm, n)]
    C = int(y.max()) + 1
    for _ in range(1000):
        idx = [[] for _ in range(n)]
        for c in range(C):
            ic = rng.permutation(np.where(y == c)[0])
            prop = rng.dirichlet(alpha * np.ones(n))
            cuts = (np.cumsum(prop) * len(ic)).astype(int)[:-1]
            for i, part in enumerate(np.split(ic, cuts)):
                idx[i].extend(part.tolist())
        sizes = np.array([len(a) for a in idx])
        if sizes.min() >= min_size:
            return [np.sort(np.array(a)) for a in idx]
    raise RuntimeError("could not satisfy min_size")


class Systems:
    """Per-client horizons H_i and minibatch sizes b_i; optionally re-drawn each round
    (stragglers) and random partial participation. Independent of gradient noise."""

    def __init__(self, n, rng, H_set=(1, 2, 4, 8, 16), b_set=(32,), redraw=False,
                 participation=1.0, H_fixed=None, b_fixed=None):
        self.n, self.rng = n, rng
        self.H_set, self.b_set, self.redraw, self.participation = H_set, b_set, redraw, participation
        self.H = np.array(H_fixed) if H_fixed is not None else rng.choice(H_set, n)
        self.b = np.array(b_fixed) if b_fixed is not None else rng.choice(b_set, n)

    def draw(self):
        if self.redraw:
            H = self.rng.choice(self.H_set, self.n)
            # keep each client's "speed class": multiply base horizon by random factor
            H = np.maximum(1, (self.H * self.rng.choice([0.5, 1, 1.5], self.n)).astype(int))
        else:
            H = self.H.copy()
        if self.participation < 1.0:
            k = max(1, int(round(self.participation * self.n)))
            active = np.sort(self.rng.choice(self.n, k, replace=False))
        else:
            active = np.arange(self.n)
        return H, self.b.copy(), active
