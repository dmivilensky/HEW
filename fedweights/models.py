"""Models with flat parameter vectors, minibatch gradients and exact per-sample
gradient second moments (for oracle noise levels)."""
from __future__ import annotations
import numpy as np


def _softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


class Softmax:
    """Multinomial logistic regression with l2 regularization (convex, smooth)."""

    def __init__(self, d, C, lam=1e-3):
        self.d, self.C, self.lam = d, C, lam
        self.dim = d * C

    def init(self, rng):
        return np.zeros(self.dim)

    def _W(self, w):
        return w.reshape(self.d, self.C)

    def loss(self, w, X, y):
        P = _softmax(X @ self._W(w))
        return -np.mean(np.log(P[np.arange(len(y)), y] + 1e-12)) + 0.5 * self.lam * w @ w

    def grad(self, w, X, y):
        P = _softmax(X @ self._W(w))
        P[np.arange(len(y)), y] -= 1.0
        return (X.T @ P).ravel() / len(y) + self.lam * w

    def grad_and_persample_var(self, w, X, y):
        """Full-data gradient (with reg) and mean_j ||g_j - gbar||^2 (per-sample variance)."""
        P = _softmax(X @ self._W(w))
        P[np.arange(len(y)), y] -= 1.0
        G = (X.T @ P).ravel() / len(y)
        sq = (X ** 2).sum(1) * (P ** 2).sum(1)
        var = max(sq.mean() - G @ G, 0.0)
        return G + self.lam * w, var

    def accuracy(self, w, X, y):
        return np.mean((X @ self._W(w)).argmax(1) == y)

    def smoothness(self, X):
        # softmax CE Hessian <= 0.5 * X^T X / m  (+ lam)
        v = np.random.default_rng(0).standard_normal(X.shape[1])
        for _ in range(50):
            v = X.T @ (X @ v) / len(X)
            v /= np.linalg.norm(v)
        return 0.5 * v @ (X.T @ (X @ v)) / len(X) + self.lam


class MLP:
    """One-hidden-layer tanh network (nonconvex, smooth)."""

    def __init__(self, d, C, h=64, lam=1e-4):
        self.d, self.C, self.h, self.lam = d, C, h, lam
        self.n1 = d * h
        self.dim = d * h + (h + 1) * C

    def init(self, rng):
        W1 = rng.standard_normal((self.d, self.h)) / np.sqrt(self.d)
        W2 = rng.standard_normal((self.h + 1, self.C)) / np.sqrt(self.h)
        return np.concatenate([W1.ravel(), W2.ravel()])

    def _unpack(self, w):
        return w[: self.n1].reshape(self.d, self.h), w[self.n1:].reshape(self.h + 1, self.C)

    def _forward(self, w, X):
        W1, W2 = self._unpack(w)
        Hh = np.tanh(X @ W1)
        H1 = np.hstack([Hh, np.ones((len(X), 1))])
        return W1, W2, Hh, H1, _softmax(H1 @ W2)

    def loss(self, w, X, y):
        *_, P = self._forward(w, X)
        return -np.mean(np.log(P[np.arange(len(y)), y] + 1e-12)) + 0.5 * self.lam * w @ w

    def _backprop(self, w, X, y):
        W1, W2, Hh, H1, P = self._forward(w, X)
        dZ = P
        dZ[np.arange(len(y)), y] -= 1.0
        dA = (dZ @ W2[:-1].T) * (1 - Hh ** 2)
        return X, dA, H1, dZ

    def grad(self, w, X, y):
        X, dA, H1, dZ = self._backprop(w, X, y)
        g = np.concatenate([(X.T @ dA).ravel(), (H1.T @ dZ).ravel()]) / len(y)
        return g + self.lam * w

    def grad_and_persample_var(self, w, X, y, chunk=8192):
        G = np.zeros(self.dim)
        sq = 0.0
        for s in range(0, len(y), chunk):
            Xc, dA, H1, dZ = self._backprop(w, X[s:s + chunk], y[s:s + chunk])
            G += np.concatenate([(Xc.T @ dA).ravel(), (H1.T @ dZ).ravel()])
            sq += ((Xc ** 2).sum(1) * (dA ** 2).sum(1) + (H1 ** 2).sum(1) * (dZ ** 2).sum(1)).sum()
        G /= len(y)
        var = max(sq / len(y) - G @ G, 0.0)
        return G + self.lam * w, var

    def accuracy(self, w, X, y):
        *_, P = self._forward(w, X)
        return np.mean(P.argmax(1) == y)
