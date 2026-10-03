"""Client side: horizon-normalized local SGD with split minibatches (noise estimate) and
optional first-step gradient sketches and SCAFFOLD control variates."""
import torch
from .server import ClientMessage


def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_flat(model, vec):
    torch.nn.utils.vector_to_parameters(vec.detach().clone(), model.parameters())


def _flat_grad(model, loss):
    gs = torch.autograd.grad(loss, list(model.parameters()))
    return torch.cat([g.reshape(-1) for g in gs])


def local_train(cid, model, x0, loader_iter, H, b, base_lr, loss_fn, sketch=None, c_i=None, c_bar=None,
                weight_decay=0.0, clip=None, cv="first"):
    """H local steps with lr = base_lr / H from the global point x0 (flattened).
    Each step draws a minibatch of size b split into two halves A,B; the step uses their average.
    Returns (message, new control variate). cv="first": c_i = minibatch gradient at x0 (SCAFFOLD option I;
    free here, because the first local step evaluates it). cv="avg": average gradient along the local path
    (option II). Option II feeds the displacement caused by c_i back into c_i and can diverge on CNNs with
    heterogeneous horizons; option I cannot."""
    set_flat(model, x0)
    lr = base_lr / H
    corr = None
    if c_i is not None:                                  # SCAFFOLD correction c_bar - c_i, computed once
        corr = (c_bar - c_i).detach().clone().contiguous()
        if not torch.isfinite(corr).all():
            raise FloatingPointError(f"client {cid}: non-finite control variate")
    v2_sum, gsum = 0.0, torch.zeros_like(x0)
    skA = skB = None
    g0 = None
    for k in range(H):
        xb, yb = next(loader_iter)
        h = max(xb.shape[0] // 2, 1)
        gA = _flat_grad(model, loss_fn(model(xb[:h]), yb[:h]))
        gB = _flat_grad(model, loss_fn(model(xb[h:2 * h]), yb[h:2 * h]))
        if weight_decay:
            wd = weight_decay * flat_params(model); gA = gA + wd; gB = gB + wd
        if k == 0 and sketch is not None:
            skA, skB = sketch(gA), sketch(gB)
        g = 0.5 * (gA + gB); d = gA - gB
        if not torch.isfinite(g).all():
            raise FloatingPointError(f"client {cid}: non-finite gradient at local step {k} "
                                     f"(H={H}, b={b}, lr={lr:.3g})")
        if clip is not None:
            gn = float(g.norm())
            if gn > clip:
                g = g * (clip / gn)
        v2_sum += float(2 * h * (d @ d) / 4.0)           # unbiased per-sample variance estimate
        gsum += g
        if k == 0:
            g0 = g.detach().clone()                      # gradient at the server point x0
        if corr is not None:
            g = g + corr
        with torch.no_grad():
            set_flat(model, flat_params(model) - lr * g)
    delta = flat_params(model) - x0
    return ClientMessage(cid=cid, delta=delta, v2=v2_sum / H, H=H, b=b, skA=skA, skB=skB), (g0 if cv == "first" else gsum / H)
