import torch.nn as nn
import torch.nn.functional as F


class MaxPool2d(nn.Module):
    """Max-pooling with the same forward pass as nn.MaxPool2d. Computing it with return_indices=True makes it
    twice differentiable in every PyTorch version, which the Hessian-vector products of theta = lr * lambda_max need."""

    def __init__(self, k):
        super().__init__(); self.k = k

    def forward(self, x):
        return F.max_pool2d(x, self.k, return_indices=True)[0]
import torchvision


def make_model(name, in_shape, num_classes):
    if name == "logreg":
        d = 1
        for s in in_shape:
            d *= s
        return nn.Sequential(nn.Flatten(), nn.Linear(d, num_classes))
    if name == "mlp":
        d = 1
        for s in in_shape:
            d *= s
        return nn.Sequential(nn.Flatten(), nn.Linear(d, 200), nn.ReLU(), nn.Linear(200, 200), nn.ReLU(), nn.Linear(200, num_classes))
    if name == "cnn":                      # LeNet-style, for (F)MNIST / CIFAR
        c = in_shape[0]; s = in_shape[1]
        return nn.Sequential(nn.Conv2d(c, 32, 5, padding=2), nn.ReLU(), MaxPool2d(2),
                             nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), MaxPool2d(2), nn.Flatten(),
                             nn.Linear(64 * (s // 4) ** 2, 512), nn.ReLU(), nn.Linear(512, num_classes))
    if name == "resnet18":                 # GroupNorm instead of BatchNorm (standard in FL)
        m = torchvision.models.resnet18(num_classes=num_classes, norm_layer=lambda ch: nn.GroupNorm(2, ch))
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); m.maxpool = nn.Identity()
        return m
    raise ValueError(name)
