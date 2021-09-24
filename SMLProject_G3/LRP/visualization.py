import numpy as np
import torch
import matplotlib.pyplot as plt

def project(X, output_range=(0, 1)):
    absmax   = np.abs(X).max(axis=tuple(range(1, len(X.shape))), keepdims=True)
    X       /= absmax + (absmax == 0).astype(float)
    X        = (X+1) / 2. # range [0, 1]
    X        = output_range[0] + X * (output_range[1] - output_range[0]) # range [x, y]
    return X


def heatmap(X, cmap_name="seismic"):
    cmap = plt.cm.get_cmap(cmap_name)

    if X.shape[1] in [1, 3]: X = X.permute(0, 2, 3, 1).detach().cpu().numpy()
    if isinstance(X, torch.Tensor): X = X.detach().cpu().numpy()

    shape = X.shape
    tmp = X.sum(axis=-1) # Reduce channel axis

    tmp = project(tmp, output_range=(0, 255)).astype(int)
    tmp = cmap(tmp.flatten())[:, :3].T
    tmp = tmp.T

    shape = list(shape)
    shape[-1] = 3
    return tmp.reshape(shape).astype(np.float32)


def clip_quantile(X, quantile=1):
    """Clip the values of X into the given quantile."""
    if isinstance(X, torch.Tensor): X = X.detach().cpu().numpy()
    if not isinstance(quantile, (list, tuple)):
        quantile = (quantile, 100-quantile)

    low = np.percentile(X, quantile[0])
    high = np.percentile(X, quantile[1])
    X[X < low] = low
    X[X > high] = high

    return X


def heatmap_grid(a, cmap_name="seismic", heatmap_fn=heatmap):
    a = heatmap_fn(a, cmap_name=cmap_name)
    return a

