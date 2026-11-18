"""Noise-level estimation with the authors' pretrained VI-non-IID network.
Re-implements the math of IPS-RNA/utils/estimate_sigma_using_VInonIID.py without its
hardcoded paths, DataParallel/.cuda() calls or global state."""
import math

import numpy as np
import torch

from common import VENDOR, use_vendor

WEIGHTS = VENDOR / "IPS-RNA" / "utils" / "TrainedModel" / "Non-IID-Unet" / "model_state_10"
LOG_MIN, LOG_MAX = math.log(1e-8), math.log(1e4)      # clip bounds used by the authors


def load_vdn(device, weights=WEIGHTS):
    use_vendor("IPS-RNA")
    from utils.networks import VDN
    # weights_only=False: newer PyTorch defaults to True, which can reject older checkpoints.
    sd = torch.load(weights, map_location="cpu", weights_only=False)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}  # saved from DataParallel
    net = VDN(1, wf=64, dep_U=4)
    net.load_state_dict(sd, strict=True)
    return net.to(device).eval()


@torch.no_grad()
def estimate_sigma_map(net, y, device):
    """y: (H, W) array. Returns the per-pixel noise std map (H, W) as a float numpy array."""
    x = torch.from_numpy(np.asarray(y, dtype=np.float32))[None, None].to(device)
    H, W = x.shape[2:]
    m = 2 ** 4                                          # authors pad to a multiple of 2**(dep_U-1), dep_U=5
    xp = torch.nn.functional.pad(x, (0, (-W) % m, 0, (-H) % m), mode="reflect")
    phi = net(xp, "sigma")[:, :, :H, :W].clamp(LOG_MIN, LOG_MAX)
    alpha, beta = torch.exp(phi[:, :1]), torch.exp(phi[:, 1:])
    return torch.sqrt(beta / (alpha + 1)).squeeze().cpu().numpy()
