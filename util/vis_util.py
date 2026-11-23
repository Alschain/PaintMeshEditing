import numpy as np

try:
    import torch
except ImportError:
    torch = None


def vis_normal_map(normal):
    """Visualize a normal map (H, W, 3) into [0, 255] without mutating input.

    Valid unit normals in [-1, 1] are mapped cleanly to [0, 255], while zero
    background pixels remain pure black (0, 0, 0).
    """
    if torch is not None and isinstance(normal, torch.Tensor):
        n = normal.detach().clone().float()
        valid = torch.norm(n, dim=-1, keepdim=True) > 1e-4
        out = torch.zeros_like(n)
        if valid.any():
            n_unit = torch.nn.functional.normalize(n, p=2, dim=-1, eps=1e-6)
            out = ((n_unit * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) * valid.float()
        return out
    else:
        n = np.array(normal, dtype=np.float32, copy=True)
        norms = np.linalg.norm(n, axis=-1, keepdims=True)
        valid = norms > 1e-4
        out = np.zeros_like(n)
        if np.any(valid):
            n_unit = n / np.maximum(norms, 1e-6)
            out = np.clip((n_unit * 0.5 + 0.5) * 255.0, 0.0, 255.0) * valid.astype(np.float32)
        return out.astype(np.uint8)


def vis_depth_map(depth):
    """Visualize a depth map (H, W) or (H, W, 1) into 3-channel [0, 255]."""
    if torch is not None and isinstance(depth, torch.Tensor):
        d = depth.detach().float().clone()
        if d.ndim == 3 and d.shape[-1] == 1:
            d = d[..., 0]
        valid = d > 0
        out = torch.zeros_like(d)
        if valid.any():
            d_min = d[valid].min()
            d_max = d[valid].max()
            denom = d_max - d_min
            if denom > 1e-6:
                out[valid] = 1.0 - (d[valid] - d_min) / denom
            else:
                out[valid] = 1.0
        out = (out.unsqueeze(-1).repeat(1, 1, 3) * 255.0).clamp(0.0, 255.0)
        return out
    else:
        d = np.array(depth, dtype=np.float32, copy=True)
        if d.ndim == 3 and d.shape[-1] == 1:
            d = d[..., 0]
        valid = d > 0
        out = np.zeros_like(d)
        if np.any(valid):
            d_min = float(d[valid].min())
            d_max = float(d[valid].max())
            denom = d_max - d_min
            if denom > 1e-6:
                out[valid] = 1.0 - (d[valid] - d_min) / denom
            else:
                out[valid] = 1.0
        gray = np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)