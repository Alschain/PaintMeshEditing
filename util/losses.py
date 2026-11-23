import torch
import torch.nn.functional as F

try:
    import torchvision
except ImportError:
    torchvision = None


def _ensure_nchw(img):
    """Convert (H, W, C) or (B, H, W, C) tensor to (B, C, H, W)."""
    if img.ndim == 3:
        return img.permute(2, 0, 1).unsqueeze(0)
    if img.ndim == 4 and img.shape[-1] in (1, 3, 4):
        return img.permute(0, 3, 1, 2)
    return img


def _gaussian_kernel_2d(kernel_size, sigma, device, dtype):
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    kernel = g.unsqueeze(1) * g.unsqueeze(0)
    return kernel


def gaussian_blur_hwc(img, kernel_size=9, sigma=None):
    """Differentiable 2D Gaussian blur for (H, W, C) tensors."""
    k = int(kernel_size)
    if k <= 1:
        return img
    if k % 2 == 0:
        k += 1
    if sigma is None:
        sigma = max(0.5, 0.3 * ((k - 1) * 0.5 - 1) + 0.8)

    x = _ensure_nchw(img)
    c = x.shape[1]
    kernel = _gaussian_kernel_2d(k, sigma, x.device, x.dtype).view(1, 1, k, k).repeat(c, 1, 1, 1)
    pad = k // 2
    x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    blurred = F.conv2d(x, kernel, groups=c)
    return blurred.squeeze(0).permute(1, 2, 0)


# ---------------------------------------------------------------------------
# 2D Image & Multi-Modal Guidance Losses
# ---------------------------------------------------------------------------

def image_l1_loss(pred, target, mask=None):
    if mask is not None:
        return F.l1_loss(pred * mask, target * mask)
    return F.l1_loss(pred, target)


def image_l2_loss(pred, target, mask=None):
    if mask is not None:
        return F.mse_loss(pred * mask, target * mask)
    return F.mse_loss(pred, target)


def ssim_loss(pred, target, mask=None, window_size=11, sigma=1.5):
    """1 - SSIM loss over (H, W, C) images using pure PyTorch convolutions."""
    if mask is not None:
        pred = pred * mask
        target = target * mask

    x = _ensure_nchw(pred)
    y = _ensure_nchw(target)
    channels = x.shape[1]

    k = int(window_size)
    if k % 2 == 0:
        k += 1
    window = _gaussian_kernel_2d(k, sigma, x.device, x.dtype).view(1, 1, k, k).repeat(channels, 1, 1, 1)
    pad = k // 2

    mu_x = F.conv2d(x, window, padding=pad, groups=channels)
    mu_y = F.conv2d(y, window, padding=pad, groups=channels)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=pad, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=pad, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=channels) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2) + 1e-8
    )

    if mask is not None:
        m = _ensure_nchw(mask)
        weight = m.expand_as(ssim_map)
        denom = weight.sum().clamp(min=1.0)
        return ((1.0 - ssim_map) * weight).sum() / denom

    return (1.0 - ssim_map).mean()


def normal_map_loss(pred_normal, target_normal, mask=None, cosine_weight=0.5):
    """Combined L1 + cosine similarity loss on surface normal maps."""
    if mask is not None:
        pred_m = pred_normal * mask
        target_m = target_normal * mask
        l1 = F.l1_loss(pred_m, target_m)

        valid = (mask[..., 0] > 0.5) & (target_normal.norm(dim=-1) > 1e-3) & (pred_normal.norm(dim=-1) > 1e-3)
        if valid.any():
            p_n = F.normalize(pred_normal[valid], dim=-1, eps=1e-6)
            t_n = F.normalize(target_normal[valid], dim=-1, eps=1e-6)
            cos_loss = (1.0 - (p_n * t_n).sum(dim=-1)).mean()
        else:
            cos_loss = pred_normal.sum() * 0.0
        return l1 + cosine_weight * cos_loss

    l1 = F.l1_loss(pred_normal, target_normal)
    p_n = F.normalize(pred_normal, dim=-1, eps=1e-6)
    t_n = F.normalize(target_normal, dim=-1, eps=1e-6)
    cos_loss = (1.0 - (p_n * t_n).sum(dim=-1)).mean()
    return l1 + cosine_weight * cos_loss


def depth_map_loss(pred_depth, target_depth, mask=None, grad_weight=0.25):
    """Depth map constraint: masked L1 + spatial gradient consistency."""
    if pred_depth.ndim == 2:
        pred_depth = pred_depth.unsqueeze(-1)
    if target_depth.ndim == 2:
        target_depth = target_depth.unsqueeze(-1)

    if mask is not None:
        valid = (mask > 0.5) & (target_depth > 0) & (pred_depth > 0)
    else:
        valid = (target_depth > 0) & (pred_depth > 0)

    if valid.any():
        l1 = F.l1_loss(pred_depth[valid], target_depth[valid])
    else:
        l1 = F.l1_loss(pred_depth, target_depth)

    if grad_weight > 0.0:
        pd = pred_depth * (mask if mask is not None else 1.0)
        td = target_depth * (mask if mask is not None else 1.0)
        dx_p = pd[:, 1:, :] - pd[:, :-1, :]
        dx_t = td[:, 1:, :] - td[:, :-1, :]
        dy_p = pd[1:, :, :] - pd[:-1, :, :]
        dy_t = td[1:, :, :] - td[:-1, :, :]
        grad_loss = F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)
        return l1 + grad_weight * grad_loss

    return l1


def silhouette_loss(pred_silhouette, target_silhouette, iou_weight=0.5):
    """Differentiable silhouette constraint combining L1 and soft IoU loss."""
    if pred_silhouette.ndim == 2:
        pred_silhouette = pred_silhouette.unsqueeze(-1)
    if target_silhouette.ndim == 2:
        target_silhouette = target_silhouette.unsqueeze(-1)

    pred = pred_silhouette.clamp(0.0, 1.0)
    target = target_silhouette.clamp(0.0, 1.0)

    l1 = F.l1_loss(pred, target)
    intersection = (pred * target).sum()
    union = (pred + target - pred * target).sum().clamp(min=1e-6)
    soft_iou = 1.0 - intersection / union
    return l1 + iou_weight * soft_iou


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        if torchvision is None:
            raise ImportError("torchvision is required for VGGPerceptualLoss.")

        try:
            weights = torchvision.models.VGG16_Weights.DEFAULT
            vgg = torchvision.models.vgg16(weights=weights).features
        except Exception:
            vgg = torchvision.models.vgg16(pretrained=True).features

        blocks = [
            vgg[:4].eval(),
            vgg[4:9].eval(),
            vgg[9:16].eval(),
            vgg[16:23].eval(),
        ]
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = F.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=(0, 1, 2, 3), style_layers=()):
        input = _ensure_nchw(input)
        target = _ensure_nchw(target)
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss = loss + F.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss = loss + F.l1_loss(gram_x, gram_y)
        return loss


# ---------------------------------------------------------------------------
# 3D Mesh Geometry Regularization Losses
# ---------------------------------------------------------------------------

def normal_consistency_loss(face_normals, connected_faces):
    """Compute the normal consistency term as the cosine similarity between neighboring face normals.
    Modified from PyTorch3D.
    """
    if connected_faces is None or connected_faces.numel() == 0:
        return face_normals.sum() * 0.0
    loss = 1.0 - torch.cosine_similarity(
        face_normals[connected_faces[:, 0]],
        face_normals[connected_faces[:, 1]],
        dim=1,
    )
    return (loss ** 2).mean()


def relative_normal_consistency_loss(
    vertex_normals, face_normals, edges, connected_faces, ori_link_nc, ori_link_face_nc
):
    """Preserve original relative vertex and face normal differences across edges."""
    v_term = (((vertex_normals[edges[:, 0]] - vertex_normals[edges[:, 1]]) - ori_link_nc) ** 2).mean()
    if connected_faces is not None and connected_faces.numel() > 0:
        f_term = (
            ((face_normals[connected_faces[:, 0]] - face_normals[connected_faces[:, 1]]) - ori_link_face_nc) ** 2
        ).mean()
    else:
        f_term = v_term * 0.0
    return v_term + f_term


def laplacian_loss(laplacian, vertices):
    """Compute the Laplacian term as the mean squared Euclidean norm of differential coordinates.
    Modified from PyTorch3D.
    """
    loss = torch.sparse.mm(laplacian, vertices)
    loss = loss.norm(dim=1) ** 2
    return loss.mean()


def differential_laplacian_loss(laplacian, vertices, ori_delta):
    """Detail-preserving Laplacian loss: penalizes deviation from original differential coordinates."""
    cur_delta = torch.sparse.mm(laplacian, vertices)
    return ((cur_delta - ori_delta) ** 2).sum(dim=1).mean()


def edge_length_loss(edges_len, ori_edge_len):
    return ((edges_len - ori_edge_len) ** 2).mean()


def area_preservation_loss(face_area, ori_face_area):
    return ((face_area - ori_face_area) ** 2).mean()


def vertex_displacement_loss(vertices, ori_vertices, weights=None):
    """L2 anchor regularization preventing unnecessary drift from the original mesh."""
    diff_sq = ((vertices - ori_vertices) ** 2).sum(dim=-1)
    if weights is not None:
        return (diff_sq * weights).mean()
    return diff_sq.mean()
