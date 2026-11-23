"""Rendering helper for the PME GUI.

Uses NvdiffrastMeshUlitRenderer when PyTorch + nvdiffrast + CUDA are available,
and falls back to a high-quality vectorized NumPy/OpenCV CPU software renderer
with perspective-correct per-pixel UV interpolation, bilinear texture filtering,
smooth vertex normals, and silhouette antialiasing.
"""

import os
import cv2
import numpy as np


def check_gpu_backend():
    """Return (available: bool, status_text: str)."""
    try:
        import torch
    except ImportError:
        return False, "CPU Preview (PyTorch not installed)"
    try:
        import nvdiffrast.torch  # noqa: F401
    except ImportError:
        return False, f"CPU Preview (PyTorch {torch.__version__}, nvdiffrast missing)"
    if not torch.cuda.is_available():
        return False, f"CPU Preview (PyTorch {torch.__version__}, CUDA unavailable)"
    return True, f"Nvdiffrast GPU ({torch.cuda.get_device_name(0)})"


def load_obj_numpy(obj_path):
    """Lightweight OBJ loader returning NumPy arrays for CPU preview rendering."""
    verts, uvs = [], []
    faces_v, faces_vt = [], []

    with open(obj_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            prefix = parts[0].lower()
            if prefix == 'v':
                verts.append([float(x) for x in parts[1:4]])
            elif prefix == 'vt':
                uvs.append([float(parts[1]), 1.0 - float(parts[2])])
            elif prefix == 'f':
                tokens = parts[1:]
                if len(tokens) < 3:
                    continue
                parsed = []
                for tok in tokens:
                    sub = tok.split('/')
                    vi = int(sub[0])
                    vi = vi - 1 if vi > 0 else len(verts) + vi
                    ti = -1
                    if len(sub) > 1 and sub[1] != '':
                        ti = int(sub[1])
                        ti = ti - 1 if ti > 0 else len(uvs) + ti
                    parsed.append((vi, ti))
                v0, t0 = parsed[0]
                for i in range(len(parsed) - 2):
                    v1, t1 = parsed[i + 1]
                    v2, t2 = parsed[i + 2]
                    faces_v.append([v0, v1, v2])
                    faces_vt.append([t0, t1, t2])

    return (
        np.asarray(verts, dtype=np.float32),
        np.asarray(uvs, dtype=np.float32) if uvs else None,
        np.asarray(faces_v, dtype=np.int32),
        np.asarray(faces_vt, dtype=np.int32) if uvs else None,
    )


def compute_smooth_normals_numpy(verts, faces_v):
    """Compute outward-consistent face normals and area-weighted smooth vertex normals."""
    w_tri = verts[faces_v]  # (F, 3, 3)
    fn_raw = np.cross(w_tri[:, 1] - w_tri[:, 0], w_tri[:, 2] - w_tri[:, 0])
    fn_norm = np.linalg.norm(fn_raw, axis=1, keepdims=True)
    fn = fn_raw / np.maximum(fn_norm, 1e-8)

    # Auto-orient outward using mesh centroid
    center = 0.5 * (verts.min(axis=0) + verts.max(axis=0))
    centroids = w_tri.mean(axis=1)
    if float(np.mean(np.sum((centroids - center[None, :]) * fn, axis=1))) < 0.0:
        fn = -fn
        fn_raw = -fn_raw

    vns = np.zeros_like(verts, dtype=np.float32)
    np.add.at(vns, faces_v[:, 0], fn_raw)
    np.add.at(vns, faces_v[:, 1], fn_raw)
    np.add.at(vns, faces_v[:, 2], fn_raw)
    vn_norm = np.linalg.norm(vns, axis=1, keepdims=True)
    vns = vns / np.maximum(vn_norm, 1e-8)

    return vns.astype(np.float32), fn.astype(np.float32)


def bilinear_sample_texture(tex_img, u, v):
    """Vectorized 4-tap bilinear texture sampling for UV coordinates in [0, 1]."""
    th, tw = tex_img.shape[:2]
    # Wrap / clamp UV cleanly
    fx = np.clip(u, 0.0, 1.0) * (tw - 1)
    fy = np.clip(v, 0.0, 1.0) * (th - 1)

    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    x1 = np.minimum(x0 + 1, tw - 1)
    y1 = np.minimum(y0 + 1, th - 1)

    dx = (fx - x0)[:, None]
    dy = (fy - y0)[:, None]

    c00 = tex_img[y0, x0].astype(np.float32)
    c10 = tex_img[y0, x1].astype(np.float32)
    c01 = tex_img[y1, x0].astype(np.float32)
    c11 = tex_img[y1, x1].astype(np.float32)

    top = c00 * (1.0 - dx) + c10 * dx
    bot = c01 * (1.0 - dx) + c11 * dx
    return top * (1.0 - dy) + bot * dy


def rasterize_mesh_cpu(
    verts,
    uvs,
    faces_v,
    faces_vt,
    tex_bgr,
    intrinsic,
    extrinsic,
    render_size,
    vns=None,
    fns=None,
    bg_color=(0, 0, 0),
    shading='unlit',
    antialias=True,
    wireframe=False,
):
    """High-quality vectorized CPU rasterizer with:
      - Backface & frustum culling
      - Perspective-correct per-pixel barycentric UV interpolation + bilinear texture sampling
      - Smooth per-pixel interpolated vertex normals & exact depth
      - Silhouette edge antialiasing (matching Nvdiffrast visual quality)
    """
    h, w = int(render_size[0]), int(render_size[1])
    if vns is None or fns is None:
        vns, fns = compute_smooth_normals_numpy(verts, faces_v)

    intr = np.asarray(intrinsic, dtype=np.float32).reshape(4, 4)
    extr = np.asarray(extrinsic, dtype=np.float32).reshape(4, 4)

    fx, fy = float(intr[0, 0]), float(intr[1, 1])
    cx, cy = float(intr[0, 2]), float(intr[1, 2])

    R = extr[:3, :3]
    t = extr[:3, 3]
    v_cam = verts @ R.T + t[None, :]

    z = v_cam[:, 2]
    safe_z = np.where(z < 1e-4, 1e-4, z)
    px = fx * (v_cam[:, 0] / safe_z) + cx
    py = fy * (v_cam[:, 1] / safe_z) + cy
    pts_2d = np.stack([px, py], axis=1).astype(np.float32)

    tri_v = v_cam[faces_v]  # (F, 3, 3)
    tri_z = tri_v[:, :, 2].mean(axis=1)

    # Backface & near-plane culling
    fn_cam = fns @ R.T
    centroids_cam = tri_v.mean(axis=1)
    facing = np.sum(fn_cam * centroids_cam, axis=1) < 0.05
    valid = np.all(tri_v[:, :, 2] > 0.05, axis=1) & facing

    # Frustum culling
    p2 = pts_2d[faces_v]  # (F, 3, 2)
    in_x = (p2[:, :, 0].max(axis=1) >= 0) & (p2[:, :, 0].min(axis=1) < w)
    in_y = (p2[:, :, 1].max(axis=1) >= 0) & (p2[:, :, 1].min(axis=1) < h)
    vis_idx = np.where(valid & in_x & in_y)[0]

    order = vis_idx[np.argsort(-tri_z[vis_idx])]

    # Pass 1: Fast OpenCV triangle ID rasterization (exact visibility buffer)
    face_id_buf = np.full((h, w), -1, dtype=np.int32)
    # Use 4-bit subpixel precision (shift=4) for accurate triangle edges
    polys_sub = np.round(p2[order] * 16.0).astype(np.int32)
    for k in range(len(order)):
        cv2.fillConvexPoly(face_id_buf, polys_sub[k], int(order[k]), shift=4)

    color_buf = np.full((h, w, 3), bg_color, dtype=np.uint8)
    mask_buf = np.zeros((h, w), dtype=np.uint8)
    depth_buf = np.zeros((h, w), dtype=np.float32)
    normal_buf = np.zeros((h, w, 3), dtype=np.float32)

    ys, xs = np.where(face_id_buf >= 0)
    if len(ys) > 0:
        mask_buf[ys, xs] = 255
        fids = face_id_buf[ys, xs]
        fv = faces_v[fids]  # (K, 3)

        # Pass 2: Vectorized perspective-correct barycentric coordinates
        tris_2d = pts_2d[fv]  # (K, 3, 2)
        A = tris_2d[:, 0]
        B = tris_2d[:, 1]
        C = tris_2d[:, 2]
        P = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)

        v0 = B - A
        v1 = C - A
        v2 = P - A
        denom = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        safe_d = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        w1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / safe_d
        w2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / safe_d
        w0 = 1.0 - w1 - w2

        w0 = np.clip(w0, 0.0, 1.0)
        w1 = np.clip(w1, 0.0, 1.0)
        w2 = np.clip(w2, 0.0, 1.0)

        # Perspective correction using 1/z
        zs = safe_z[fv]  # (K, 3)
        iz0 = w0 / zs[:, 0]
        iz1 = w1 / zs[:, 1]
        iz2 = w2 / zs[:, 2]
        iz_sum = np.maximum(iz0 + iz1 + iz2, 1e-8)
        pw0 = iz0 / iz_sum
        pw1 = iz1 / iz_sum
        pw2 = iz2 / iz_sum

        # Exact interpolated camera depth
        depth_buf[ys, xs] = 1.0 / iz_sum

        # Smooth per-pixel interpolated world normal
        vn_tri = vns[fv]  # (K, 3, 3)
        pix_n = pw0[:, None] * vn_tri[:, 0] + pw1[:, None] * vn_tri[:, 1] + pw2[:, None] * vn_tri[:, 2]
        pix_n /= np.maximum(np.linalg.norm(pix_n, axis=1, keepdims=True), 1e-8)
        normal_buf[ys, xs] = pix_n

        # Per-pixel color shading
        has_tex = (
            tex_bgr is not None
            and uvs is not None
            and faces_vt is not None
            and np.all(faces_vt[fids] >= 0)
        )
        if shading == 'normal':
            pix_col = np.clip((pix_n * 0.5 + 0.5) * 255.0, 0, 255)[:, ::-1]  # RGB -> BGR
        elif shading == 'clay':
            pix_n_cam = pix_n @ R.T
            light_dir = np.array([0.25, -0.35, -0.90], dtype=np.float32)
            light_dir /= np.linalg.norm(light_dir)
            ndotl = np.clip(-np.sum(pix_n_cam * light_dir[None, :], axis=1), 0.0, 1.0)
            val = (0.30 + 0.70 * ndotl) * 230.0
            pix_col = np.stack([val, val, val], axis=-1)
        else:
            if has_tex:
                uv_tri = uvs[faces_vt[fids]]  # (K, 3, 2)
                u_pix = pw0 * uv_tri[:, 0, 0] + pw1 * uv_tri[:, 1, 0] + pw2 * uv_tri[:, 2, 0]
                v_pix = pw0 * uv_tri[:, 0, 1] + pw1 * uv_tri[:, 1, 1] + pw2 * uv_tri[:, 2, 1]
                pix_col = bilinear_sample_texture(tex_bgr, u_pix, v_pix)
            else:
                pix_col = np.full((len(ys), 3), 210.0, dtype=np.float32)

            if shading == 'lit':
                pix_n_cam = pix_n @ R.T
                light_dir = np.array([0.25, -0.35, -0.90], dtype=np.float32)
                light_dir /= np.linalg.norm(light_dir)
                ndotl = np.clip(-np.sum(pix_n_cam * light_dir[None, :], axis=1), 0.0, 1.0)
                shade = (0.48 + 0.52 * ndotl)[:, None]
                pix_col = pix_col * shade

        color_buf[ys, xs] = np.clip(pix_col, 0, 255).astype(np.uint8)

        # Pass 3: Silhouette edge antialiasing (smoothly anti-alias outer boundary without blurring interior)
        if antialias:
            aa_mask = np.zeros((h, w), dtype=np.uint8)
            for k in range(len(order)):
                cv2.fillConvexPoly(aa_mask, polys_sub[k], 255, lineType=cv2.LINE_AA, shift=4)
            # Soft boundary pixels where antialiased mask differs from binary mask
            kernel = np.ones((3, 3), dtype=np.uint8)
            dilated = cv2.dilate(mask_buf, kernel, iterations=1)
            eroded = cv2.erode(mask_buf, kernel, iterations=1)
            edge_band = (dilated > 0) & (eroded == 0)
            if np.any(edge_band):
                # Dilate foreground color 1px outward so edge blend uses true object color, not black background
                fg_dil = cv2.dilate(color_buf, kernel, iterations=1)
                soft_col = np.where(mask_buf[..., None] > 0, color_buf, fg_dil)
                alpha = cv2.GaussianBlur(aa_mask.astype(np.float32) / 255.0, (3, 3), 0.6)[..., None]
                bg_arr = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
                blended = soft_col.astype(np.float32) * alpha + bg_arr * (1.0 - alpha)
                color_buf[edge_band] = np.clip(blended[edge_band], 0, 255).astype(np.uint8)

    if wireframe:
        polys_int = np.round(p2[order]).astype(np.int32)
        for k in range(len(order)):
            cv2.polylines(color_buf, [polys_int[k]], True, (50, 55, 65), 1, cv2.LINE_AA)

    return {
        'color_bgr': color_buf,
        'mask_gray': mask_buf,
        'depth': depth_buf,
        'normal': normal_buf,
        'face_id': face_id_buf,
        'pts_2d': pts_2d,
        'backend': 'cpu_preview',
    }


def render_cpu_preview(obj_path, tex_path, intrinsic, extrinsic, render_size):
    """High-quality CPU software preview renderer matching Nvdiffrast's camera & UV sampling."""
    verts, uvs, faces_v, faces_vt = load_obj_numpy(obj_path)
    if len(verts) == 0 or len(faces_v) == 0:
        raise ValueError(f"Empty mesh: {obj_path}")

    tex_img = None
    if tex_path and os.path.exists(tex_path):
        tex_img = cv2.imread(tex_path, cv2.IMREAD_COLOR)

    return rasterize_mesh_cpu(
        verts=verts,
        uvs=uvs,
        faces_v=faces_v,
        faces_vt=faces_vt,
        tex_bgr=tex_img,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        render_size=render_size,
        bg_color=(0, 0, 0),
        shading='unlit',
        antialias=True,
    )


_GPU_RENDERER_CACHE = {}


def render_mesh_view(
    obj_path,
    tex_path,
    intrinsic,
    extrinsic,
    render_size,
    device='cuda',
    rastype='opengl',
    prefer_gpu=True,
):
    """Render mesh view via Nvdiffrast GPU when available, else high-quality CPU preview."""
    gpu_ok, _ = check_gpu_backend()
    if prefer_gpu and gpu_ok and device.startswith('cuda'):
        import torch
        from util.obj_io import load_obj
        from render.ulit_renderer import NvdiffrastMeshUlitRenderer

        size_tuple = (int(render_size[0]), int(render_size[1]))
        cache_key = (device, rastype)
        renderer = _GPU_RENDERER_CACHE.get(cache_key)
        if renderer is None:
            renderer = NvdiffrastMeshUlitRenderer({
                'device': device,
                'rastype': rastype,
                'render_size': size_tuple,
            })
            _GPU_RENDERER_CACHE[cache_key] = renderer
        else:
            renderer.set_render_size(size_tuple)

        mesh = load_obj(obj_path, tex_path if tex_path else None, device)
        renderer.set_camera_intrinsic(intrinsic)
        renderer.set_camera_extrinsic(extrinsic)

        with torch.no_grad():
            res = renderer(mesh, render_normal=True)

        if res['color'] is not None:
            color_bgr = np.clip(res['color'][0].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        else:
            n_np = res['normal'][0].detach().cpu().numpy()
            color_bgr = np.clip((n_np * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

        mask_gray = np.clip(res['mask'][0, ..., 0].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        depth = res['depth'][0, ..., 0].detach().cpu().numpy()
        normal = res['normal'][0].detach().cpu().numpy()

        return {
            'color_bgr': color_bgr,
            'mask_gray': mask_gray,
            'depth': depth,
            'normal': normal,
            'backend': 'nvdiffrast_gpu',
            'mesh': mesh,
            'renderer': renderer,
        }

    return render_cpu_preview(obj_path, tex_path, intrinsic, extrinsic, render_size)


def load_normal_map_file(path, target_size=None):
    """Load normal map from .npy (float HxWx3) or image file (uint8 RGB/BGR -> [-1, 1])."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        arr = np.load(path).astype(np.float32)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read normal map image: {path}")
        arr = (img.astype(np.float32) / 255.0) * 2.0 - 1.0

    if target_size is not None and arr.shape[:2] != (int(target_size[0]), int(target_size[1])):
        arr = cv2.resize(arr, (int(target_size[1]), int(target_size[0])), interpolation=cv2.INTER_LINEAR)
    return arr


def load_depth_map_file(path, target_size=None):
    """Load depth map from .npy (float HxW) or 16-bit/8-bit PNG."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read depth map image: {path}")
        if img.ndim == 3:
            img = img[..., 0]
        if img.dtype == np.uint16:
            arr = img.astype(np.float32) / 10000.0
        else:
            arr = img.astype(np.float32) / 255.0

    if target_size is not None and arr.shape[:2] != (int(target_size[0]), int(target_size[1])):
        arr = cv2.resize(arr, (int(target_size[1]), int(target_size[0])), interpolation=cv2.INTER_LINEAR)
    return arr
