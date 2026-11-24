"""Interactive 3D Viewport with Real-Time Arcball Rotation and Direct 3D Mesh Painting.

Works out-of-the-box with NumPy + OpenCV + Tkinter on any machine, and automatically
accelerates rendering with Nvdiffrast GPU when PyTorch + CUDA + nvdiffrast are available.
"""

import math
import os
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from gui.preview_renderer import (
    check_gpu_backend,
    compute_smooth_normals_numpy,
    load_obj_numpy,
    rasterize_mesh_cpu,
)


def _rodrigues_matrix(axis, angle):
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8 or abs(angle) < 1e-8:
        return np.eye(3, dtype=np.float32)
    u = axis / norm
    ux, uy, uz = float(u[0]), float(u[1]), float(u[2])
    c = math.cos(angle)
    s = math.sin(angle)
    omc = 1.0 - c
    return np.array([
        [c + ux * ux * omc,      ux * uy * omc - uz * s, ux * uz * omc + uy * s],
        [uy * ux * omc + uz * s, c + uy * uy * omc,      uy * uz * omc - ux * s],
        [uz * ux * omc - uy * s, uz * uy * omc + ux * s, c + uz * uz * omc],
    ], dtype=np.float32)


class ArcballCamera:
    """Gimbal-lock-free virtual trackball camera matching PME's pinhole extrinsic convention.

    Extrinsic E = [R | t; 0 | 1] transforms world points to camera space (+X right, +Y down, +Z forward).
    """

    def __init__(self):
        self.R = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
        self.target = np.zeros(3, dtype=np.float32)
        self.distance = 2.5
        self.pan_cam = np.zeros(2, dtype=np.float32)

    def set_from_extrinsic(self, extrinsic, mesh_center=None):
        E = np.asarray(extrinsic, dtype=np.float32).reshape(4, 4)
        R = E[:3, :3].copy()
        t = E[:3, 3].copy()
        # Orthonormalize R via SVD in case of slight numerical drift
        u, _, vt = np.linalg.svd(R)
        R = (u @ vt).astype(np.float32)
        if np.linalg.det(R) < 0:
            u[:, -1] *= -1
            R = (u @ vt).astype(np.float32)

        self.R = R
        if mesh_center is not None:
            self.target = np.asarray(mesh_center, dtype=np.float32).reshape(3).copy()
        else:
            self.target = np.zeros(3, dtype=np.float32)

        t_cam = self.R @ self.target + t
        self.pan_cam = t_cam[:2].copy()
        self.distance = float(max(0.15, t_cam[2]))

    def get_extrinsic(self):
        E = np.eye(4, dtype=np.float32)
        E[:3, :3] = self.R
        cam_target = np.array([self.pan_cam[0], self.pan_cam[1], self.distance], dtype=np.float32)
        E[:3, 3] = cam_target - self.R @ self.target
        return E

    @staticmethod
    def _project_to_arcball(x, y, width, height):
        r = 0.5 * min(max(1, width), max(1, height))
        nx = (x - 0.5 * width) / r
        ny = (y - 0.5 * height) / r
        d2 = nx * nx + ny * ny
        if d2 <= 0.5:
            nz = -math.sqrt(max(0.0, 1.0 - d2))
        else:
            nz = -0.5 / math.sqrt(d2)
        v = np.array([nx, ny, nz], dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / max(n, 1e-8)

    def apply_arcball_drag(self, R_start, x0, y0, x1, y1, width, height):
        p0 = self._project_to_arcball(x0, y0, width, height)
        p1 = self._project_to_arcball(x1, y1, width, height)
        axis = np.cross(p0, p1)
        dot = float(np.clip(np.dot(p0, p1), -1.0, 1.0))
        angle = math.acos(dot) * 1.5
        dR = _rodrigues_matrix(axis, angle)
        self.R = (dR @ R_start).astype(np.float32)

    def pan(self, dx, dy, width, height):
        scale = self.distance / max(200.0, float(min(width, height)))
        self.pan_cam[0] += float(dx) * scale
        self.pan_cam[1] += float(dy) * scale

    def dolly(self, factor):
        self.distance = float(np.clip(self.distance * factor, 0.15, 50.0))


class Viewport3D(ttk.Frame):
    """Real-time 3D interactive viewport supporting Arcball rotation & direct 3D texture painting."""

    SHADING_MODES = ('Textured (Unlit)', 'Textured + Lit', 'Normal Shading', 'Clay Lit')
    MODES = ('orbit', 'paint3d', 'erase3d', 'pick3d')

    def __init__(
        self,
        parent,
        on_color_picked=None,
        on_texture_modified=None,
        on_camera_changed=None,
        on_status_update=None,
    ):
        super().__init__(parent)
        self.on_color_picked = on_color_picked
        self.on_texture_modified = on_texture_modified
        self.on_camera_changed = on_camera_changed
        self.on_status_update = on_status_update

        self.gpu_ok, _ = check_gpu_backend()
        self.camera = ArcballCamera()

        # Mesh geometry & UV texture state
        self.obj_path = None
        self.tex_path = None
        self.verts = None         # (V, 3) float32
        self.uvs = None           # (VT, 2) float32
        self.faces_v = None       # (F, 3) int32
        self.faces_vt = None      # (F, 3) int32
        self.vertex_normals = None  # (V, 3) float32 (smooth vertex normals)
        self.face_normals = None  # (F, 3) float32 (outward-oriented)
        self.mesh_center = np.zeros(3, dtype=np.float32)
        self.mesh_radius = 1.0
        self.winding_sign = 1.0

        self.orig_tex_bgr = None
        self.tex_bgr = None
        self.face_colors_bgr = None  # (F, 3) uint8
        self.intrinsic = None
        self.render_size = (1024, 1024)

        # Cached render buffers for hit-testing & 3D painting
        self._last_color_bgr = None
        self._last_face_id = None     # (H_vp, W_vp) int32
        self._last_pts_2d = None      # (V, 2) float32 in viewport coords
        self._vp_size = (512, 512)    # (H, W) of current rendered viewport image
        self._display_rect = (0, 0, 512, 512)  # (ox, oy, dw, dh) on Tk canvas

        # Interaction state
        self.mode = 'orbit'
        self.shading = self.SHADING_MODES[1]
        self.show_wireframe = False
        self.brush_radius = 14
        self.brush_color_bgr = (0, 0, 255)  # Default red label

        self._dragging_orbit = False
        self._dragging_pan = False
        self._painting = False
        self._drag_start_xy = None
        self._drag_start_R = None
        self._last_paint_xy = None
        self._wheel_after_id = None
        self._texture_dirty = False

        # 3D Texture Undo / Redo stack
        self._undo_stack = []
        self._redo_stack = []
        self._max_history = 15

        self._tk_img = None
        self._build_ui()

    def _build_ui(self):
        bar = ttk.Frame(self, style='Panel.TFrame', padding=(8, 4))
        bar.pack(side=tk.TOP, fill=tk.X)

        self._mode_btns = {}
        for key, text in [
            ('orbit', 'Orbit (O)'),
            ('paint3d', '3D Paint (P)'),
            ('erase3d', '3D Erase (R)'),
            ('pick3d', '3D Pick (K)'),
        ]:
            btn = ttk.Button(bar, text=text, style='Tool.TButton', command=lambda k=key: self.set_mode(k))
            btn.pack(side=tk.LEFT, padx=2)
            self._mode_btns[key] = btn

        ttk.Separator(bar, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

        self.var_shading = tk.StringVar(value=self.shading)
        shade_cb = ttk.Combobox(
            bar,
            textvariable=self.var_shading,
            values=self.SHADING_MODES,
            state='readonly',
            width=16,
        )
        shade_cb.pack(side=tk.LEFT, padx=2)
        shade_cb.bind('<<ComboboxSelected>>', lambda _e: self._on_shading_changed())

        self.var_wire = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Wireframe", variable=self.var_wire, command=self._on_wire_changed).pack(
            side=tk.LEFT, padx=(6, 4)
        )

        ttk.Button(bar, text="Undo 3D", command=self.undo).pack(side=tk.RIGHT, padx=2)
        ttk.Button(bar, text="Reset Tex", command=self.reset_texture).pack(side=tk.RIGHT, padx=2)

        self.canvas = tk.Canvas(self, bg='#121318', highlightthickness=0, cursor='fleur')
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.bind('<ButtonPress-1>', self._on_left_down)
        self.canvas.bind('<B1-Motion>', self._on_left_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_left_up)
        self.canvas.bind('<ButtonPress-3>', self._on_right_down)
        self.canvas.bind('<B3-Motion>', self._on_right_drag)
        self.canvas.bind('<ButtonRelease-3>', self._on_right_up)
        self.canvas.bind('<ButtonPress-2>', self._on_mid_down)
        self.canvas.bind('<B2-Motion>', self._on_mid_drag)
        self.canvas.bind('<ButtonRelease-2>', self._on_mid_up)
        self.canvas.bind('<Motion>', self._on_mouse_move)
        self.canvas.bind('<Leave>', lambda _e: self.canvas.delete('cursor'))
        self.canvas.bind('<MouseWheel>', self._on_wheel)
        self.canvas.bind('<Button-4>', lambda _e: self._zoom_step(0.9))
        self.canvas.bind('<Button-5>', lambda _e: self._zoom_step(1.1))
        self.canvas.bind('<Configure>', self._on_resize)

        self.set_mode('orbit')

    # ------------------------------------------------------------------
    # Mesh / Texture / Camera Loading
    # ------------------------------------------------------------------

    def load_mesh(self, obj_path, tex_path=None, intrinsic=None, extrinsic=None, render_size=(1024, 1024)):
        verts, uvs, faces_v, faces_vt = load_obj_numpy(obj_path)
        if len(verts) == 0 or len(faces_v) == 0:
            raise ValueError(f"Empty OBJ mesh: {obj_path}")

        self.obj_path = obj_path
        self.tex_path = tex_path
        self.verts = verts
        self.uvs = uvs
        self.faces_v = faces_v
        self.faces_vt = faces_vt
        self.render_size = (int(render_size[0]), int(render_size[1]))

        bbox_min = verts.min(axis=0)
        bbox_max = verts.max(axis=0)
        self.mesh_center = 0.5 * (bbox_min + bbox_max)
        self.mesh_radius = float(max(1e-3, 0.5 * np.linalg.norm(bbox_max - bbox_min)))

        # Compute face normals and auto-detect consistent outward orientation
        self._recompute_face_normals()

        # Load or synthesize editable UV texture
        tex_img = None
        if tex_path and os.path.exists(tex_path):
            tex_img = cv2.imread(tex_path, cv2.IMREAD_COLOR)
        if tex_img is None and uvs is not None:
            tex_img = np.full((1024, 1024, 3), 210, dtype=np.uint8)

        self.orig_tex_bgr = tex_img.copy() if tex_img is not None else None
        self.tex_bgr = tex_img.copy() if tex_img is not None else None
        self._texture_dirty = False
        self._undo_stack.clear()
        self._redo_stack.clear()

        self._refresh_face_colors()

        if intrinsic is not None:
            self.intrinsic = np.asarray(intrinsic, dtype=np.float32).reshape(4, 4).copy()
        else:
            f = 1.2 * max(self.render_size)
            self.intrinsic = np.array([
                [f, 0, self.render_size[1] * 0.5, 0],
                [0, f, self.render_size[0] * 0.5, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)

        if extrinsic is not None:
            self.camera.set_from_extrinsic(extrinsic, mesh_center=self.mesh_center)
        else:
            self.camera.target = self.mesh_center.copy()
            self.camera.distance = self.mesh_radius * 3.0
            self.camera.pan_cam[:] = 0.0

        self.render(interactive=False)

    def update_vertices(self, new_verts):
        """Update 3D viewport geometry live after optimization without reloading OBJ."""
        if self.verts is None:
            return
        arr = np.asarray(new_verts, dtype=np.float32)
        if arr.shape == self.verts.shape:
            self.verts = arr.copy()
            self._recompute_face_normals()
            self.render(interactive=False)

    def _recompute_face_normals(self):
        self.vertex_normals, self.face_normals = compute_smooth_normals_numpy(self.verts, self.faces_v)

    def _refresh_face_colors(self, face_indices=None):
        if self.faces_v is None:
            return
        if (
            self.tex_bgr is not None
            and self.uvs is not None
            and self.faces_vt is not None
            and np.all(self.faces_vt >= 0)
        ):
            th, tw = self.tex_bgr.shape[:2]
            if face_indices is None:
                uv_c = self.uvs[self.faces_vt].mean(axis=1)
                u_idx = np.clip((uv_c[:, 0] * (tw - 1)).astype(np.int32), 0, tw - 1)
                v_idx = np.clip((uv_c[:, 1] * (th - 1)).astype(np.int32), 0, th - 1)
                self.face_colors_bgr = self.tex_bgr[v_idx, u_idx].copy()
            else:
                uv_c = self.uvs[self.faces_vt[face_indices]].mean(axis=1)
                u_idx = np.clip((uv_c[:, 0] * (tw - 1)).astype(np.int32), 0, tw - 1)
                v_idx = np.clip((uv_c[:, 1] * (th - 1)).astype(np.int32), 0, th - 1)
                self.face_colors_bgr[face_indices] = self.tex_bgr[v_idx, u_idx]
        else:
            if face_indices is None or self.face_colors_bgr is None:
                self.face_colors_bgr = np.full((len(self.faces_v), 3), 205, dtype=np.uint8)

    def get_extrinsic(self):
        return self.camera.get_extrinsic()

    def set_extrinsic(self, extrinsic):
        self.camera.set_from_extrinsic(extrinsic, mesh_center=self.mesh_center)
        self.render(interactive=False)

    def save_painted_texture(self, path):
        if self.tex_bgr is None:
            raise ValueError("No UV texture available to save.")
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        cv2.imwrite(path, self.tex_bgr)
        self.tex_path = path
        self._texture_dirty = False
        return path

    def has_3d_paint_edits(self):
        if self.tex_bgr is None or self.orig_tex_bgr is None:
            return False
        if self.tex_bgr.shape != self.orig_tex_bgr.shape:
            return False
        return bool(np.any(self.tex_bgr != self.orig_tex_bgr))

    def project_3d_paint_mask(self, intrinsic, extrinsic, render_size):
        """Project UV texels painted directly in 3D onto the given 2D camera viewpoint."""
        if not self.has_3d_paint_edits() or self.verts is None or self.faces_v is None:
            return None
        uv_diff = np.any(self.tex_bgr != self.orig_tex_bgr, axis=-1)
        mask_tex = np.where(uv_diff[..., None], 255, 0).astype(np.uint8)
        mask_tex = np.repeat(mask_tex, 3, axis=-1)
        res = rasterize_mesh_cpu(
            verts=self.verts,
            uvs=self.uvs,
            faces_v=self.faces_v,
            faces_vt=self.faces_vt,
            tex_bgr=mask_tex,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            render_size=render_size,
            vns=self.vertex_normals,
            fns=self.face_normals,
            bg_color=(0, 0, 0),
            shading='unlit',
            antialias=False,
        )
        return np.where(res['color_bgr'][..., 0] > 127, 255, 0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Modes & Undo/Redo
    # ------------------------------------------------------------------

    def set_mode(self, mode):
        if mode not in self.MODES:
            return
        self.mode = mode
        for k, btn in self._mode_btns.items():
            btn.configure(style='ToolActive.TButton' if k == mode else 'Tool.TButton')
        cursor_map = {
            'orbit': 'fleur',
            'paint3d': 'crosshair',
            'erase3d': 'circle',
            'pick3d': 'tcross',
        }
        self.canvas.configure(cursor=cursor_map.get(mode, 'fleur'))

    def _on_shading_changed(self):
        self.shading = self.var_shading.get()
        self.render(interactive=False)

    def _on_wire_changed(self):
        self.show_wireframe = bool(self.var_wire.get())
        self.render(interactive=False)

    def _push_undo(self):
        if self.tex_bgr is None or self.face_colors_bgr is None:
            return
        self._undo_stack.append((self.tex_bgr.copy(), self.face_colors_bgr.copy()))
        if len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append((self.tex_bgr.copy(), self.face_colors_bgr.copy()))
        self.tex_bgr, self.face_colors_bgr = self._undo_stack.pop()
        self._texture_dirty = True
        self.render(interactive=False)
        if self.on_texture_modified:
            self.on_texture_modified(self.tex_bgr)

    def reset_texture(self):
        if self.orig_tex_bgr is None:
            return
        self._push_undo()
        self.tex_bgr = self.orig_tex_bgr.copy()
        self._refresh_face_colors()
        self._texture_dirty = True
        self.render(interactive=False)
        if self.on_texture_modified:
            self.on_texture_modified(self.tex_bgr)

    # ------------------------------------------------------------------
    # Real-Time 3D Rendering
    # ------------------------------------------------------------------

    def _compute_viewport_resolution(self, interactive=False):
        cw = max(self.canvas.winfo_width(), 320)
        ch = max(self.canvas.winfo_height(), 320)
        ref_h, ref_w = self.render_size
        aspect = ref_w / max(1.0, float(ref_h))

        # Fit aspect inside canvas while capping resolution for fast software rasterization
        max_dim = 420 if interactive else 600
        if cw / max(1.0, float(ch)) > aspect:
            dh = min(ch, max_dim)
            dw = max(64, int(round(dh * aspect)))
        else:
            dw = min(cw, max_dim)
            dh = max(64, int(round(dw / aspect)))
        return int(dh), int(dw)

    def render(self, interactive=False):
        if self.verts is None or self.faces_v is None:
            self._draw_placeholder()
            return

        vh, vw = self._compute_viewport_resolution(interactive=interactive)
        self._vp_size = (vh, vw)

        extr = self.camera.get_extrinsic()
        intr = self.intrinsic.copy()
        # Scale intrinsic to viewport resolution
        sx = vw / max(1.0, float(self.render_size[1]))
        sy = vh / max(1.0, float(self.render_size[0]))
        intr_scaled = intr.copy()
        intr_scaled[0, 0] *= sx
        intr_scaled[1, 1] *= sy
        intr_scaled[0, 2] *= sx
        intr_scaled[1, 2] *= sy

        if not interactive:
            # High-quality per-pixel perspective-correct UV + bilinear texture + smooth normals + AA
            shading_map = {
                'Textured (Unlit)': 'unlit',
                'Textured + Lit': 'lit',
                'Normal Shading': 'normal',
                'Clay Lit': 'clay',
            }
            res = rasterize_mesh_cpu(
                verts=self.verts,
                uvs=self.uvs,
                faces_v=self.faces_v,
                faces_vt=self.faces_vt,
                tex_bgr=self.tex_bgr,
                intrinsic=intr_scaled,
                extrinsic=extr,
                render_size=(vh, vw),
                vns=self.vertex_normals,
                fns=self.face_normals,
                bg_color=(24, 25, 30),
                shading=shading_map.get(self.shading, 'lit'),
                antialias=True,
                wireframe=self.show_wireframe,
            )
            self._last_color_bgr = res['color_bgr']
            self._last_face_id = res['face_id']
            self._last_pts_2d = res['pts_2d']
            self._blit_to_canvas()
            return

        # Fast interactive path while actively dragging Arcball / zooming
        fx, fy = float(intr_scaled[0, 0]), float(intr_scaled[1, 1])
        cx, cy = float(intr_scaled[0, 2]), float(intr_scaled[1, 2])

        R = extr[:3, :3]
        t = extr[:3, 3]
        v_cam = self.verts @ R.T + t[None, :]

        z = v_cam[:, 2]
        safe_z = np.where(z < 1e-4, 1e-4, z)
        px = fx * (v_cam[:, 0] / safe_z) + cx
        py = fy * (v_cam[:, 1] / safe_z) + cy
        pts_2d = np.stack([px, py], axis=1).astype(np.float32)
        self._last_pts_2d = pts_2d

        tri_v = v_cam[self.faces_v]  # (F, 3, 3)
        tri_z = tri_v[:, :, 2].mean(axis=1)

        # Near-plane & backface culling (vectorized)
        fn_cam = self.face_normals @ R.T
        centroids_cam = tri_v.mean(axis=1)
        facing = np.sum(fn_cam * centroids_cam, axis=1) < 0.05
        valid = np.all(tri_v[:, :, 2] > 0.05, axis=1) & facing

        # Frustum culling (vectorized)
        p2 = pts_2d[self.faces_v]  # (F, 3, 2)
        in_x = (p2[:, :, 0].max(axis=1) >= 0) & (p2[:, :, 0].min(axis=1) < vw)
        in_y = (p2[:, :, 1].max(axis=1) >= 0) & (p2[:, :, 1].min(axis=1) < vh)
        vis_idx = np.where(valid & in_x & in_y)[0]

        if len(vis_idx) > 10000:
            step = max(1, len(vis_idx) // 10000)
            vis_idx = vis_idx[::step]

        order = vis_idx[np.argsort(-tri_z[vis_idx])]

        if self.shading == 'Normal Shading':
            n_vis = np.clip((self.face_normals[order] * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
            draw_colors = n_vis[:, ::-1]
        elif self.shading == 'Clay Lit':
            light_dir = np.array([0.25, -0.35, -0.90], dtype=np.float32)
            light_dir /= np.linalg.norm(light_dir)
            ndotl = np.clip(-np.sum(fn_cam[order] * light_dir[None, :], axis=1), 0.0, 1.0)
            c_u8 = np.clip((0.30 + 0.70 * ndotl) * 230.0, 0, 255).astype(np.uint8)
            draw_colors = np.stack([c_u8, c_u8, c_u8], axis=-1)
        elif self.shading == 'Textured + Lit':
            light_dir = np.array([0.25, -0.35, -0.90], dtype=np.float32)
            light_dir /= np.linalg.norm(light_dir)
            ndotl = np.clip(-np.sum(fn_cam[order] * light_dir[None, :], axis=1), 0.0, 1.0)
            shade = (0.48 + 0.52 * ndotl)[:, None]
            draw_colors = np.clip(self.face_colors_bgr[order].astype(np.float32) * shade, 0, 255).astype(np.uint8)
        else:
            draw_colors = self.face_colors_bgr[order]

        color_buf = np.full((vh, vw, 3), (24, 25, 30), dtype=np.uint8)
        polys = np.round(p2[order]).astype(np.int32)
        for k in range(len(order)):
            col = (int(draw_colors[k, 0]), int(draw_colors[k, 1]), int(draw_colors[k, 2]))
            cv2.fillConvexPoly(color_buf, polys[k], col)

        self._last_color_bgr = color_buf
        self._blit_to_canvas()

    def _draw_placeholder(self):
        self.canvas.delete('all')
        cw = max(self.canvas.winfo_width(), 300)
        ch = max(self.canvas.winfo_height(), 300)
        self.canvas.create_text(
            cw // 2,
            ch // 2,
            text="Load a mesh to inspect in the 3D Interactive Viewport\n(Left-drag: Arcball Orbit | Right-drag: Orbit/Pan | Wheel: Zoom | 3D Paint: Direct UV Painting)",
            fill='#6b7280',
            font=('Segoe UI', 10),
            justify='center',
        )

    def _blit_to_canvas(self):
        if self._last_color_bgr is None:
            return
        cw = max(self.canvas.winfo_width(), 200)
        ch = max(self.canvas.winfo_height(), 200)
        ih, iw = self._last_color_bgr.shape[:2]

        scale = min((cw - 12) / max(1.0, float(iw)), (ch - 12) / max(1.0, float(ih)))
        dw = max(1, int(round(iw * scale)))
        dh = max(1, int(round(ih * scale)))
        ox = (cw - dw) // 2
        oy = (ch - dh) // 2
        self._display_rect = (ox, oy, dw, dh)

        rgb = cv2.cvtColor(self._last_color_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((dw, dh), Image.BILINEAR)
        self._tk_img = ImageTk.PhotoImage(pil_img)

        self.canvas.delete('vp3d')
        self.canvas.create_image(ox, oy, anchor='nw', image=self._tk_img, tags='vp3d')
        self.canvas.tag_lower('vp3d')

    # ------------------------------------------------------------------
    # Direct 3D Mesh Surface Painting (UV Texture + Live Feedback)
    # ------------------------------------------------------------------

    def _canvas_to_vp(self, cx, cy):
        ox, oy, dw, dh = self._display_rect
        ih, iw = self._vp_size
        vx = (cx - ox) * (iw / max(1.0, float(dw)))
        vy = (cy - oy) * (ih / max(1.0, float(dh)))
        return vx, vy

    def _paint_3d_at(self, vx, vy, erase=False):
        if (
            self._last_face_id is None
            or self._last_pts_2d is None
            or self.faces_v is None
        ):
            return False

        vh, vw = self._vp_size
        ix, iy = int(round(vx)), int(round(vy))
        rad = max(1, int(round(self.brush_radius * (vw / max(1.0, float(self._display_rect[2]))))))

        x0 = max(0, ix - rad)
        x1 = min(vw - 1, ix + rad)
        y0 = max(0, iy - rad)
        y1 = min(vh - 1, iy + rad)
        if x0 > x1 or y0 > y1:
            return False

        # Circular brush mask in viewport buffer coordinates
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        in_circle = ((xx - vx) ** 2 + (yy - vy) ** 2) <= (rad ** 2)
        sub_fids = self._last_face_id[y0:y1 + 1, x0:x1 + 1]
        hit_mask = in_circle & (sub_fids >= 0)
        if not np.any(hit_mask):
            return False

        hit_xs = xx[hit_mask].astype(np.float32)
        hit_ys = yy[hit_mask].astype(np.float32)
        hit_fids = sub_fids[hit_mask]
        unique_fids = np.unique(hit_fids)

        # Paint into editable UV texture via exact barycentric interpolation
        if (
            self.tex_bgr is not None
            and self.uvs is not None
            and self.faces_vt is not None
            and np.all(self.faces_vt[unique_fids] >= 0)
        ):
            th, tw = self.tex_bgr.shape[:2]
            tris_2d = self._last_pts_2d[self.faces_v[hit_fids]]  # (K, 3, 2)
            A = tris_2d[:, 0]
            B = tris_2d[:, 1]
            C = tris_2d[:, 2]
            P = np.stack([hit_xs, hit_ys], axis=1)

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
            w_sum = np.maximum(w0 + w1 + w2, 1e-6)
            w0 /= w_sum
            w1 /= w_sum
            w2 /= w_sum

            uv_tri = self.uvs[self.faces_vt[hit_fids]]  # (K, 3, 2)
            uv_hit = (
                w0[:, None] * uv_tri[:, 0]
                + w1[:, None] * uv_tri[:, 1]
                + w2[:, None] * uv_tri[:, 2]
            )
            tx = np.clip(np.round(uv_hit[:, 0] * (tw - 1)).astype(np.int32), 0, tw - 1)
            ty = np.clip(np.round(uv_hit[:, 1] * (th - 1)).astype(np.int32), 0, th - 1)

            # Adaptive UV splat radius so high-res textures have solid, gap-free strokes
            uv_splat = max(1, int(round(max(tw, th) / 380.0)))
            if erase and self.orig_tex_bgr is not None:
                for dx in range(-uv_splat, uv_splat + 1):
                    for dy in range(-uv_splat, uv_splat + 1):
                        sx = np.clip(tx + dx, 0, tw - 1)
                        sy = np.clip(ty + dy, 0, th - 1)
                        self.tex_bgr[sy, sx] = self.orig_tex_bgr[sy, sx]
            else:
                col_arr = np.array(self.brush_color_bgr, dtype=np.uint8)
                for dx in range(-uv_splat, uv_splat + 1):
                    for dy in range(-uv_splat, uv_splat + 1):
                        sx = np.clip(tx + dx, 0, tw - 1)
                        sy = np.clip(ty + dy, 0, th - 1)
                        self.tex_bgr[sy, sx] = col_arr

            # Also paint UV triangle interiors of fully covered hit faces for seamless coverage
            if not erase:
                col_t = tuple(int(c) for c in self.brush_color_bgr)
                uv_polys = self.uvs[self.faces_vt[unique_fids]]
                uv_px = np.stack([
                    np.clip(np.round(uv_polys[:, :, 0] * (tw - 1)), 0, tw - 1),
                    np.clip(np.round(uv_polys[:, :, 1] * (th - 1)), 0, th - 1),
                ], axis=-1).astype(np.int32)
                for poly in uv_px:
                    cv2.fillConvexPoly(self.tex_bgr, poly, col_t)
            elif self.orig_tex_bgr is not None:
                mask_uv = np.zeros((th, tw), dtype=np.uint8)
                uv_polys = self.uvs[self.faces_vt[unique_fids]]
                uv_px = np.stack([
                    np.clip(np.round(uv_polys[:, :, 0] * (tw - 1)), 0, tw - 1),
                    np.clip(np.round(uv_polys[:, :, 1] * (th - 1)), 0, th - 1),
                ], axis=-1).astype(np.int32)
                for poly in uv_px:
                    cv2.fillConvexPoly(mask_uv, poly, 255)
                sel = mask_uv > 0
                self.tex_bgr[sel] = self.orig_tex_bgr[sel]

            self._refresh_face_colors(unique_fids)
        else:
            # Untextured mesh fallback: update per-face display colors directly
            if erase:
                self.face_colors_bgr[unique_fids] = (205, 205, 205)
            else:
                self.face_colors_bgr[unique_fids] = np.array(self.brush_color_bgr, dtype=np.uint8)

        # Immediate visual feedback on current color buffer (zero lag during stroke)
        if self._last_color_bgr is not None:
            if erase:
                self._last_color_bgr[y0:y1 + 1, x0:x1 + 1][hit_mask] = self.face_colors_bgr[hit_fids]
            else:
                self._last_color_bgr[y0:y1 + 1, x0:x1 + 1][hit_mask] = np.array(self.brush_color_bgr, dtype=np.uint8)
            self._blit_to_canvas()

        self._texture_dirty = True
        return True

    def _pick_color_3d(self, cx, cy):
        if self._last_color_bgr is None or self._last_face_id is None:
            return
        vx, vy = self._canvas_to_vp(cx, cy)
        vh, vw = self._vp_size
        ix, iy = int(round(vx)), int(round(vy))
        if 0 <= ix < vw and 0 <= iy < vh:
            fid = int(self._last_face_id[iy, ix])
            if fid >= 0 and self.face_colors_bgr is not None:
                bgr = tuple(int(c) for c in self.face_colors_bgr[fid])
                self.brush_color_bgr = bgr
                if self.on_color_picked:
                    self.on_color_picked(bgr)

    # ------------------------------------------------------------------
    # Mouse Events
    # ------------------------------------------------------------------

    def _on_left_down(self, event):
        if self.verts is None:
            return
        if self.mode == 'orbit':
            self._dragging_orbit = True
            self._drag_start_xy = (event.x, event.y)
            self._drag_start_R = self.camera.R.copy()
        elif self.mode == 'pick3d':
            self._pick_color_3d(event.x, event.y)
        elif self.mode in ('paint3d', 'erase3d'):
            if self._last_face_id is None:
                self.render(interactive=False)
            self._push_undo()
            self._painting = True
            vx, vy = self._canvas_to_vp(event.x, event.y)
            self._last_paint_xy = (vx, vy)
            self._paint_3d_at(vx, vy, erase=(self.mode == 'erase3d'))

    def _on_left_drag(self, event):
        if self._dragging_orbit and self._drag_start_xy is not None:
            cw = max(self.canvas.winfo_width(), 200)
            ch = max(self.canvas.winfo_height(), 200)
            self.camera.apply_arcball_drag(
                self._drag_start_R,
                self._drag_start_xy[0],
                self._drag_start_xy[1],
                event.x,
                event.y,
                cw,
                ch,
            )
            self.render(interactive=True)
            if self.on_camera_changed:
                self.on_camera_changed(self.camera.get_extrinsic())
        elif self._painting:
            self._draw_brush_cursor(event.x, event.y)
            vx, vy = self._canvas_to_vp(event.x, event.y)
            px, py = self._last_paint_xy or (vx, vy)
            dist = math.hypot(vx - px, vy - py)
            steps = max(1, int(dist / max(2.0, self.brush_radius * 0.25)))
            erase = (self.mode == 'erase3d')
            for i in range(1, steps + 1):
                t = i / steps
                self._paint_3d_at(px + (vx - px) * t, py + (vy - py) * t, erase=erase)
            self._last_paint_xy = (vx, vy)

    def _on_left_up(self, _event):
        if self._dragging_orbit:
            self._dragging_orbit = False
            self.render(interactive=False)
            if self.on_camera_changed:
                self.on_camera_changed(self.camera.get_extrinsic())
        if self._painting:
            self._painting = False
            self._last_paint_xy = None
            self.render(interactive=False)
            if self.on_texture_modified and self.tex_bgr is not None:
                self.on_texture_modified(self.tex_bgr)

    def _on_right_down(self, event):
        # Right-drag ALWAYS orbits smoothly even while in 3D Paint mode!
        if self.verts is None:
            return
        self._dragging_orbit = True
        self._drag_start_xy = (event.x, event.y)
        self._drag_start_R = self.camera.R.copy()

    def _on_right_drag(self, event):
        if self._dragging_orbit and self._drag_start_xy is not None:
            cw = max(self.canvas.winfo_width(), 200)
            ch = max(self.canvas.winfo_height(), 200)
            self.camera.apply_arcball_drag(
                self._drag_start_R,
                self._drag_start_xy[0],
                self._drag_start_xy[1],
                event.x,
                event.y,
                cw,
                ch,
            )
            self.render(interactive=True)
            if self.on_camera_changed:
                self.on_camera_changed(self.camera.get_extrinsic())

    def _on_right_up(self, _event):
        if self._dragging_orbit:
            self._dragging_orbit = False
            self.render(interactive=False)
            if self.on_camera_changed:
                self.on_camera_changed(self.camera.get_extrinsic())

    def _on_mid_down(self, event):
        if self.verts is None:
            return
        self._dragging_pan = True
        self._drag_start_xy = (event.x, event.y)

    def _on_mid_drag(self, event):
        if self._dragging_pan and self._drag_start_xy is not None:
            dx = event.x - self._drag_start_xy[0]
            dy = event.y - self._drag_start_xy[1]
            self._drag_start_xy = (event.x, event.y)
            self.camera.pan(dx, dy, self.canvas.winfo_width(), self.canvas.winfo_height())
            self.render(interactive=True)
            if self.on_camera_changed:
                self.on_camera_changed(self.camera.get_extrinsic())

    def _on_mid_up(self, _event):
        if self._dragging_pan:
            self._dragging_pan = False
            self.render(interactive=False)

    def _on_wheel(self, event):
        factor = 0.9 if event.delta > 0 else 1.11
        self._zoom_step(factor)

    def _zoom_step(self, factor):
        if self.verts is None:
            return
        self.camera.dolly(factor)
        self.render(interactive=True)
        if self._wheel_after_id is not None:
            self.after_cancel(self._wheel_after_id)
        self._wheel_after_id = self.after(90, lambda: self.render(interactive=False))
        if self.on_camera_changed:
            self.on_camera_changed(self.camera.get_extrinsic())

    def _on_resize(self, _event):
        if self.verts is not None:
            self._blit_to_canvas()

    def _on_mouse_move(self, event):
        self._draw_brush_cursor(event.x, event.y)
        if self.on_status_update and self.verts is not None:
            self.on_status_update(
                f"3D Viewport ({self.mode})  |  V: {len(self.verts)}  F: {len(self.faces_v)}  "
                f"|  Dist: {self.camera.distance:.2f}  |  Right-drag: Orbit  Middle-drag: Pan  Wheel: Zoom"
            )

    def _draw_brush_cursor(self, cx, cy):
        self.canvas.delete('cursor')
        if self.mode in ('paint3d', 'erase3d') and self.verts is not None:
            r = max(3, int(self.brush_radius))
            col = '#fbbf24' if self.mode == 'erase3d' else '#ffffff'
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=col, width=1.5, tags='cursor')
