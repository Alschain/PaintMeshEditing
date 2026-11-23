import os
import sys
import torch
import numpy as np
from torch import nn


def _ensure_env_tools_in_path():
    """Ensure conda/venv Scripts, Library/bin, and pip ninja bin are on PATH so
    nvdiffrast C++ extensions load reliably even when launched without `conda activate`.
    """
    env_dir = os.path.dirname(sys.executable)
    candidates = [
        env_dir,
        os.path.join(env_dir, 'Scripts'),
        os.path.join(env_dir, 'Library', 'bin'),
        os.path.join(env_dir, 'bin'),
    ]
    try:
        import ninja
        if hasattr(ninja, 'BIN_DIR') and ninja.BIN_DIR:
            candidates.insert(0, ninja.BIN_DIR)
    except Exception:
        pass

    cur_path = os.environ.get('PATH', '')
    cur_parts = set(p.lower() for p in cur_path.split(os.pathsep) if p)
    to_add = [c for c in candidates if os.path.isdir(c) and c.lower() not in cur_parts]
    if to_add:
        os.environ['PATH'] = os.pathsep.join(to_add) + os.pathsep + cur_path


_ensure_env_tools_in_path()
import nvdiffrast.torch as dr



class NvdiffrastMeshUlitRenderer(nn.Module):
    def __init__(self, args):
        super(NvdiffrastMeshUlitRenderer, self).__init__()
        self.args = args
        self.render_size = self.args['render_size']
        self.device = args['device']
        self.glctx = None

        self.extrinsic = torch.eye(4).to(self.device)
        self.intrinsic = torch.eye(4).to(self.device)
        self.ndc_proj = None


    def forward(self, mesh, antialias=True, render_normal=False):
        out = {}

        if self.ndc_proj is None:
            self.calc_ndc_projection()
        if self.glctx is None:
            if self.args.get('rastype', 'opengl') == 'opengl':
                try:
                    self.glctx = dr.RasterizeGLContext()
                except Exception:
                    self.glctx = dr.RasterizeCudaContext()
            else:
                self.glctx = dr.RasterizeCudaContext()

        if mesh.vs.shape[-1] == 3:
            ones = torch.ones((mesh.vs.shape[0], 1), dtype=mesh.vs.dtype, device=self.device)
            vertex = torch.cat([mesh.vs, ones], dim=-1).unsqueeze(0)
        else:
            vertex = mesh.vs.unsqueeze(0)
        vertex_ndc = torch.bmm(self.ndc_proj, vertex.permute(0, 2, 1)).permute(0, 2, 1)

        f_v_idx = mesh.f_v_idx.int().contiguous()
        rast_out, rast_db = dr.rasterize(self.glctx, vertex_ndc.contiguous(), f_v_idx, resolution=self.render_size)

        ndc_depth = rast_out[..., 2:3]
        mask = (rast_out[..., 3] > 0).float().unsqueeze(-1)

        img = None
        if mesh.vts is not None and mesh.tex is not None:
            tri_uv = mesh.f_vt_idx.int().contiguous() if mesh.f_vt_idx is not None else f_v_idx
            interp_out, interp_out_db = dr.interpolate(
                mesh.vts.contiguous(), rast_out, tri_uv, rast_db, diff_attrs='all'
            )
            img = dr.texture(mesh.tex.unsqueeze(0).contiguous(), interp_out, interp_out_db)
            img = img * mask

        if render_normal:
            mesh.update_normal()
            interp_out, _ = dr.interpolate(mesh.vns.contiguous(), rast_out, f_v_idx, rast_db, diff_attrs='all')
            interp_out = torch.nn.functional.normalize(interp_out, p=2, dim=-1, eps=1e-6)
            out['normal'] = interp_out * mask

        depth = self.calc_ndc_to_depth(ndc_depth)
        depth = mask * depth
        if antialias:
            if img is not None:
                img = dr.antialias(img.contiguous(), rast_out, vertex_ndc.contiguous(), f_v_idx)
            if render_normal and 'normal' in out:
                out['normal'] = dr.antialias(out['normal'].contiguous(), rast_out, vertex_ndc.contiguous(), f_v_idx)
            mask_aa = dr.antialias(mask.contiguous(), rast_out, vertex_ndc.contiguous(), f_v_idx)
        else:
            mask_aa = mask

        out['mask'] = mask
        out['silhouette'] = mask_aa
        out['depth'] = depth
        out['color'] = img
        out['face_id'] = rast_out[..., 3].long() - 1

        return out

    def set_render_size(self, render_size):
        size = (int(render_size[0]), int(render_size[1]))
        if tuple(self.render_size) != size:
            self.render_size = size
            self.ndc_proj = None

    def set_camera_intrinsic(self, intrinsic):
        if isinstance(intrinsic, np.ndarray):
            intrinsic = torch.FloatTensor(intrinsic)

        self.intrinsic = intrinsic.float().reshape(4, 4).to(self.device)
        self.ndc_proj = None

    def set_camera_extrinsic(self, extrinsic):
        if isinstance(extrinsic, np.ndarray):
            extrinsic = torch.FloatTensor(extrinsic)

        self.extrinsic = extrinsic.float().reshape(4, 4).to(self.device)
        self.ndc_proj = None

    def calc_ndc_projection(self):
        P = self.calc_projection(
            self.intrinsic[0][0], self.intrinsic[1][1],
            self.intrinsic[0][2], self.intrinsic[1][2],
            self.render_size[1], self.render_size[0]
        ).to(self.device)
        extrinsic = self.extrinsic.to(self.device)
        self.ndc_proj = torch.matmul(P, extrinsic).unsqueeze(0).to(self.device)
        return
      
    def calc_ndc_to_depth(self, ndc_z, near=0.1, far=20.0):
        return (2 * near * far) / (far + near - ndc_z * (far - near))
    
    def calc_projection(self, fx, fy, cx, cy, w, h, near=0.1, far=20.0):
        fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)
        w, h = float(w), float(h)
        P = torch.tensor([
            [2.0 * fx / w, 0.0, (2.0 * cx - w) / w, 0.0],
            [0.0, 2.0 * fy / h, (2.0 * cy - h) / h, 0.0],
            [0.0, 0.0, (far + near) / (far - near), -(2.0 * far * near) / (far - near)],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=torch.float32)

        return P


