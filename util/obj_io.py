import os
import cv2
import torch

import numpy as np

from util.mesh import Mesh


def _parse_face_token(token, num_v, num_vt, num_vn):
    parts = token.split('/')
    v = int(parts[0])
    v_idx = v - 1 if v > 0 else num_v + v

    vt_idx = -1
    if len(parts) > 1 and parts[1] != "":
        vt = int(parts[1])
        vt_idx = vt - 1 if vt > 0 else num_vt + vt

    vn_idx = -1
    if len(parts) > 2 and parts[2] != "":
        vn = int(parts[2])
        vn_idx = vn - 1 if vn > 0 else num_vn + vn

    return v_idx, vt_idx, vn_idx


def load_obj(obj_path, tex_path=None, device='cuda'):
    with open(obj_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # verts attributes
    verts, uvs, vert_normals = [], [], []
    for line in lines:
        parts = line.split()
        if len(parts) == 0:
            continue

        prefix = parts[0].lower()
        if prefix == 'v':
            verts.append([float(v) for v in parts[1:4]])
        elif prefix == 'vt':
            val = [float(v) for v in parts[1:3]]
            uvs.append([val[0], 1.0 - val[1]])
        elif prefix == 'vn':
            vert_normals.append([float(v) for v in parts[1:4]])

    num_v, num_vt, num_vn = len(verts), len(uvs), len(vert_normals)

    # load faces
    f_v_idx, f_vt_idx, f_vn_idx = [], [], []
    has_vt_idx, has_vn_idx = False, False
    for line in lines:
        parts = line.split()
        if len(parts) == 0:
            continue
        prefix = parts[0].lower()
        if prefix == 'f':
            vs = parts[1:]
            nv = len(vs)
            if nv < 3:
                continue
            v0, t0, n0 = _parse_face_token(vs[0], num_v, num_vt, num_vn)
            for i in range(nv - 2):
                v1, t1, n1 = _parse_face_token(vs[i + 1], num_v, num_vt, num_vn)
                v2, t2, n2 = _parse_face_token(vs[i + 2], num_v, num_vt, num_vn)
                f_v_idx.append([v0, v1, v2])
                f_vt_idx.append([t0, t1, t2])
                f_vn_idx.append([n0, n1, n2])
                if t0 >= 0 and t1 >= 0 and t2 >= 0:
                    has_vt_idx = True
                if n0 >= 0 and n1 >= 0 and n2 >= 0:
                    has_vn_idx = True

    verts = torch.tensor(verts, dtype=torch.float32, device=device)
    uvs = torch.tensor(uvs, dtype=torch.float32, device=device) if len(uvs) > 0 else None
    vert_normals = torch.tensor(vert_normals, dtype=torch.float32, device=device) if len(vert_normals) > 0 else None

    f_v_idx = torch.tensor(f_v_idx, dtype=torch.int32, device=device).contiguous()
    f_vt_idx = torch.tensor(f_vt_idx, dtype=torch.int32, device=device).contiguous() if (uvs is not None and has_vt_idx) else None
    f_vn_idx = torch.tensor(f_vn_idx, dtype=torch.int32, device=device).contiguous() if (vert_normals is not None and has_vn_idx) else None

    texture = None
    if tex_path is not None and os.path.exists(tex_path):
        tex_img = cv2.imread(tex_path, -1)
        if tex_img is not None:
            if tex_img.ndim == 2:
                tex_img = cv2.cvtColor(tex_img, cv2.COLOR_GRAY2BGR)
            texture = torch.tensor(tex_img[..., :3], dtype=torch.float32, device=device).contiguous() / 255.

    return Mesh(vs=verts, vns=vert_normals, vts=uvs, vcs=None, f_v_idx=f_v_idx, f_vt_idx=f_vt_idx, f_vn_idx=f_vn_idx, tex=texture)

def save_obj(obj_path, mesh):
    parent_dir = os.path.dirname(obj_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    obj_name = os.path.splitext(os.path.basename(obj_path))[0]
    print("Writing mesh: ", obj_path)
    with open(obj_path, "w", encoding="utf-8") as f:
        if mesh.tex is not None:
            f.write(f"mtllib {obj_name}.mtl\n")
            f.write(f"usemtl {obj_name}\n")

        vs = mesh.vs.detach().cpu().numpy() if mesh.vs is not None else None
        vns = mesh.vns.detach().cpu().numpy() if mesh.vns is not None else None
        vts = mesh.vts.detach().cpu().numpy() if mesh.vts is not None else None

        f_v_idx = mesh.f_v_idx.detach().cpu().numpy() if mesh.f_v_idx is not None else None
        f_vn_idx = mesh.f_vn_idx.detach().cpu().numpy() if mesh.f_vn_idx is not None else None
        f_vt_idx = mesh.f_vt_idx.detach().cpu().numpy() if mesh.f_vt_idx is not None else None

        print("    writing %d vertices" % (len(vs) if vs is not None else 0))
        if vs is not None:
            for v in vs:
                f.write('v {} {} {} \n'.format(v[0], v[1], v[2]))

        if vts is not None:
            print("    writing %d texcoords" % len(vts))
            assert len(f_v_idx) == len(f_vt_idx)
            for v in vts:
                f.write('vt {} {} \n'.format(v[0], 1.0 - v[1]))

        if vns is not None:
            print("    writing %d normals" % len(vns))
            assert len(f_v_idx) == len(f_vn_idx)
            for v in vns:
                f.write('vn {} {} {}\n'.format(v[0], v[1], v[2]))

        # Write faces
        print("    writing %d faces" % len(f_v_idx))
        for i in range(len(f_v_idx)):
            f.write("f ")
            for j in range(3):
                v_str = str(int(f_v_idx[i][j]) + 1)
                vt_str = '' if (vts is None or f_vt_idx is None) else str(int(f_vt_idx[i][j]) + 1)
                vn_str = '' if (vns is None or f_vn_idx is None) else str(int(f_vn_idx[i][j]) + 1)
                f.write(' %s/%s/%s' % (v_str, vt_str, vn_str))
            f.write("\n")

    if mesh.tex is not None:
        tex_file = os.path.join(parent_dir, obj_name + '.png') if parent_dir else (obj_name + '.png')
        tex = mesh.tex.detach().cpu().numpy()
        if np.max(tex) <= 1.0:
            tex = tex * 255.0
        cv2.imwrite(tex_file, np.clip(tex, 0, 255).astype(np.uint8))

        mtl_file = os.path.join(parent_dir, obj_name + '.mtl') if parent_dir else (obj_name + '.mtl')
        print("Writing material: ", mtl_file)
        with open(mtl_file, 'w', encoding='utf-8') as f:
            f.write(f'newmtl {obj_name}\n')
            f.write('Kd 1 1 1\n')
            f.write('Ks 0 0 0\n')
            f.write('Ka 0 0 0\n')
            f.write(f'map_Kd {obj_name}.png\n')
    return