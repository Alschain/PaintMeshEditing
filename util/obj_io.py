import os
import cv2
import torch

import numpy as np

from util.mesh import Mesh


def load_obj(obj_path, tex_path=None, device='cuda'):
    with open(obj_path) as f:
        lines = f.readlines()

    # verts attributes
    verts, uvs, vert_normals  = [], [], []
    for line in lines:
        if len(line.split()) == 0:
            continue
        
        prefix = line.split()[0].lower()
        if prefix == 'v':
            verts.append([float(v) for v in line.split()[1:]])
        elif prefix == 'vt':
            val = [float(v) for v in line.split()[1:]]
            uvs.append([val[0], 1.0 - val[1]])
        elif prefix == 'vn':
            vert_normals.append([float(v) for v in line.split()[1:]])

    uv = False if len(uvs) == 0 else True
    nor = False if len(vert_normals) == 0 else True

    # load faces
    f_v_idx, f_vt_idx, f_vn_idx = [], [], []

    if uv and nor:
        for line in lines:
            if len(line.split()) == 0:
                continue
            prefix = line.split()[0].lower()
            if prefix == 'f':
                vs = line.split()[1:]
                nv = len(vs)
                vv = vs[0].split('/')
                v0 = int(vv[0]) - 1
                t0 = int(vv[1]) - 1 if vv[1] != "" else -1
                n0 = int(vv[2]) - 1 if vv[2] != "" else -1
                for i in range(nv - 2):
                    vv = vs[i + 1].split('/')
                    v1 = int(vv[0]) - 1
                    t1 = int(vv[1]) - 1 if vv[1] != "" else -1
                    n1 = int(vv[2]) - 1 if vv[2] != "" else -1
                    vv = vs[i + 2].split('/')
                    v2 = int(vv[0]) - 1
                    t2 = int(vv[1]) - 1 if vv[1] != "" else -1
                    n2 = int(vv[2]) - 1 if vv[2] != "" else -1
                    f_v_idx.append([v0, v1, v2])
                    f_vt_idx.append([t0, t1, t2])
                    f_vn_idx.append([n0, n1, n2])
        assert len(f_v_idx) == len(f_vt_idx) and len(f_v_idx) == len (f_vn_idx)
    elif uv:
        for line in lines:
            if len(line.split()) == 0:
                continue
            prefix = line.split()[0].lower()
            if prefix == 'f':
                vs = line.split()[1:]
                nv = len(vs)
                vv = vs[0].split('/')
                v0 = int(vv[0]) - 1
                t0 = int(vv[1]) - 1 if vv[1] != "" else -1
                for i in range(nv - 2):
                    vv = vs[i + 1].split('/')
                    v1 = int(vv[0]) - 1
                    t1 = int(vv[1]) - 1 if vv[1] != "" else -1
                    vv = vs[i + 2].split('/')
                    v2 = int(vv[0]) - 1
                    t2 = int(vv[1]) - 1 if vv[1] != "" else -1
                    f_v_idx.append([v0, v1, v2])
                    f_vt_idx.append([t0, t1, t2])
        assert len(f_v_idx) == len(f_vt_idx)
    elif nor:
        for line in lines:
            if len(line.split()) == 0:
                continue
            prefix = line.split()[0].lower()
            if prefix == 'f':
                vs = line.split()[1:]
                nv = len(vs)
                vv = vs[0].split('/')
                v0 = int(vv[0]) - 1
                n0 = int(vv[1]) - 1 if vv[1] != "" else -1
                for i in range(nv - 2):
                    vv = vs[i + 1].split('/')
                    v1 = int(vv[0]) - 1
                    n1 = int(vv[1]) - 1 if vv[1] != "" else -1
                    vv = vs[i + 2].split('/')
                    v2 = int(vv[0]) - 1
                    n2 = int(vv[1]) - 1 if vv[1] != "" else -1
                    f_v_idx.append([v0, v1, v2])
                    f_vn_idx.append([n0, n1, n2])
        assert len(f_v_idx) == len (f_vn_idx)
    else:
        for line in lines:
            if len(line.split()) == 0:
                continue
            prefix = line.split()[0].lower()
            if prefix == 'f':
                vs = line.split()[1:]
                nv = len(vs)
                vv = vs[0].split('/')
                v0 = int(vv[0]) - 1
                for i in range(nv - 2):
                    vv = vs[i + 1].split('/')
                    v1 = int(vv[0]) - 1
                    vv = vs[i + 2].split('/')
                    v2 = int(vv[0]) - 1
                    f_v_idx.append([v0, v1, v2])


    verts = torch.tensor(verts, dtype=torch.float32, device=device)
    uvs = torch.tensor(uvs, dtype=torch.float32, device=device) if len(uvs) > 0 else None
    vert_normals = torch.tensor(vert_normals, dtype=torch.float32, device=device) if len(vert_normals) > 0 else None
    
    f_v_idx = torch.tensor(f_v_idx, dtype=torch.int32, device=device).contiguous()
    f_vt_idx = torch.tensor(f_vt_idx, dtype=torch.int32, device=device).contiguous() if len(f_vt_idx) != 0 else None
    f_vn_idx = torch.tensor(f_vn_idx, dtype=torch.int32, device=device).contiguous() if len(f_vn_idx) != 0 else None

    texture = torch.tensor(cv2.imread(tex_path, -1)[...,:3], dtype=torch.float32, device=device).contiguous() / 255. if tex_path is not None else None

    return Mesh(vs=verts, vns=vert_normals, vts=uvs, vcs=None, f_v_idx=f_v_idx, f_vt_idx=f_vt_idx, f_vn_idx=f_vn_idx, tex=texture)

def save_obj(obj_path, mesh):
    parent_dir = os.path.dirname(obj_path)
    os.makedirs(parent_dir, exist_ok=True)
    obj_name = os.path.basename(obj_path).split('.')[0]
    print("Writing mesh: ", obj_path)
    with open(obj_path, "w") as f:
        f.write(f"mtllib {obj_name}.mtl\n")

        vs = mesh.vs.detach().cpu().numpy() if mesh.vs is not None else None
        vns = mesh.vns.detach().cpu().numpy() if mesh.vns is not None else None
        vts = mesh.vts.detach().cpu().numpy() if mesh.vts is not None else None

        f_v_idx = mesh.f_v_idx.detach().cpu().numpy() if mesh.f_v_idx is not None else None
        f_vn_idx = mesh.f_vn_idx.detach().cpu().numpy() if mesh.f_vn_idx is not None else None
        f_vt_idx = mesh.f_vt_idx.detach().cpu().numpy() if mesh.f_vt_idx is not None else None

        print("    writing %d vertices" % len(vs))
        for v in vs:
            f.write('v {} {} {} \n'.format(v[0], v[1], v[2]))
       
        if vts is not None:
            print("    writing %d texcoords" % len(vts))
            assert(len(f_v_idx) == len(f_vt_idx))
            for v in vts:
                f.write('vt {} {} \n'.format(v[0], 1.0 - v[1]))

        if vns is not None:
            print("    writing %d normals" % len(vns))
            assert(len(f_v_idx) == len(f_vn_idx))
            for v in vns:
                f.write('vn {} {} {}\n'.format(v[0], v[1], v[2]))


        # Write faces
        print("    writing %d faces" % len(f_v_idx))
        for i in range(len(f_v_idx)):
            f.write("f ")
            if vts is not None and vns is not None:
                for j in range(3):
                    f.write(' %s/%s/%s' % (str(f_v_idx[i][j]+1), str(f_vt_idx[i][j]+1), str(f_vn_idx[i][j]+1)))
            elif vts is not None and vns is None:
                for j in range(3):
                    f.write(' %s/%s' % (str(f_v_idx[i][j]+1), str(f_vt_idx[i][j]+1)))
            elif vns is not None and vts is None:
                for j in range(3):
                    f.write(' %s/%s' % (str(f_v_idx[i][j]+1), str(f_vn_idx[i][j]+1)))
            else:
                for j in range(3):
                    f.write(' %s' % (str(f_v_idx[i][j]+1)))

            
            f.write("\n")

    if mesh.tex is not None:
        tex_file = os.path.join(parent_dir, obj_name + '.png')
        tex = mesh.tex.detach().cpu().numpy()
        if np.max(tex) <= 1:
            tex = tex * 255
        cv2.imwrite(tex_file, tex.astype(np.uint8))


        mtl_file = os.path.join(parent_dir, obj_name + '.mtl')
        print("Writing material: ", mtl_file)
        with open(mtl_file, 'w') as f:
            f.write(f'newmtl {obj_name}\n')
            f.write('Kd 1 1 1\n')
            f.write('Ks 0 0 0\n')
            f.write('Ka 0 0 0\n')
            f.write(f'map_Kd {obj_name}.png')
    return
