
import torch
import numpy as np

from util.obj_io import load_obj, save_obj
from render.ulit_renderer import NvdiffrastMeshUlitRenderer
from core.editing import PaintMeshEditing

if __name__ == '__main__':
    import cv2
    device = 'cuda'
    args = {
        # global params
        'device' : 'cuda',
        'rastype': 'opengl',

        # rasterization settings
        'render_size': (1024, 1024), # H x W
        }

    
    mesh_path = './sample_data/mesh/cow.obj'
    texture_path = './sample_data/mesh/cow_texture_with_label.png'
    label_path = './sample_data/label.png'
    mask_path = './sample_data/mask.png'
    intr_path = './sample_data/intr.txt'
    extr_path = './sample_data/extr.txt'
    intrinsic = np.loadtxt(intr_path).reshape(4,4)
    extrinsic = np.loadtxt(extr_path).reshape(4,4)

    mesh = load_obj(mesh_path, texture_path, 'cuda')
    mesh.vs.requires_grad_(True)


    renderer = NvdiffrastMeshUlitRenderer(args)
    renderer.set_camera_intrinsic(intrinsic)
    renderer.set_camera_extrinsic(extrinsic)

    mask = torch.FloatTensor(cv2.imread(mask_path, -1)).to(device).unsqueeze(-1) / 255
    label = torch.FloatTensor(cv2.imread(label_path, -1)[...,:3]).to(device) / 255.


    pme = PaintMeshEditing()

    pme.optimize(mesh, label, mask, renderer)

    save_obj('./output/deformed.obj', mesh)
