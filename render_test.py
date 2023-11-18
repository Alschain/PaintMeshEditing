
import numpy as np

from util.obj_io import load_obj, save_obj
from render.ulit_renderer import NvdiffrastMeshUlitRenderer

if __name__ == '__main__':
    import cv2

    args = {
        # global params
        'device' : 'cuda',
        'rastype': 'opengl',

        # rasterization settings
        'render_size': (1024, 1024), # H x W
        }

    
    mesh_path = './sample_data/mesh/cow.obj'
    texture_path = './sample_data/mesh/cow_texture_with_label.png'
    intr_path = './sample_data/intr.txt'
    extr_path = './sample_data/extr.txt'

    intrinsic = np.loadtxt(intr_path).reshape(4,4)
    extrinsic = np.loadtxt(extr_path).reshape(4,4)

    mesh = load_obj(mesh_path, texture_path, 'cuda')

    renderer = NvdiffrastMeshUlitRenderer(args)
    renderer.set_camera_intrinsic(intrinsic)
    renderer.set_camera_extrinsic(extrinsic)

    mask, depth, image = renderer(mesh)
    image = image[0].cpu().numpy()
    depth = depth[0].cpu().numpy()
    cv2.imwrite('nv_color.png', image.astype(np.uint8))
    cv2.imwrite('nv_depth.png', (depth * 10000).astype(np.ushort))
