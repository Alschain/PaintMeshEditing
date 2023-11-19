import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.losses import *
from util.mesh_util import *
from util.vis_util import *
from util.obj_io import *

from tqdm import tqdm

class PaintMeshEditing:
    def __init__(self, args={}):

        self.iteration = 2000 if 'iteration' not in args.keys() else args['iteration']
        self.lr = 3e-4 if 'lr' not in args.keys() else args['lr']
        self.log_interval = 10 if 'log_interval' not in args.keys() else args['log_interval']
        self.save_interval = 20 if 'save_interval' not in args.keys() else args['save_iterval']
        self.display_image = True if 'display_image' not in args.keys() else args['display_image']
        self.save_mesh_interval = 100 if 'save_mesh_interval' not in args.keys() else args['save_mesh_interval']

        self.SCALE = 1 if 'SCALE' not in args.keys() else args['SCALE']
        self.USE_BLUR = False if 'USE_BLUR' not in args.keys() else args['USE_BLUR']
        self.BLUR_KERNEL_SIZE = 9 if 'BLUR_KERNEL_SIZE' not in args.keys() else args['BLUR_KERNEL_SIZE']


    def optimize(self, mesh, label_image, mask_image, renderer, normal_image=None, out_dir='./output'):
        os.makedirs(out_dir, exist_ok=True)
        ori_vertex_normals, ori_face_normals = compute_normals(mesh.vs.detach(), mesh.f_v_idx)
        ori_link_nc = (ori_vertex_normals[mesh.edges[:,0]] - ori_vertex_normals[mesh.edges[:,1]])
        ori_link_face_nc = (ori_face_normals[mesh.connected_faces[:,0]] - ori_face_normals[mesh.connected_faces[:,1]])
        ori_face_area = compute_triangle_area(mesh.vs.clone().detach(), mesh.f_v_idx)
        ori_edge_len = compute_edge_length(mesh.vs.clone().detach(), mesh.edges)

        optimizer  = torch.optim.Adam([mesh.vs], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.0, 10**(-x*0.0002))) 

        loss_weights = {'img_loss':100, 'nc_loss':10,'edge_loss':100,'area_loss':100000,'lap_loss':0, 'normal_loss':0.1}
        img_cnt = 0
        # for it in tqdm(range(self.iteration)):
        for it in range(self.iteration):
            losses = {}
            render_result = renderer(mesh, render_normal=(normal_image is not None))

            image = render_result['color'][0] * mask_image
            losses['img_loss'] = F.l1_loss(image, label_image)

            laplacian = compute_laplacian_uniform(mesh.vs, mesh.edges)
            losses['lap_loss'] = laplacian_loss(laplacian, mesh.vs)

            vertex_normals, face_normals = compute_normals(mesh.vs, mesh.f_v_idx)
            losses['nc_loss'] = (((vertex_normals[mesh.edges[:,0]] - vertex_normals[mesh.edges[:,1]]) - ori_link_nc) ** 2).mean()+\
                    (((face_normals[mesh.connected_faces[:,0]] - face_normals[mesh.connected_faces[:,1]]) - ori_link_face_nc)**2).mean()

            edges_len = compute_edge_length(mesh.vs, mesh.edges)
            face_area = compute_triangle_area(mesh.vs, mesh.f_v_idx)
            losses['edge_loss'] = ((edges_len - ori_edge_len)**2).mean()
            losses['area_loss'] = ((face_area - ori_face_area)**2).mean()

            if normal_image is not None:
                normal_map = render_result['normal'][0] * mask_image
                losses['normal_loss'] = F.l1_loss(normal_map, normal_image*mask_image)


            # TODO: adaptive adjustment
            # for key, value in losses.items():
            #     if it == 0:
            #         ratio = 0.6 / value.item() if value.item() != 0 else 1
            #         loss_weights[key] = ratio * 0.25
            #         loss_weights['min_'+key] = loss_weights[key] * 0.02
            #     else:
            #         loss_weights[key] = (loss_weights[key] - loss_weights['min_'+key]) * 10**(-it*0.000001) + loss_weights['min_'+key]


            all_loss = 0.
            for key, value in losses.items():
                all_loss = all_loss + value * loss_weights[key]

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            scheduler.step()

            if self.log_interval and ((it+1) % self.log_interval) == 0:
                logline = f'iter \t{it},'
                for key, value in losses.items():
                    logline = logline + '\t {0}:{1:.6f},'.format(key, value.item()*loss_weights[key])
                logline = logline[:-1]
                print(logline)

            if self.display_image or self.save_interval:
                # Render images, don't need to track any gradients
                with torch.no_grad():
                    # Render
                    render_result = renderer(mesh, render_normal=(normal_image is not None))

                    image = render_result['color'][0].detach().cpu()
                    img_gt = label_image.detach().cpu()

                    if normal_image is not None:
                        normal_map = render_result['normal'][0].detach().cpu()
                        normal_map_show = vis_normal_map(normal_map) / 255.
                        result_image = torch.cat([image, normal_map_show, img_gt], axis=1)
                    else:
                        result_image = torch.cat([image, img_gt], axis=1)

                np_result_image = result_image.detach().cpu().numpy() * 255
                if self.display_image:
                    cv2.imshow('display', np_result_image.astype(np.uint8))
                    cv2.waitKey(1)
                if self.save_interval and (it % self.save_interval) == 0:
                    cv2.imwrite(out_dir + '/' + ('img_%06d.png' % img_cnt), np_result_image)
                    img_cnt = img_cnt+1

            if ((it + 1) % self.save_mesh_interval) == 0:
                save_obj(out_dir + '/mesh/' + ('mesh_%06d.obj' % img_cnt), mesh)


        return