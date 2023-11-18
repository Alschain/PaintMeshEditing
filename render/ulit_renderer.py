import torch
import numpy as np
from torch import nn
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


    def forward(self, mesh, antialias=True):
        if self.ndc_proj is None:
            self.calc_ndc_projection()
        if self.glctx is None:
            self.glctx = dr.RasterizeGLContext() if 'opengl' != self.args['rastype'] else dr.RasterizeCudaContext()

        if mesh.vs.shape[-1] == 3:
            vertex = torch.cat([mesh.vs, torch.ones([mesh.vs.shape[0], 1]).to(self.device)], dim=-1).unsqueeze(0)
        vertex_ndc = torch.bmm(self.ndc_proj, vertex.permute(0,2,1)).permute(0,2,1)

        rast_out, rast_db = dr.rasterize(self.glctx, vertex_ndc.contiguous(), mesh.f_v_idx, resolution=self.render_size)

        ndc_depth = rast_out[...,2:3]
        mask = (rast_out[..., 3] > 0).float().unsqueeze(-1)
        

        img = None
        if mesh.vts is not None and mesh.tex is not None:
            if mesh.f_vt_idx is not None:
                interp_out, inpterp_out_db = dr.interpolate(mesh.vts, rast_out, mesh.f_vt_idx, rast_db, diff_attrs='all')
            else:
                interp_out, inpterp_out_db = dr.interpolate(mesh.vts, rast_out, mesh.f_v_idx, rast_db, diff_attrs='all')
                
            img = dr.texture(mesh.tex.unsqueeze(0), interp_out, inpterp_out_db)
            img = img * mask

        depth = self.calc_ndc_to_depth(ndc_depth)
        depth = mask * depth
        if antialias:
            img = dr.antialias(img.contiguous(), rast_out, vertex_ndc.contiguous(), mesh.f_v_idx.int())
        
        return mask, depth, img

    def set_camera_intrinsic(self, intrinsic):
        if type(intrinsic) == np.ndarray:
            intrinsic = torch.FloatTensor(intrinsic)

        self.intrinsic = intrinsic.reshape(4,4)

    def set_camera_extrinsic(self, extrinsic):
        if type(extrinsic) == np.ndarray:
            extrinsic = torch.FloatTensor(extrinsic)

        self.extrinsic = extrinsic.reshape(4,4)

    def calc_ndc_projection(self):
        P = self.calc_projection(
            self.intrinsic[0][0], self.intrinsic[1][1],
            self.intrinsic[0][2], self.intrinsic[1][2],
            self.render_size[1], self.render_size[0]
            )
        self.ndc_proj = torch.matmul(P, self.extrinsic).unsqueeze(0).to(self.device)
        return
      
    def calc_ndc_to_depth(self, ndc_z, near=0.1, far=20.0):
        return (2 * near * far) / (far + near - ndc_z * (far - near))
    
    def calc_projection(self, fx, fy, cx, cy, w, h, near=0.1, far=20.0):
        P = torch.FloatTensor([
            [2*fx/w, 0, (2*cx-w)/w, 0],
            [0, 2*fy/h, (2*cy-h)/h, 0],
            [0, 0, (far+near)/(far-near), -(2*far*near)/(far-near)],
            [0, 0, 1, 0]])
        
        return P


