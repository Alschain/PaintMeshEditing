import torch
from util.mesh_util import *

blendshapes_names = [
            "mouthClose", "mouthDimple_L", "mouthDimple_R", "mouthFrown_L", "mouthFrown_R",
            "mouthFunnel", "mouthLeft", "mouthLowerDown_L", "mouthLowerDown_R", "mouthPress_L", "mouthPress_R",
            "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
            "mouthSmile_L", "mouthSmile_R", "mouthStretch_L", "mouthStretch_R", "mouthUpperUp_L", "mouthUpperUp_R",
            "noseSneer_L", "noseSneer_R", "jawForward", "jawLeft", "jawOpen", "jawRight",
            "browDown_L", "browDown_R", "browInnerUp_L", "browInnerUp_R", "browOuterUp_L", "browOuterUp_R",
            "cheekPuff_L", "cheekPuff_R", "cheekRaiser_L", "cheekRaiser_R", "cheekSquint_L", "cheekSquint_R",
            "eyeBlink_L", "eyeBlink_R", "eyeLookDown_L", "eyeLookDown_R", "eyeLookIn_L", "eyeLookIn_R",
            "eyeLookOut_L", "eyeLookOut_R", "eyeLookUp_L", "eyeLookUp_R", "eyeSquint_L", "eyeSquint_R",
            "eyeWide_L", "eyeWide_R", "generic_neutral_mesh", "PupilDilate_L", "PupilDilate_R"
            ]

class Mesh:
    def __init__(self, vs=None, vns=None, vts=None, vcs=None, f_v_idx=None, f_vt_idx=None, f_vn_idx=None, tex=None):
        '''
        vs          :   vertex positions
        vns         :   vertex normals
        vts         :   uv coordinates
        vcs         :   vertex colors
        f_v_idx     :   triangle vertex position index
        f_vt_idx    :   triangle uv coordinates index
        f_vn_idx    :   triangle vertex normal index
        tex         :   texture map
        '''
        self.vs = vs
        self.vns = vns
        self.vts = vts
        self.vcs = vcs
        
        self.f_ns = None
        self.f_v_idx = f_v_idx
        self.f_vt_idx = f_vt_idx if f_vt_idx is not None else f_v_idx
        self.f_vn_idx = f_vn_idx if f_vn_idx is not None else f_v_idx

        self.tex = tex

        self.edges = find_edges(self.f_v_idx, remove_duplicates=True) if self.f_v_idx is not None else None
        self.connected_faces = find_connected_faces(self.f_v_idx) if self.f_v_idx is not None else None


    def clone(self):
        out = Mesh()
        if self.vs is not None:
            out.vs = self.vs.clone()
        if self.vns is not None:
            out.vns = self.vns.clone()
        if self.vts is not None:
            out.vts = self.vts.clone()
        if self.vcs is not None:
            out.vcs = self.vcs.clone()


        if self.f_ns is not None:
            out.f_ns = self.f_ns.clone()        
        if self.f_v_idx is not None:
            out.f_v_idx = self.f_v_idx.clone()
        if self.f_vt_idx is not None:
            out.f_vt_idx = self.f_vt_idx.clone()
        if self.f_vn_idx is not None:
            out.f_vn_idx = self.f_vn_idx.clone()
        
        if self.tex is not None:
            out.tex = self.tex.clone()

        if self.edges is not None:
            out.edges = self.edges.clone()
        if self.connected_faces is not None:
            out.connected_faces = self.connected_faces.clone()
        return out

    def update_normal(self):
        self.vns, self.f_ns = compute_normals(self.vs, self.f_v_idx)


class MeshWithBlendShapes():
    def __init__(self, base_mesh, blendshapes_names, bs_meshes:dict):
        self.base_mesh = base_mesh.clone()
        self.device = base_mesh.vs.device

        self.blendshapes_names = blendshapes_names
        
        self.bs_meshes = {}
        for key in self.blendshapes_names:
            if key in bs_meshes.keys():
                self.bs_meshes[key] = bs_meshes[key].clone()
            else:
                self.bs_meshes[key] = None

        
        self.bs_offsets           = {}
        self.trainable_indices    = {}
        self.trainable_parameters = {}

        self.get_bs_parameters()
    
    def get_bs_parameters(self, eps=1e-6):
        for bs_name, bs_mesh in self.bs_meshes.items():
            if bs_mesh is not None:
                transformed_vertices_placeholder = torch.sqrt(torch.sum((bs_mesh.vs - self.base_mesh.vs)**2, dim=-1)) > eps
                transformed_vertices_index = transformed_vertices_placeholder.nonzero(as_tuple=True)[0].long()
                if transformed_vertices_index.shape[0] != 0:
                    self.bs_offsets[bs_name] = (bs_mesh.vs - self.base_mesh.vs)[transformed_vertices_index]
                    self.trainable_indices[bs_name] = transformed_vertices_index
                    self.trainable_parameters[bs_name] = torch.ones(transformed_vertices_index.shape[0]).to(self.device)
                else:
                    self.bs_offsets[bs_name] = None
                    self.trainable_indices[bs_name] = None
                    self.trainable_parameters[bs_name] = None
        return
    
    def set_bs_trainable(self):
        for key in self.trainable_parameters.keys():
            if key is not None:
                self.trainable_parameters[key].requires_grad_(True)
        return
    
    def set_bs_untrainable(self):
        for key in self.trainable_parameters.keys():
            if key is not None:
                self.trainable_parameters[key].requires_grad_(False)
        return

    def clone(self):
        out = MeshWithBlendShapes(self.base_mesh, self.blendshapes_names, self.bs_meshes)

        out.bs_offsets = {}
        out.trainable_indices = {}
        out.trainable_parameters = {}

        for key in self.blendshapes_names:
            out.bs_offsets[key] = self.bs_offsets[key].clone() if self.bs_offsets[key] is not None else None
            out.trainable_indices[key] = self.trainable_indices[key].clone() if self.trainable_indices[key] is not None else None
            out.trainable_parameters[key] = self.trainable_parameters[key].clone() if self.trainable_parameters[key] is not None else None

        return out        

    def apply_bs_weights(self, bs_weights:dict):
        ret_mesh = self.base_mesh.clone()
        for bs_name in self.blendshapes_names:
            if self.bs_offsets[bs_name] is not None:
                current_weight = 0.
                if bs_name in bs_weights.keys():
                    current_weight = bs_weights[bs_name]
                
                ret_mesh.vs[self.trainable_indices[bs_name]] = \
                        ret_mesh.vs[self.trainable_indices[bs_name]] + \
                        current_weight * self.bs_offsets[bs_name] * self.trainable_parameters[bs_name].detach().unsqueeze(-1)

        return ret_mesh
