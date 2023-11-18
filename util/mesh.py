from util.mesh_util import *

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
    
    def update_vs(self, vs):
        self.vs = vs