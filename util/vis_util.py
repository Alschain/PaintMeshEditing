import torch
import numpy as np

def vis_normal_map(normal):
    normal[...,0] = (normal[...,0] - torch.min(normal[...,0])) / (torch.max(normal[...,0]) - torch.min(normal[...,0]))
    normal[...,1] = (normal[...,1] - torch.min(normal[...,1])) / (torch.max(normal[...,1]) - torch.min(normal[...,1]))
    normal[...,2] = (normal[...,2] - torch.min(normal[...,2])) / (torch.max(normal[...,2]) - torch.min(normal[...,2]))

    normal = normal * 255
    return normal