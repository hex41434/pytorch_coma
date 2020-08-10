import torch_geometric.utils as uti
from torch_geometric.io import *
import os

pth_binary = 'rawData/FS/appliedForceMinus_Face3/Solid_th_2mm_force_Minus10000.0.ply'
pth_ascii = 'rawData/ascii_Solid_th_2mm_force_Plus141800.0.ply'
ply_b = read_ply(os.path.join('../',pth_binary))
ply_a = read_ply(os.path.join('../',pth_ascii))
