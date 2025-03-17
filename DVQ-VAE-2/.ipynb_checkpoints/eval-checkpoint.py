from network.softNet import softNet
import torch
import trimesh
import os
from utils import utils_loss
softnet = softNet()
checkpoint = torch.load("/home/zhaozhe/Pycode/MM24/soft/lightning/lightning_logs/version_9/checkpoints/epoch=436-step=48070 copy.ckpt", map_location=torch.device('cpu'))['state_dict']
softnet.load_state_dict(checkpoint)
softnet.eval()
softnet = softnet.to('cuda')
from scipy.linalg import orthogonal_procrustes
import numpy as np
def align_w_scale(mtx1, mtx2, return_trafo=False):
    """ Align the predicted entity in some optimality sense with the ground truth. """
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

    # apply trafos to the second matrix
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1
    if return_trafo:
        #return R, s, s1, t1 - t2
        return R, s, s1, t1 , t2 , s2
    else:
        return mtx2_t


file = '0058.ply'
object_full = trimesh.load(os.path.join("/data/zz/HMDO/sequence01/object_mesh/", '0000.ply')) 
object_vertices_org = object_full.vertices#, face_id = trimesh.sample.sample_surface(object_full,3000)
object_full = object_full.vertices
hand_vertices , face_id =  trimesh.sample.sample_surface(trimesh.load(os.path.join("/data/zz/HMDO/sequence01/hand_mesh/", file)), 1000)
object_vertices_full = trimesh.load(os.path.join("/data/zz/HMDO/sequence01/object_mesh/", file))
object_vertices = object_vertices_full.vertices #, face_id =  trimesh.sample.sample_surface(object_vertices_full, 3000)
object_vertices_full = object_vertices_full.vertices
with open(os.path.join("/data/zz/HMDO/sequence01/hand_annotation/",file[:4]+'.txt'), 'r') as file:
    content = file.read()
numbers = content.split(',')
scale=float(numbers[0])
hand_vertices=hand_vertices/scale
object_vertices=object_vertices/scale
object_vertices_org = object_vertices_org/scale
object_full = object_full/scale
object_vertices_full = object_vertices_full/scale
#object_vertices_org = align_w_scale(object_vertices,object_vertices_org)
R, s, s1, t1 , t2 , s2 = align_w_scale(object_vertices_full,object_full,return_trafo = True)
object_vertices_org = np.dot((object_vertices_org - t2)/s2, R.T) * s* s1 + t1

obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(torch.tensor(object_vertices).unsqueeze(0).float(), torch.tensor(hand_vertices).unsqueeze(0).float())
contact_map = obj_nn_dist_recon
contact_map_bool = (obj_nn_dist_recon<1e-4)
print(contact_map.sum())
print(contact_map.size())
data_dict = {}
data_dict['grid_size'] = torch.tensor([0.01]).float().cuda()
data_dict['offset'] = torch.tensor([torch.tensor(object_vertices_org).size(0)]).cuda()
print(data_dict['offset'])
data_dict['coord'] = torch.tensor(object_vertices_org).float().cuda()
data_dict['feat'] = torch.cat((contact_map_bool.unsqueeze(2),contact_map.unsqueeze(2)),dim=2).squeeze(0).cuda()
object_pred ,movement= softnet(data_dict)

from pytorch3d.loss import chamfer_distance
loss_org ,_= chamfer_distance(torch.tensor(object_vertices_org).unsqueeze(0).float().cuda(),torch.tensor(object_vertices).unsqueeze(0).float().cuda(),point_reduction='sum', batch_reduction='mean')
loss_pred ,_= chamfer_distance(object_pred.unsqueeze(0).cuda(),torch.tensor(object_vertices).unsqueeze(0).float().cuda(),point_reduction='sum', batch_reduction='mean')
print('loss_org:',loss_org)
print('loss_pred:',loss_pred)

#object_vertices = torch.cat((torch.tensor(object_vertices_org),object_pred.squeeze(0).detach()),0)
object_vertices = object_pred.detach().cpu()#.squeeze(0)#torch.tensor(object_vertices)#torch.tensor(object_vertices_org)
import open3d as o3d
pcd = o3d.geometry.PointCloud()
#obj_hand =torch.cat((object_vertices,torch.tensor(hand_vertices)),0)
pcd.points = o3d.utility.Vector3dVector(np.array(object_vertices))
#o3d.visualization.draw_geometries([pcd],
#                                 point_show_normal=False)#show p3d
#pcd.estimate_normals()
# estimate radius for rolling ball
#pcd = pcd.voxel_down_sample(voxel_size=0.0001)
pcd.paint_uniform_color([0, 0, 255])
color = np.array(pcd.colors)
inliner = [i for i in range(0, 3000) if obj_nn_dist_recon[0][i]  <1e-4]
print(inliner)
color[inliner] = [255 ,0 ,0]
pcd.colors=o3d.utility.Vector3dVector(color[:, :3])
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
o3d.visualization.draw_geometries([pcd],
                                point_show_normal=False)#show p3d
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 3* avg_dist   
alpha = 0.03
radii = [0.005, 0.01, 0.02, 0.04]
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
    pcd, alpha)
####dele  ##
# o3d.visualization.draw_geometries([mesh], window_name='Open3D downSample', width=800, height=600, left=50,
####                                  top=50, point_show_normal=True, mesh_show_wireframe=False, mesh_show_back_face=True,)
# create the triangular mesh with the vertices and faces from open3d
tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                          vertex_normals=np.asarray(mesh.vertex_normals))
alpha=0.03
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
    pcd, alpha)
meshes = [mesh]
#o3d.visualization.draw_geometries([mesh],
#                                 point_show_normal=False)   