import os
from torch import optim, nn, utils, Tensor
import lightning as L
from pytorch3d.structures import Meshes
import mano
from network.softNet_ import softNet
from torch.utils.data import DataLoader
import torch
import trimesh
import numpy as np
from metric.simulate import run_simulation
from network.softNet_ import softNet
import igl
from scipy.spatial import KDTree
import open3d as o3d
from utils import utils_loss
import math
softnet = softNet()
checkpoint = torch.load("/root/Pycode/softv0/logs/lightning_logs/version_0/checkpoints/epoch=299-step=7500.ckpt", map_location=torch.device('cpu'))['state_dict']
softnet.load_state_dict(checkpoint)
softnet.eval()
softnet = softnet.to('cuda')



def pad_point_cloud_with_zeros(x, target_num=12):

    n, f = x.shape

    if n == target_num:
        return x, index  # 不需要填充
    if n < target_num:
        # 计算需要填充的点数
        num_to_pad = target_num - n
        
        # 创建全零填充的点特征，形状为 b * num_to_pad * f
        zero_padding = torch.zeros(num_to_pad, f, device=x.device)

        # 将输入点云和零填充拼接起来，形成 b * target_num * f 的输出
        x_padded = torch.cat([x, zero_padding], dim=0)

    return x_padded
def pad_point_cloud_with_zeros_1d(x, target_num=12):
    
    n= x.shape
    f =1
    if n == target_num:
        return x, index  # 不需要填充
    
    if n < target_num:
        # 计算需要填充的点数
        num_to_pad = target_num - n
        
        # 创建全零填充的点特征，形状为 b * num_to_pad * f
        zero_padding = torch.zeros(num_to_pad, f, device=x.device)

        # 将输入点云和零填充拼接起来，形成 b * target_num * f 的输出
        x_padded = torch.cat([x, zero_padding], dim=0)

    return x_padded


def pad_point_cloud_with_zeros_index(x, index, target_num=12):
    n, f = x.shape

    if n == target_num:
        return x, index  # 不需要填充
    
    if n < target_num:
        # 计算需要填充的点数
        num_to_pad = target_num - n
        
        # 创建全零填充的点特征，形状为 b * num_to_pad * f
        zero_padding = torch.zeros(num_to_pad, f, device=x.device)

        # 将输入点云和零填充拼接起来，形成 b * target_num * f 的输出
        x_padded = torch.cat([x, zero_padding], dim=0)

        # 更新 index，填充部分使用一个固定值，比如 -1 表示未定义的聚类中心
        index_padding = torch.full((1,num_to_pad), -1, dtype=index.dtype, device=index.device)
        index_padded = torch.cat([index.unsqueeze(0), index_padding], dim=1)


    return x_padded, index_padded.squeeze(0)


def find_point_distances(meshA, meshB):
    # 将mesh转换为点云和法线
    verticesA = torch.tensor(meshA.vertices, dtype=torch.float32)
    normalsA = torch.tensor(meshA.vertex_normals, dtype=torch.float32)*-1
    verticesB = torch.tensor(meshB.vertices, dtype=torch.float32)
    
    # 创建一个空的三角形光线拦截器
    ray_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(meshB)
    
    # 初始化距离张量
    distances = torch.full((verticesA.shape[0],), float(0), dtype=torch.float32)
    
    # 遍历meshA中的每个点
    for i, (point, normal) in enumerate(zip(verticesA, normalsA)):
        # 创建从该点出发的射线，方向为法线方向
        ray_origins = point.unsqueeze(0).numpy()
        ray_directions = normal.unsqueeze(0).numpy()
        
        # 计算射线与meshB的交点
        locations, index_ray, index_tri = ray_intersector.intersects_location(ray_origins, ray_directions)
        
        # 如果有交点，计算距离
        if len(locations) > 0:
            # 取第一个交点（假设只有一个交点）
            intersection_point = torch.tensor(locations[0], dtype=torch.float32)
            distance = torch.norm(intersection_point - point)
            
            # 更新距离
            distances[i] = distance
    
    return distances


def intersect_vox_soft(obj_mesh, hand_mesh):
    '''
    Evaluating intersection between hand and object
    :param pitch: voxel size
    :return: intersection volume
    '''
    hand_vertices = hand_mesh.vertices

    mesh_obj = Meshes(verts= torch.tensor( obj_mesh.vertices).unsqueeze(0), faces=torch.tensor(obj_mesh.faces).unsqueeze(0))  
    obj_normal = mesh_obj.verts_normals_packed().view(1,-1, 3)

    hand_nn_dist_recon, hand_nn_idx_recon = utils_loss.get_NN( torch.tensor(hand_vertices).unsqueeze(0).float(),torch.tensor(obj_mesh.vertices).unsqueeze(0).float())
    contact_map_bool_hand = (hand_nn_dist_recon<1e-4).float()
    interior_hand = utils_loss.get_interior(obj_normal, torch.tensor(obj_mesh.vertices).unsqueeze(0).float(), torch.tensor(hand_vertices).unsqueeze(0).float(), hand_nn_idx_recon).type(torch.bool)
    contact_map_bool_hand[interior_hand] =  (contact_map_bool_hand[interior_hand]*-1).float()
    interior_f = torch.zeros(torch.tensor(obj_mesh.vertices).size(0)) 

    interior_f[hand_nn_idx_recon.squeeze(0)] = hand_nn_dist_recon.squeeze(1)

    mesh_ = Meshes(verts= torch.tensor(hand_vertices).unsqueeze(0), faces=torch.tensor(hand_mesh.faces).unsqueeze(0))  
    hand_normal = mesh_.verts_normals_packed().view(-1, 779, 3)
    hand_normal_prior = hand_normal[0][(contact_map_bool_hand==-1).cpu().detach().bool().squeeze(0)].unsqueeze(0)
    hand_vertices_prior = torch.tensor(hand_vertices[(contact_map_bool_hand==-1).cpu().detach().bool().squeeze(0)]).float().unsqueeze(0)
    obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(torch.tensor(obj_mesh.vertices).unsqueeze(0).float(), hand_vertices_prior)
    if hand_normal_prior.size(1)>0:
        interior = utils_loss.get_interior(hand_normal_prior, hand_vertices_prior, torch.tensor(obj_mesh.vertices).unsqueeze(0).float(), obj_nn_idx_recon).type(torch.bool)
        NN_src_xyz = batched_index_select(hand_vertices_prior, obj_nn_idx_recon)  # [B, 3000, 3]
        NN_vector = NN_src_xyz - torch.tensor(obj_mesh.vertices).unsqueeze(0).float()  # [B, 3000, 3]
        # get surface normal of NN src xyz for every trg xyz, should be a [B, 3000, 3] vector
        NN_src_normal = batched_index_select(hand_normal_prior, obj_nn_idx_recon)
        #interior = interior。float
        interior_d = (NN_vector * NN_src_normal).sum(dim=-1)
        #interior = interior。float


        
        obj_nn_dist_recon[interior]= obj_nn_dist_recon[interior]*-1
        contact_map_bool = ((obj_nn_dist_recon<1e-4)&(obj_nn_dist_recon > -2e-4)).float()
        contact_map = obj_nn_dist_recon
        contact_map_bool[interior] = (contact_map_bool[interior]*-1).float()
        contact_map = contact_map *abs(contact_map_bool)
        interior_d = interior_d * contact_map_bool

        distance = find_point_distances(obj_mesh,hand_mesh)
        distance = distance * (contact_map_bool==-1).float().squeeze(0)
        return distance.sum()
    else:
        return 0



def intersect_vox(obj_mesh, hand_mesh, pitch=0.5):
    '''
    Evaluating intersection between hand and object
    :param pitch: voxel size
    :return: intersection volume
    '''
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume

def mesh_vert_int_exts(obj1_mesh, obj2_verts):
    inside = obj1_mesh.ray.contains_points(obj2_verts)
    sign = (inside.astype(int) * 2) - 1
    return sign
def uniform_box_sampling(min_corner, max_corner, res = 0.005):
    x_min = min_corner[0] - res
    x_max = max_corner[0] + res
    y_min = min_corner[1] - res
    y_max = max_corner[1] + res
    z_min = min_corner[2] - res
    z_max = max_corner[2] + res

    h = int((x_max-x_min)/res)+1
    l = int((y_max-y_min)/res)+1
    w = int((z_max-z_min)/res)+1

    # print('Sampling size: %d x %d x %d'%(h, l, w))

    with torch.no_grad():
        xyz = x = torch.zeros(h, l, w, 3, dtype=torch.float32) + torch.tensor([x_min, y_min, z_min], dtype=torch.float32)
        for i in range(1,h):
            xyz[i,0,0] = xyz[i-1,0,0] + torch.tensor([res,0,0])
        for i in range(1,l):
            xyz[:,i,0] = xyz[:,i-1,0] + torch.tensor([0,res,0])
        for i in range(1,w):
            xyz[:,:,i] = xyz[:,:,i-1] + torch.tensor([0,0,res])
    return res, xyz



def bounding_box_intersection(min_corner0, max_corner0, min_corner1, max_corner1):
    min_x = max(min_corner0[0], min_corner1[0])
    min_y = max(min_corner0[1], min_corner1[1])
    min_z = max(min_corner0[2], min_corner1[2])

    max_x = min(max_corner0[0], max_corner1[0])
    max_y = min(max_corner0[1], max_corner1[1])
    max_z = min(max_corner0[2], max_corner1[2])

    if max_x > min_x and max_y > min_y and max_z > min_z:
        # print('Intersected bounding box size: %f x %f x %f'%(max_x - min_x, max_y - min_y, max_z - min_z))
        return np.array([min_x, min_y, min_z]), np.array([max_x, max_y, max_z])
    else:
        return np.zeros((1,3), dtype = np.float32), np.zeros((1,3), dtype = np.float32)

def writeOff(output, vertex, face):
    with open(output, 'w') as f:
        f.write("COFF\n")
        f.write("%d %d 0\n" %(vertex.shape[0], face.shape[0]))
        for row in range(0, vertex.shape[0]):
            f.write("%f %f %f\n" %(vertex[row, 0], vertex[row, 1], vertex[row, 2]))
        for row in range(0, face.shape[0]):
            f.write("3 %d %d %d\n" %(face[row, 0], face[row, 1], face[row, 2]))

def intersection_eval(mesh0, mesh1, res=0.005, scale=1., trans=None, visualize_flag=False, visualize_file='output.off'):
    '''Calculate intersection depth and volumn of the two inputs meshes.
    args:
        mesh1, mesh2 (Trimesh.trimesh): input meshes
        res (float): voxel resolustion in meter(m)
        scale (float): scaling factor
        trans (float) (1, 3): translation
    returns:
        volume (float): intersection volume in cm^3
        mesh_mesh_dist (float): maximum depth from the center of voxel to the surface of another mesh
    '''
    # mesh0 = trimesh.load(mesh_file_0, process=False)
    # mesh1 = trimesh.load(mesh_file_1, process=False)

    # scale = 1 # 10
    # res = 0.5
    mesh0.vertices = mesh0.vertices * scale
    mesh1.vertices = mesh1.vertices * scale

    S, I, C = igl.signed_distance(mesh0.vertices + 1e-10, mesh1.vertices, mesh1.faces, return_normals=False)

    mesh_mesh_distance = S.min()
    # print("dist", S)
    # print("Mesh to mesh distance: %f cm" % mesh_mesh_distance)

    #### print("Mesh to mesh distance: %f" % (max(S.min(), 0)))

    if mesh_mesh_distance > 0:
        # print('No intersection!')
        return 0, mesh_mesh_distance

    # Get bounding box for each mesh:
    min_corner0 = np.array([mesh0.vertices[:,0].min(), mesh0.vertices[:,1].min(), mesh0.vertices[:,2].min()])
    max_corner0 = np.array([mesh0.vertices[:,0].max(), mesh0.vertices[:,1].max(), mesh0.vertices[:,2].max()])

    min_corner1 = np.array([mesh1.vertices[:,0].min(), mesh1.vertices[:,1].min(), mesh1.vertices[:,2].min()])
    max_corner1 = np.array([mesh1.vertices[:,0].max(), mesh1.vertices[:,1].max(), mesh1.vertices[:,2].max()])

    # Compute the intersection of two bounding boxes:
    min_corner_i, max_corner_i = bounding_box_intersection(min_corner0, max_corner0, min_corner1, max_corner1)
    if ((min_corner_i - max_corner_i)**2).sum() == 0:
        # print('No intersection!')
        return 0, mesh_mesh_distance

    # Uniformly sample the intersection bounding box:
    _, xyz = uniform_box_sampling(min_corner_i, max_corner_i, res)
    xyz = xyz.view(-1, 3)
    xyz = xyz.detach().cpu().numpy()

    S, I, C = igl.signed_distance(xyz, mesh0.vertices, mesh0.faces, return_normals=False)

    inside_sample_index = np.argwhere(S < 0.0)
    # print("inside sample index", inside_sample_index, len(inside_sample_index))

    # Compute the signed distance for inside_samples to mesh 1:
    inside_samples = xyz[inside_sample_index[:,0], :]

    S, I, C = igl.signed_distance(inside_samples, mesh1.vertices, mesh1.faces, return_normals=False)

    inside_both_sample_index = np.argwhere(S < 0)

    # Compute intersection volume:
    i_v = inside_both_sample_index.shape[0] * (res**3)
    # print("Intersected volume: %f cm^3" % (i_v))

    # Visualize intersection volume:
    if visualize_flag:
        writeOff(visualize_file, inside_samples[inside_both_sample_index[:,0], :], np.zeros((0,3)))

    # From (m) to (cm)
    return i_v * 1e6, mesh_mesh_distance * 1e2

def seal(mesh_to_seal):
    '''
    Seal MANO hand wrist to make it wathertight.
    An average of wrist vertices is added along with its faces to other wrist vertices.
    '''
    circle_v_id = np.array([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120], dtype = np.int32)
    center = (mesh_to_seal.vertices[circle_v_id, :]).mean(0)

    mesh_to_seal.vertices = np.vstack([mesh_to_seal.vertices, center])
    center_v_id = mesh_to_seal.vertices.shape[0] - 1

    # pylint: disable=unsubscriptable-object # pylint/issues/3139
    for i in range(circle_v_id.shape[0]):
        new_faces = [circle_v_id[i-1], circle_v_id[i], center_v_id] 
        mesh_to_seal.faces = np.vstack([mesh_to_seal.faces, new_faces])
    return mesh_to_seal

def batched_index_select(input, index, dim=1):
    '''
    :param input: [B, N1, *]
    :param dim: the dim to be selected
    :param index: [B, N2]
    :return: [B, N2, *] selected result
    '''
    views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim=dim, index=index)

def get_faces_xyz(faces_idx, xyz):
    '''
    :param faces_idx: [B, N1, 3]. N1 is number of faces (1538 for MANO), index of face vertices in N2
    :param xyz: [B, N2, 3]. N2 is number of points.
    :return: faces_xyz: [B, N1, 3, 3] faces vertices coordinate
    '''
    B, N1, D = faces_idx.size()
    N2 = xyz.size(1)
    xyz_replicated = xyz.cpu().unsqueeze(1).repeat(1,N1,1,1)  # use cpu to save CUDA memory
    faces_idx_replicated = faces_idx.unsqueeze(-1).repeat(1,1,1,D).type(torch.LongTensor)
    return torch.gather(xyz_replicated, dim=2, index=faces_idx_replicated).to(faces_idx.device)

def batch_mesh_contains_points(
    ray_origins, # point cloud as origin of rays
    obj_triangles,
    direction=torch.Tensor([0.4395064455, 0.617598629942, 0.652231566745]),
):
    """Times efficient but memory greedy !
    Computes ALL ray/triangle intersections and then counts them to determine
    if point inside mesh
    Args:
    ray_origins: (batch_size x point_nb x 3)
    obj_triangles: (batch_size, triangle_nb, vertex_nb=3, vertex_coords=3)
    tol_thresh: To determine if ray and triangle are //
    Returns:
    exterior: (batch_size, point_nb) 1 if the point is outside mesh, 0 else
    """
    tol_thresh = 0.0000001
    batch_size = obj_triangles.shape[0]
    triangle_nb = obj_triangles.shape[1]
    point_nb = ray_origins.shape[1]

    # Batch dim and triangle dim will flattened together
    batch_points_size = batch_size * triangle_nb
    # Direction is random but shared
    v0, v1, v2 = obj_triangles[:, :, 0], obj_triangles[:, :, 1], obj_triangles[:, :, 2]
    # Get edges
    v0v1 = v1 - v0
    v0v2 = v2 - v0

    direction = direction.to(ray_origins.device)
    # Expand needed vectors
    batch_direction = direction.view(1, 1, 3).expand(batch_size, triangle_nb, 3)

    # Compute ray/triangle intersections
    pvec = torch.cross(batch_direction, v0v2, dim=2)
    dets = torch.bmm(
        v0v1.view(batch_points_size, 1, 3), pvec.view(batch_points_size, 3, 1)
    ).view(batch_size, triangle_nb)

    # Check if ray and triangle are parallel
    parallel = abs(dets) < tol_thresh
    invdet = 1 / (dets + 0.1 * tol_thresh)

    # Repeat mesh info as many times as there are rays
    triangle_nb = v0.shape[1]
    v0 = v0.repeat(1, point_nb, 1)
    v0v1 = v0v1.repeat(1, point_nb, 1)
    v0v2 = v0v2.repeat(1, point_nb, 1)
    hand_verts_repeated = (
        ray_origins.view(batch_size, point_nb, 1, 3)
        .repeat(1, 1, triangle_nb, 1)
        .view(ray_origins.shape[0], triangle_nb * point_nb, 3)
    )
    pvec = pvec.repeat(1, point_nb, 1)
    invdet = invdet.repeat(1, point_nb)
    tvec = hand_verts_repeated - v0
    u_val = (
        torch.bmm(
            tvec.view(batch_size * tvec.shape[1], 1, 3),
            pvec.view(batch_size * tvec.shape[1], 3, 1),
        ).view(batch_size, tvec.shape[1])
        * invdet
    )
    # Check ray intersects inside triangle
    u_correct = (u_val > 0) * (u_val < 1)
    qvec = torch.cross(tvec, v0v1, dim=2)

    batch_direction = batch_direction.repeat(1, point_nb, 1)
    v_val = (
        torch.bmm(
            batch_direction.view(batch_size * qvec.shape[1], 1, 3),
            qvec.view(batch_size * qvec.shape[1], 3, 1),
        ).view(batch_size, qvec.shape[1])
        * invdet
    )
    v_correct = (v_val > 0) * (u_val + v_val < 1)
    t = (
        torch.bmm(
            v0v2.view(batch_size * qvec.shape[1], 1, 3),
            qvec.view(batch_size * qvec.shape[1], 3, 1),
        ).view(batch_size, qvec.shape[1])
        * invdet
    )
    # Check triangle is in front of ray_origin along ray direction
    t_pos = t >= tol_thresh
    parallel = parallel.repeat(1, point_nb)
    # # Check that all intersection conditions are met
    try:
        not_parallel = 1 - parallel
    except:
        not_parallel = parallel==False
    final_inter = v_correct * u_correct * not_parallel * t_pos
    # Reshape batch point/vertices intersection matrix
    # final_intersections[batch_idx, point_idx, triangle_idx] == 1 means ray
    # intersects triangle
    final_intersections = final_inter.view(batch_size, point_nb, triangle_nb)
    # Check if intersection number accross mesh is odd to determine if point is
    # outside of mesh
    exterior = final_intersections.sum(2) % 2 == 0
    return exterior
def get_NN(src_xyz, trg_xyz, k=1):
    '''
    :param src_xyz: [B, N1, 3]
    :param trg_xyz: [B, N2, 3]
    :return: nn_dists, nn_dix: all [B, 3000] tensor for NN distance and index in N2
    '''
    B = src_xyz.size(0)
    src_lengths = torch.full(
        (src_xyz.shape[0],), src_xyz.shape[1], dtype=torch.int64, device=src_xyz.device
    )  # [B], N for each num
    trg_lengths = torch.full(
        (trg_xyz.shape[0],), trg_xyz.shape[1], dtype=torch.int64, device=trg_xyz.device
    )
    src_nn = knn_points(src_xyz, trg_xyz, lengths1=src_lengths, lengths2=trg_lengths, K=k)  # [dists, idx]
    nn_dists = src_nn.dists[..., 0]
    nn_idx = src_nn.idx[..., 0]
 
    return nn_dists, nn_idx


total_penetr_vol=0.0
total_simu_disp=0.0

total_penetr_vol_deform=0.0
total_simu_disp_deform=0.0

total_penetr_vol_TTA=0.0
total_simu_disp_TTA=0.0

total_penetr_vol_deform_TTA=0.0
total_simu_disp_deform_TTA=0.0

total_num = 0
total_time = -3.0
total_contact =0
total_high=0
vol_list = []
mesh_dist_list = []

for i in range(12):
        for j in range(10):
                if i  == 7 :
                      continue
                #try:
                if True:
                    org_mesh = trimesh.load_mesh("/root/Pycode/vis_deform_300/obj_{}_{}.ply".format(i,j))
                    hand_mesh= trimesh.load_mesh("/root/Pycode/vis_deform_300/hand_org_{}_{}.ply".format(i,j))
                    obj_mesh = trimesh.load_mesh("/root/Pycode/vis_deform_300/obj_{}_{}.ply".format(i,j))
                    point_num = obj_mesh.vertices.shape[0]  
                    obj_mean = obj_mesh.vertices.mean(axis=0)
                    obj_mesh.vertices = obj_mesh.vertices - obj_mean
                    hand_mesh.vertices = hand_mesh.vertices - obj_mean
                    org_mesh.vertices = org_mesh.vertices - obj_mean
                    obj_mean = obj_mesh.vertices.mean(axis=0)



                    with torch.no_grad():
                            rh_mano = mano.load(model_path='./models/mano/MANO_RIGHT.pkl',
                                                    model_type='mano',
                                                    use_pca=False,
                                                    num_pca_comps=51,
                                                    batch_size=1,
                                                    flat_hand_mean=True)
                    rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3).contiguous() # [1, 1538, 3], face triangle indexes  

                    hand_vertices = hand_mesh.vertices

                    hand_vertices = torch.tensor(hand_vertices).unsqueeze(0).float()

                    object_org_scaled = torch.tensor(obj_mesh.vertices).unsqueeze(0).float()
                    object_vertices_full = obj_mesh
                    object_vertices = torch.tensor(obj_mesh.vertices).unsqueeze(0).float()

                    mesh_obj = Meshes(verts= torch.tensor( obj_mesh.vertices).unsqueeze(0), faces=torch.tensor(obj_mesh.faces).unsqueeze(0))  
                    obj_normal = mesh_obj.verts_normals_packed().view(1,-1, 3)

                    hand_nn_dist_recon, hand_nn_idx_recon = utils_loss.get_NN( hand_vertices,torch.tensor(obj_mesh.vertices).unsqueeze(0).float())
                    contact_map_bool_hand = (hand_nn_dist_recon<3e-4).float()
                    interior_hand = utils_loss.get_interior(obj_normal, torch.tensor(obj_mesh.vertices).unsqueeze(0).float(),hand_vertices, hand_nn_idx_recon).type(torch.bool)
                    contact_map_bool_hand[interior_hand] =  (contact_map_bool_hand[interior_hand]*-1).float()
                    interior_f = torch.zeros(torch.tensor(obj_mesh.vertices).size(0)) 

                    interior_f[hand_nn_idx_recon.squeeze(0)] = hand_nn_dist_recon.squeeze(1)

                                        
                    hand_vertices_prior = torch.tensor(hand_vertices[0][(contact_map_bool_hand==-1).cpu().detach().bool().squeeze(0)]).float().unsqueeze(0)
                    # d =  torch.norm(torch.tensor(movement_gt), dim=1, keepdim=True)
                    # movement_gt = movement_gt*(movement_gt>3e-3)

                    obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(object_org_scaled, hand_vertices_prior)
                    mesh = Meshes(verts=hand_vertices, faces=rh_faces)
                    hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)
                    hand_normal_prior = hand_normal[0][(contact_map_bool_hand==-1).cpu().detach().bool().squeeze(0)].unsqueeze(0)
                    # if not nn_dist:
                    #     nn_dist, nn_idx = utils_loss.get_NN(obj_xyz, hand_xyz)

                    interior = utils_loss.get_interior(hand_normal_prior, hand_vertices_prior, object_org_scaled, obj_nn_idx_recon).type(torch.bool)  # True for interior

                    NN_src_xyz = batched_index_select(hand_vertices_prior, obj_nn_idx_recon)  # [B, 3000, 3]
                    NN_vector = NN_src_xyz - object_org_scaled  # [B, 3000, 3]
                    # get surface normal of NN src xyz for every trg xyz, should be a [B, 3000, 3] vector
                    NN_src_normal = batched_index_select(hand_normal_prior, obj_nn_idx_recon)
                    #interior = interior。float
                    interior_d = (NN_vector * NN_src_normal).sum(dim=-1)
                    #interior_dist=nn_dist[interior]
                    obj_nn_dist_recon[interior]= obj_nn_dist_recon[interior]*-1



                    contact_map_bool = ((obj_nn_dist_recon<1e-4)&(obj_nn_dist_recon > -2e-4)).float()

                    interior_d = interior_d * contact_map_bool
                    contact_map_bool[interior] = contact_map_bool[interior]*-1
                    contact_map = obj_nn_dist_recon

                    obj_mesh_dis = trimesh.Trimesh(vertices = object_org_scaled[0].detach(),faces = object_vertices_full.faces)
                    hand_mesh_dis = trimesh.Trimesh(vertices = hand_vertices[0].detach(),faces = rh_faces[0].detach())
                    distance = find_point_distances(obj_mesh_dis,hand_mesh_dis)
                    distance = distance * (contact_map_bool==-1).float().squeeze(0)

                    mask_num = object_vertices.size(1)
                    contact_map_bool_base = pad_point_cloud_with_zeros(contact_map_bool.squeeze(0).unsqueeze(1), target_num=12000).squeeze(1)
                    distance_base = pad_point_cloud_with_zeros(distance.unsqueeze(1), 12000).squeeze(1)
                    object_vertices = pad_point_cloud_with_zeros(object_vertices.squeeze(0), 12000).unsqueeze(0)
                    object_vertices_org = pad_point_cloud_with_zeros(object_org_scaled.squeeze(0), 12000).unsqueeze(0)
                    normal = torch.tensor(obj_mesh_dis.vertex_normals).unsqueeze(0)
                    normal = pad_point_cloud_with_zeros(normal.squeeze(0), 12000).unsqueeze(0)
                    
                    movemant_gt_base = object_vertices - object_vertices_org





                    ##############################################################################################################################################################################  L1
                    vertices = object_org_scaled[0].detach()
                    faces = object_vertices_full.faces
                    mesh_in = o3d.geometry.TriangleMesh()
                    mesh_in.vertices = o3d.utility.Vector3dVector(vertices)
                    mesh_in.triangles = o3d.utility.Vector3iVector(faces)
                    #mesh_in.compute_vertex_normals()
                    original_vertices = np.asarray(mesh_in.vertices)

                    voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 16
                    mesh_smp = mesh_in.simplify_vertex_clustering(
                        voxel_size=voxel_size,
                        contraction=o3d.geometry.SimplificationContraction.Average)
                    simplified_vertices = torch.tensor( np.asarray(mesh_smp.vertices)).unsqueeze(0).float()
                    tree = KDTree( np.asarray(mesh_smp.vertices))
                    smp_distances, indices = tree.query(original_vertices)


                    mesh_obj = Meshes(verts= simplified_vertices, faces=torch.tensor(np.asarray(mesh_smp.triangles)).unsqueeze(0))  
                    obj_normal = mesh_obj.verts_normals_packed().view(1,-1, 3)
                    hand_nn_dist_recon, hand_nn_idx_recon = utils_loss.get_NN( hand_vertices,simplified_vertices)
                    contact_map_bool_hand = (hand_nn_dist_recon<3e-4).float()
                    interior_hand = utils_loss.get_interior(obj_normal, simplified_vertices,hand_vertices, hand_nn_idx_recon).type(torch.bool)
                    contact_map_bool_hand[interior_hand] =  (contact_map_bool_hand[interior_hand]*-1).float()
                    interior_f = torch.zeros(torch.tensor(simplified_vertices).size(1)) 
                    interior_f[hand_nn_idx_recon.squeeze(0)] = hand_nn_dist_recon.squeeze(1).squeeze(0) 


                    hand_vertices_prior = torch.tensor(hand_vertices[0][(contact_map_bool_hand==-1).cpu().detach().bool().squeeze(0)]).float().unsqueeze(0)
                    # d =  torch.norm(torch.tensor(movement_gt), dim=1, keepdim=True)
                    # movement_gt = movement_gt*(movement_gt>3e-3)

                    obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(simplified_vertices, hand_vertices_prior)
                    mesh = Meshes(verts=hand_vertices, faces=rh_faces)
                    hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)
                    hand_normal_prior = hand_normal[0][(contact_map_bool_hand==-1).cpu().detach().bool().squeeze(0)].unsqueeze(0)
                    # if not nn_dist:
                    #     nn_dist, nn_idx = utils_loss.get_NN(obj_xyz, hand_xyz)
                    if hand_normal_prior.size(1)>0:
                        interior = utils_loss.get_interior(hand_normal_prior, hand_vertices_prior, simplified_vertices, obj_nn_idx_recon).type(torch.bool)  # True for interior

                        NN_src_xyz = batched_index_select(hand_vertices_prior, obj_nn_idx_recon)  # [B, 3000, 3]
                        NN_vector = NN_src_xyz - simplified_vertices  # [B, 3000, 3]
                        # get surface normal of NN src xyz for every trg xyz, should be a [B, 3000, 3] vector
                        NN_src_normal = batched_index_select(hand_normal_prior, obj_nn_idx_recon)
                        #interior = interior。float
                        interior_d = (NN_vector * NN_src_normal).sum(dim=-1)
                        #interior_dist=nn_dist[interior]
                        obj_nn_dist_recon[interior]= obj_nn_dist_recon[interior]*-1



                        contact_map_bool = ((obj_nn_dist_recon<1e-4)&(obj_nn_dist_recon > -2e-4)).float()

                        interior_d = interior_d * contact_map_bool
                        contact_map_bool[interior] = contact_map_bool[interior]*-1
                        contact_map = obj_nn_dist_recon

                        obj_mesh_dis = trimesh.Trimesh(vertices = simplified_vertices[0].detach(),faces = np.asarray(mesh_smp.triangles))
                        hand_mesh_dis = trimesh.Trimesh(vertices = hand_vertices[0].detach(),faces = rh_faces[0].detach())
                        distance = find_point_distances(obj_mesh_dis,hand_mesh_dis)

                        print('contact_map_bool',contact_map_bool.size())
                        print('distance',distance.size())
                        print('simplified_vertices',simplified_vertices.size())

                        if distance.size(0)!=contact_map_bool.size(1):
                            continue
                        distance = distance * (contact_map_bool==-1).float().squeeze(0)

                        mask_num_l1 = contact_map_bool.size(1)
                        contact_map_bool_l1 = pad_point_cloud_with_zeros(contact_map_bool.squeeze(0).unsqueeze(1), target_num=1200).squeeze(1)
                        distance_l1,indices_l1 = pad_point_cloud_with_zeros_index(distance.unsqueeze(1),torch.tensor(indices), target_num=1200)
                        distance_l1 = distance_l1.squeeze(1)
                        #object_vertices_l1 = pad_point_cloud_with_zeros(object_vertices.squeeze(0), target_num=1200).unsqueeze(0)
                        object_vertices_org_l1 = pad_point_cloud_with_zeros(simplified_vertices.squeeze(0), target_num=1200).unsqueeze(0)
                        normal_l1 = torch.tensor(obj_mesh_dis.vertex_normals)
                        normal_l1 = pad_point_cloud_with_zeros(normal_l1, target_num=1200).unsqueeze(0)
                        
                        #smp_distances_l1 = pad_point_cloud_with_zeros(torch.tensor(smp_distances).unsqueeze(0), target_num=12000)
                        smp_distances_l1,indices_l1 = pad_point_cloud_with_zeros_index(torch.tensor(smp_distances).unsqueeze(1),torch.tensor(indices), target_num=12000)



                        ##############################################################################################################################################################################  L2
                        vertices = object_org_scaled[0].detach()
                        faces = object_vertices_full.faces
                        mesh_in = mesh_smp
                        #mesh_in.compute_vertex_normals()
                        original_vertices = np.asarray(mesh_in.vertices)

                        voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 8
                        mesh_smp = mesh_in.simplify_vertex_clustering(
                            voxel_size=voxel_size,
                            contraction=o3d.geometry.SimplificationContraction.Average)
                        simplified_vertices = torch.tensor( np.asarray(mesh_smp.vertices)).unsqueeze(0).float()
                        tree = KDTree( np.asarray(mesh_smp.vertices))
                        smp_distances, indices = tree.query(original_vertices)


                        mesh_obj = Meshes(verts= simplified_vertices, faces=torch.tensor(np.asarray(mesh_smp.triangles)).unsqueeze(0))  
                        obj_normal = mesh_obj.verts_normals_packed().view(1,-1, 3)
                        hand_nn_dist_recon, hand_nn_idx_recon = utils_loss.get_NN( hand_vertices,simplified_vertices)
                        contact_map_bool_hand = (hand_nn_dist_recon<3e-4).float()
                        interior_hand = utils_loss.get_interior(obj_normal, simplified_vertices,hand_vertices, hand_nn_idx_recon).type(torch.bool)
                        contact_map_bool_hand[interior_hand] =  (contact_map_bool_hand[interior_hand]*-1).float()
                        interior_f = torch.zeros(torch.tensor(simplified_vertices).size(1)) 
                        interior_f[hand_nn_idx_recon.squeeze(0)] = hand_nn_dist_recon.squeeze(1).squeeze(0) 


                        hand_vertices_prior = torch.tensor(hand_vertices[0][(contact_map_bool_hand==-1).cpu().detach().bool().squeeze(0)]).float().unsqueeze(0)
                        # d =  torch.norm(torch.tensor(movement_gt), dim=1, keepdim=True)
                        # movement_gt = movement_gt*(movement_gt>3e-3)

                        obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(simplified_vertices, hand_vertices_prior)
                        mesh = Meshes(verts=hand_vertices, faces=rh_faces)
                        hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)
                        hand_normal_prior = hand_normal[0][(contact_map_bool_hand==-1).cpu().detach().bool().squeeze(0)].unsqueeze(0)
                        # if not nn_dist:
                        #     nn_dist, nn_idx = utils_loss.get_NN(obj_xyz, hand_xyz)

                        interior = utils_loss.get_interior(hand_normal_prior, hand_vertices_prior, simplified_vertices, obj_nn_idx_recon).type(torch.bool)  # True for interior

                        NN_src_xyz = batched_index_select(hand_vertices_prior, obj_nn_idx_recon)  # [B, 3000, 3]
                        NN_vector = NN_src_xyz - simplified_vertices  # [B, 3000, 3]
                        # get surface normal of NN src xyz for every trg xyz, should be a [B, 3000, 3] vector
                        NN_src_normal = batched_index_select(hand_normal_prior, obj_nn_idx_recon)
                        #interior = interior。float
                        interior_d = (NN_vector * NN_src_normal).sum(dim=-1)
                        #interior_dist=nn_dist[interior]
                        obj_nn_dist_recon[interior]= obj_nn_dist_recon[interior]*-1



                        contact_map_bool = ((obj_nn_dist_recon<1e-4)&(obj_nn_dist_recon > -2e-4)).float()

                        interior_d = interior_d * contact_map_bool
                        contact_map_bool[interior] = contact_map_bool[interior]*-1
                        contact_map = obj_nn_dist_recon

                        obj_mesh_dis = trimesh.Trimesh(vertices = simplified_vertices[0].detach(),faces = np.asarray(mesh_smp.triangles))
                        hand_mesh_dis = trimesh.Trimesh(vertices = hand_vertices[0].detach(),faces = rh_faces[0].detach())
                        distance = find_point_distances(obj_mesh_dis,hand_mesh_dis)


                        if distance.size(0)!=contact_map_bool.size(1):
                            continue
                        distance = distance * (contact_map_bool==-1).float().squeeze(0)
                        print(contact_map_bool.size())
                        mask_num_l2 = contact_map_bool.size(1)
                        contact_map_bool_l2 = pad_point_cloud_with_zeros(contact_map_bool.squeeze(0).unsqueeze(1), target_num=300).squeeze(1)
                        distance_l2 = pad_point_cloud_with_zeros(distance.unsqueeze(1), target_num=300)
                        distance_l2 = distance_l2.squeeze(1)
                        #object_vertices_l1 = pad_point_cloud_with_zeros(object_vertices.squeeze(0), target_num=1200).unsqueeze(0)
                        object_vertices_org_l2 = pad_point_cloud_with_zeros(simplified_vertices.squeeze(0), target_num=300).unsqueeze(0)
                        normal_l2 = torch.tensor(obj_mesh_dis.vertex_normals)
                        normal_l2 = pad_point_cloud_with_zeros(normal_l2, target_num=300).unsqueeze(0)
                        
                        #smp_distances_l2 = pad_point_cloud_with_zeros(torch.tensor(smp_distances).unsqueeze(0), target_num=1000)
                        smp_distances_l2,indices_l2 = pad_point_cloud_with_zeros_index(torch.tensor(smp_distances).unsqueeze(1),torch.tensor(indices), target_num=1200)

                        object_org_scaled =  pad_point_cloud_with_zeros(object_org_scaled.squeeze(0),12000).unsqueeze(0)
                        print('add')


                        data = {
                                    'contact_map_bool': contact_map_bool_base.unsqueeze(0).cuda(),
                                    'object_vertices': object_vertices.float().cuda(),
                                    'object_vertices_org': object_org_scaled.float().cuda(),
                                    #'offset': offset_batch.squeeze(0).squeeze(0),
                                    'movement_gt':movemant_gt_base.float().cuda(),
                                    'distance':torch.tensor(distance_base).unsqueeze(0).float().detach().cuda(),
                                    'normal':normal.float().cuda(),
                                    'mask_num':[mask_num],
                                    'contact_map_bool_l1': contact_map_bool_l1.unsqueeze(0).cuda(),
                                    #'object_vertices_l1': object_vertices_batch_l1.squeeze(0).float(),
                                    'object_vertices_org_l1': object_vertices_org_l1.float().cuda(),
                                    'distance_l1':distance_l1.float().detach().unsqueeze(0).cuda(),
                                    'normal_l1':normal_l1.float().cuda(),
                                    'mask_num_l1':[mask_num_l1],
                                    'index_l1':indices_l1.unsqueeze(0).cuda(),
                                    'smp_distances_l1':smp_distances_l1.float().cuda(),
                                    'contact_map_bool_l2': contact_map_bool_l2.unsqueeze(0).cuda(),
                                    #'object_vertices_l2': object_vertices_batch_l2.squeeze(0).float(),
                                    'object_vertices_org_l2': object_vertices_org_l2.float().cuda(),
                                    'distance_l2':distance_l2.float().detach().unsqueeze(0).cuda(),
                                    'normal_l2':normal_l2.float().cuda(),
                                    'mask_num_l2':[mask_num_l2],
                                    'index_l2':indices_l2.unsqueeze(0).cuda(),
                                    'smp_distances_l2':smp_distances_l2.float().cuda(),

                            }


                        object_pred ,movement= softnet(data)
                        from pytorch3d.ops import taubin_smoothing
                        from pytorch3d.structures import Meshes
                        verts=[object_pred.squeeze(0)[:point_num,:].to('cuda')]
                        #faces=[face[0]]
                        face = [torch.tensor(obj_mesh.faces).to('cuda')]
                        pytorch3d_mesh = Meshes(verts,face)
                        #pytorch3d_mesh = taubin_smoothing(meshes=pytorch3d_mesh, lambd=0.53, mu= -0.53, num_iter= 5)
                        obj_mesh.vertices = pytorch3d_mesh.verts_list()[0].cpu().detach().numpy()
                        mask = abs(contact_map_bool_base[:point_num]).cpu().detach().bool().squeeze(0)
                        #print(mask.size())
                        colors = np.ones((len(obj_mesh.vertices), 4)) * [1, 1, 1, 1]  # RGBA 白色
                        colors[abs(contact_map_bool_base[:point_num]).cpu().detach().bool().squeeze(0)] = [0, 0, 1, 1]  
                        colors[(contact_map_bool_base[:point_num]==-1).cpu().detach().bool().squeeze(0)] = [1, 0, 0, 1]  # RGBA 红色
                        obj_mesh.visual.vertex_colors = colors
                        #trimesh.Scene([obj_mesh, hand_mesh]).show()

                        obj_mesh.export("./vis_deform/"+'obj_{}_{}'.format(i,j)+".ply")
                        hand_mesh.export("./vis_deform/"+'hand_{}_{}'.format(i,j)+".ply")

                        hand = seal(hand_mesh)
                        # object=org_mesh
                        # trimesh.repair.fix_normals(object)
                        # object = trimesh.convex.convex_hull(object)
                        # vol, mesh_dist = intersection_eval(hand, object, res=0.001, visualize_flag=True)

                        # penetr_vol=vol
                        penetr_vol = intersect_vox_soft(org_mesh, hand)*1e6
                        if  penetr_vol < 1e-8:
                                sample_contact = False
                        else:
                                sample_contact = True
                        # simulation displacement
                        vhacd_exe = "/root/Pycode/v-hacd-master/TestVHACD"
                        try:
                            simu_disp = run_simulation(hand_mesh.vertices, rh_faces.reshape((-1, 3)),
                                                        org_mesh.vertices, org_mesh.faces.reshape((-1, 3)),
                                                        vhacd_exe=vhacd_exe)
                                #print('run success')
                        except:
                                simu_disp = 0.10
                        penetr_vol_org = penetr_vol
                        simu_disp_org = simu_disp



                        # object=obj_mesh
                        # trimesh.repair.fix_normals(object)
                        # object = trimesh.convex.convex_hull(object)
                        # vol, mesh_dist = intersection_eval(hand, object, res=0.001, visualize_flag=True)
                        
                        # penetr_vol=vol
                        penetr_vol = intersect_vox_soft(obj_mesh, hand)*1e6
                        if  penetr_vol < 1e-8:
                                sample_contact = False
                        else:
                                sample_contact = True
                        # simulation displacement
                        vhacd_exe = "/root/Pycode/v-hacd-master/TestVHACD"
                        try:
                                simu_disp = run_simulation(hand_mesh.vertices, rh_faces.reshape((-1, 3)),
                                                        obj_mesh.vertices, obj_mesh.faces.reshape((-1, 3)),
                                                        vhacd_exe=vhacd_exe)
                                #print('run success')
                        except:
                                simu_disp = 0.10
                        penetr_vol_deform = penetr_vol
                        simu_disp_deform = simu_disp

                        



                    hand_mesh_TTA= trimesh.load_mesh("/root/Pycode/vis_deform_300/hand_{}_{}.ply".format(i,j))
                    obj_mesh_TTA = trimesh.load_mesh("/root/Pycode/vis_deform_300/obj_{}_{}.ply".format(i,j))
                    org_mesh_TTA = trimesh.load_mesh("/root/Pycode/vis_deform_300/obj_{}_{}.ply".format(i,j))

                    obj_mean_TTA = obj_mesh_TTA.vertices.mean(axis=0)
                    obj_mesh_TTA.vertices = obj_mesh_TTA.vertices - obj_mean_TTA
                    hand_mesh_TTA.vertices = hand_mesh_TTA.vertices - obj_mean_TTA
                    org_mesh_TTA.vertices = org_mesh_TTA.vertices - obj_mean_TTA
                    hand_mesh = hand_mesh_TTA
                    obj_mesh = obj_mesh_TTA
                    org_mesh = org_mesh_TTA
                    obj_mean = obj_mean_TTA
                    
                    with torch.no_grad():
                            rh_mano = mano.load(model_path='./models/mano/MANO_RIGHT.pkl',
                                                    model_type='mano',
                                                    use_pca=False,
                                                    num_pca_comps=51,
                                                    batch_size=1,
                                                    flat_hand_mean=True)
                    rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3).contiguous() # [1, 1538, 3], face triangle indexes  

                    hand_vertices = hand_mesh.vertices

                    hand_vertices = torch.tensor(hand_vertices).unsqueeze(0).float()

                    object_org_scaled = torch.tensor(obj_mesh.vertices).unsqueeze(0).float()
                    object_vertices_full = obj_mesh
                    object_vertices = torch.tensor(obj_mesh.vertices).unsqueeze(0).float()

                    mesh_obj = Meshes(verts= torch.tensor( obj_mesh.vertices).unsqueeze(0), faces=torch.tensor(obj_mesh.faces).unsqueeze(0))  
                    obj_normal = mesh_obj.verts_normals_packed().view(1,-1, 3)

                    hand_nn_dist_recon, hand_nn_idx_recon = utils_loss.get_NN( hand_vertices,torch.tensor(obj_mesh.vertices).unsqueeze(0).float())
                    contact_map_bool_hand = (hand_nn_dist_recon<3e-4).float()
                    interior_hand = utils_loss.get_interior(obj_normal, torch.tensor(obj_mesh.vertices).unsqueeze(0).float(),hand_vertices, hand_nn_idx_recon).type(torch.bool)
                    contact_map_bool_hand[interior_hand] =  (contact_map_bool_hand[interior_hand]*-1).float()
                    interior_f = torch.zeros(torch.tensor(obj_mesh.vertices).size(0)) 

                    interior_f[hand_nn_idx_recon.squeeze(0)] = hand_nn_dist_recon.squeeze(1)

                                        
                    hand_vertices_prior = torch.tensor(hand_vertices[0][(contact_map_bool_hand==-1).cpu().detach().bool().squeeze(0)]).float().unsqueeze(0)
                    # d =  torch.norm(torch.tensor(movement_gt), dim=1, keepdim=True)
                    # movement_gt = movement_gt*(movement_gt>3e-3)

                    obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(object_org_scaled, hand_vertices_prior)
                    mesh = Meshes(verts=hand_vertices, faces=rh_faces)
                    hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)
                    hand_normal_prior = hand_normal[0][(contact_map_bool_hand==-1).cpu().detach().bool().squeeze(0)].unsqueeze(0)
                    # if not nn_dist:
                    #     nn_dist, nn_idx = utils_loss.get_NN(obj_xyz, hand_xyz)
                    print(hand_vertices_prior.size())
                    interior = utils_loss.get_interior(hand_normal_prior, hand_vertices_prior, object_org_scaled, obj_nn_idx_recon).type(torch.bool)  # True for interior

                    NN_src_xyz = batched_index_select(hand_vertices_prior, obj_nn_idx_recon)  # [B, 3000, 3]
                    NN_vector = NN_src_xyz - object_org_scaled  # [B, 3000, 3]
                    # get surface normal of NN src xyz for every trg xyz, should be a [B, 3000, 3] vector
                    NN_src_normal = batched_index_select(hand_normal_prior, obj_nn_idx_recon)
                    #interior = interior。float
                    interior_d = (NN_vector * NN_src_normal).sum(dim=-1)
                    #interior_dist=nn_dist[interior]
                    obj_nn_dist_recon[interior]= obj_nn_dist_recon[interior]*-1



                    contact_map_bool = ((obj_nn_dist_recon<1e-4)&(obj_nn_dist_recon > -2e-4)).float()

                    interior_d = interior_d * contact_map_bool
                    contact_map_bool[interior] = contact_map_bool[interior]*-1
                    contact_map = obj_nn_dist_recon

                    obj_mesh_dis = trimesh.Trimesh(vertices = object_org_scaled[0].detach(),faces = object_vertices_full.faces)
                    hand_mesh_dis = trimesh.Trimesh(vertices = hand_vertices[0].detach(),faces = rh_faces[0].detach())
                    distance = find_point_distances(obj_mesh_dis,hand_mesh_dis)
                    distance = distance * (contact_map_bool==-1).float().squeeze(0)

                    mask_num = object_vertices.size(1)
                    contact_map_bool_base = pad_point_cloud_with_zeros(contact_map_bool.squeeze(0).unsqueeze(1), target_num=12000).squeeze(1)
                    distance_base = pad_point_cloud_with_zeros(distance.unsqueeze(1), 12000).squeeze(1)
                    object_vertices = pad_point_cloud_with_zeros(object_vertices.squeeze(0), 12000).unsqueeze(0)
                    object_vertices_org = pad_point_cloud_with_zeros(object_org_scaled.squeeze(0), 12000).unsqueeze(0)
                    normal = torch.tensor(obj_mesh_dis.vertex_normals).unsqueeze(0)
                    normal = pad_point_cloud_with_zeros(normal.squeeze(0), 12000).unsqueeze(0)
                    
                    movemant_gt_base = object_vertices - object_vertices_org






                    ##############################################################################################################################################################################  L1
                    vertices = object_org_scaled[0].detach()
                    faces = object_vertices_full.faces
                    mesh_in = o3d.geometry.TriangleMesh()
                    mesh_in.vertices = o3d.utility.Vector3dVector(vertices)
                    mesh_in.triangles = o3d.utility.Vector3iVector(faces)
                    #mesh_in.compute_vertex_normals()
                    original_vertices = np.asarray(mesh_in.vertices)

                    voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 16
                    mesh_smp = mesh_in.simplify_vertex_clustering(
                        voxel_size=voxel_size,
                        contraction=o3d.geometry.SimplificationContraction.Average)
                    simplified_vertices = torch.tensor( np.asarray(mesh_smp.vertices)).unsqueeze(0).float()
                    tree = KDTree( np.asarray(mesh_smp.vertices))
                    smp_distances, indices = tree.query(original_vertices)


                    mesh_obj = Meshes(verts= simplified_vertices, faces=torch.tensor(np.asarray(mesh_smp.triangles)).unsqueeze(0))  
                    obj_normal = mesh_obj.verts_normals_packed().view(1,-1, 3)
                    hand_nn_dist_recon, hand_nn_idx_recon = utils_loss.get_NN( hand_vertices,simplified_vertices)
                    contact_map_bool_hand = (hand_nn_dist_recon<3e-4).float()
                    interior_hand = utils_loss.get_interior(obj_normal, simplified_vertices,hand_vertices, hand_nn_idx_recon).type(torch.bool)
                    contact_map_bool_hand[interior_hand] =  (contact_map_bool_hand[interior_hand]*-1).float()
                    interior_f = torch.zeros(torch.tensor(simplified_vertices).size(1)) 
                    interior_f[hand_nn_idx_recon.squeeze(0)] = hand_nn_dist_recon.squeeze(1).squeeze(0) 


                    hand_vertices_prior = torch.tensor(hand_vertices[0][(contact_map_bool_hand==-1).cpu().detach().bool().squeeze(0)]).float().unsqueeze(0)
                    # d =  torch.norm(torch.tensor(movement_gt), dim=1, keepdim=True)
                    # movement_gt = movement_gt*(movement_gt>3e-3)

                    obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(simplified_vertices, hand_vertices_prior)
                    mesh = Meshes(verts=hand_vertices, faces=rh_faces)
                    hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)
                    hand_normal_prior = hand_normal[0][(contact_map_bool_hand==-1).cpu().detach().bool().squeeze(0)].unsqueeze(0)
                    # if not nn_dist:
                    #     nn_dist, nn_idx = utils_loss.get_NN(obj_xyz, hand_xyz)
                    if hand_normal_prior.size(1)>0:
                        interior = utils_loss.get_interior(hand_normal_prior, hand_vertices_prior, simplified_vertices, obj_nn_idx_recon).type(torch.bool)  # True for interior

                        NN_src_xyz = batched_index_select(hand_vertices_prior, obj_nn_idx_recon)  # [B, 3000, 3]
                        NN_vector = NN_src_xyz - simplified_vertices  # [B, 3000, 3]
                        # get surface normal of NN src xyz for every trg xyz, should be a [B, 3000, 3] vector
                        NN_src_normal = batched_index_select(hand_normal_prior, obj_nn_idx_recon)
                        #interior = interior。float
                        interior_d = (NN_vector * NN_src_normal).sum(dim=-1)
                        #interior_dist=nn_dist[interior]
                        obj_nn_dist_recon[interior]= obj_nn_dist_recon[interior]*-1



                        contact_map_bool = ((obj_nn_dist_recon<1e-4)&(obj_nn_dist_recon > -2e-4)).float()

                        interior_d = interior_d * contact_map_bool
                        contact_map_bool[interior] = contact_map_bool[interior]*-1
                        contact_map = obj_nn_dist_recon

                        obj_mesh_dis = trimesh.Trimesh(vertices = simplified_vertices[0].detach(),faces = np.asarray(mesh_smp.triangles))
                        hand_mesh_dis = trimesh.Trimesh(vertices = hand_vertices[0].detach(),faces = rh_faces[0].detach())
                        distance = find_point_distances(obj_mesh_dis,hand_mesh_dis)

                        print('contact_map_bool',contact_map_bool.size())
                        print('distance',distance.size())
                        print('simplified_vertices',simplified_vertices.size())

                        if distance.size(0)!=contact_map_bool.size(1):
                            continue
                        distance = distance * (contact_map_bool==-1).float().squeeze(0)

                        mask_num_l1 = contact_map_bool.size(1)
                        contact_map_bool_l1 = pad_point_cloud_with_zeros(contact_map_bool.squeeze(0).unsqueeze(1), target_num=1200).squeeze(1)
                        distance_l1,indices_l1 = pad_point_cloud_with_zeros_index(distance.unsqueeze(1),torch.tensor(indices), target_num=1200)
                        distance_l1 = distance_l1.squeeze(1)
                        #object_vertices_l1 = pad_point_cloud_with_zeros(object_vertices.squeeze(0), target_num=1200).unsqueeze(0)
                        object_vertices_org_l1 = pad_point_cloud_with_zeros(simplified_vertices.squeeze(0), target_num=1200).unsqueeze(0)
                        normal_l1 = torch.tensor(obj_mesh_dis.vertex_normals)
                        normal_l1 = pad_point_cloud_with_zeros(normal_l1, target_num=1200).unsqueeze(0)
                        
                        #smp_distances_l1 = pad_point_cloud_with_zeros(torch.tensor(smp_distances).unsqueeze(0), target_num=12000)
                        smp_distances_l1,indices_l1 = pad_point_cloud_with_zeros_index(torch.tensor(smp_distances).unsqueeze(1),torch.tensor(indices), target_num=12000)



                        ##############################################################################################################################################################################  L2
                        vertices = object_org_scaled[0].detach()
                        faces = object_vertices_full.faces
                        mesh_in = mesh_smp
                        #mesh_in.compute_vertex_normals()
                        original_vertices = np.asarray(mesh_in.vertices)

                        voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 8
                        mesh_smp = mesh_in.simplify_vertex_clustering(
                            voxel_size=voxel_size,
                            contraction=o3d.geometry.SimplificationContraction.Average)
                        simplified_vertices = torch.tensor( np.asarray(mesh_smp.vertices)).unsqueeze(0).float()
                        tree = KDTree( np.asarray(mesh_smp.vertices))
                        smp_distances, indices = tree.query(original_vertices)


                        mesh_obj = Meshes(verts= simplified_vertices, faces=torch.tensor(np.asarray(mesh_smp.triangles)).unsqueeze(0))  
                        obj_normal = mesh_obj.verts_normals_packed().view(1,-1, 3)
                        hand_nn_dist_recon, hand_nn_idx_recon = utils_loss.get_NN( hand_vertices,simplified_vertices)
                        contact_map_bool_hand = (hand_nn_dist_recon<3e-4).float()
                        interior_hand = utils_loss.get_interior(obj_normal, simplified_vertices,hand_vertices, hand_nn_idx_recon).type(torch.bool)
                        contact_map_bool_hand[interior_hand] =  (contact_map_bool_hand[interior_hand]*-1).float()
                        interior_f = torch.zeros(torch.tensor(simplified_vertices).size(1)) 
                        interior_f[hand_nn_idx_recon.squeeze(0)] = hand_nn_dist_recon.squeeze(1).squeeze(0) 


                        hand_vertices_prior = torch.tensor(hand_vertices[0][(contact_map_bool_hand==-1).cpu().detach().bool().squeeze(0)]).float().unsqueeze(0)
                        # d =  torch.norm(torch.tensor(movement_gt), dim=1, keepdim=True)
                        # movement_gt = movement_gt*(movement_gt>3e-3)

                        obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(simplified_vertices, hand_vertices_prior)
                        mesh = Meshes(verts=hand_vertices, faces=rh_faces)
                        hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)
                        hand_normal_prior = hand_normal[0][(contact_map_bool_hand==-1).cpu().detach().bool().squeeze(0)].unsqueeze(0)
                        # if not nn_dist:
                        #     nn_dist, nn_idx = utils_loss.get_NN(obj_xyz, hand_xyz)

                        interior = utils_loss.get_interior(hand_normal_prior, hand_vertices_prior, simplified_vertices, obj_nn_idx_recon).type(torch.bool)  # True for interior

                        NN_src_xyz = batched_index_select(hand_vertices_prior, obj_nn_idx_recon)  # [B, 3000, 3]
                        NN_vector = NN_src_xyz - simplified_vertices  # [B, 3000, 3]
                        # get surface normal of NN src xyz for every trg xyz, should be a [B, 3000, 3] vector
                        NN_src_normal = batched_index_select(hand_normal_prior, obj_nn_idx_recon)
                        #interior = interior。float
                        interior_d = (NN_vector * NN_src_normal).sum(dim=-1)
                        #interior_dist=nn_dist[interior]
                        obj_nn_dist_recon[interior]= obj_nn_dist_recon[interior]*-1



                        contact_map_bool = ((obj_nn_dist_recon<1e-4)&(obj_nn_dist_recon > -2e-4)).float()

                        interior_d = interior_d * contact_map_bool
                        contact_map_bool[interior] = contact_map_bool[interior]*-1
                        contact_map = obj_nn_dist_recon

                        obj_mesh_dis = trimesh.Trimesh(vertices = simplified_vertices[0].detach(),faces = np.asarray(mesh_smp.triangles))
                        hand_mesh_dis = trimesh.Trimesh(vertices = hand_vertices[0].detach(),faces = rh_faces[0].detach())
                        distance = find_point_distances(obj_mesh_dis,hand_mesh_dis)


                        if distance.size(0)!=contact_map_bool.size(1):
                            continue
                        distance = distance * (contact_map_bool==-1).float().squeeze(0)

                        mask_num_l2 = contact_map_bool.size(1)
                        contact_map_bool_l2 = pad_point_cloud_with_zeros(contact_map_bool.squeeze(0).unsqueeze(1), target_num=300).squeeze(1)
                        distance_l2 = pad_point_cloud_with_zeros(distance.unsqueeze(1), target_num=300)
                        distance_l2 = distance_l2.squeeze(1)
                        #object_vertices_l1 = pad_point_cloud_with_zeros(object_vertices.squeeze(0), target_num=1200).unsqueeze(0)
                        object_vertices_org_l2 = pad_point_cloud_with_zeros(simplified_vertices.squeeze(0), target_num=300).unsqueeze(0)
                        normal_l2 = torch.tensor(obj_mesh_dis.vertex_normals)
                        normal_l2 = pad_point_cloud_with_zeros(normal_l2, target_num=300).unsqueeze(0)
                        
                        #smp_distances_l2 = pad_point_cloud_with_zeros(torch.tensor(smp_distances).unsqueeze(0), target_num=1000)
                        smp_distances_l2,indices_l2 = pad_point_cloud_with_zeros_index(torch.tensor(smp_distances).unsqueeze(1),torch.tensor(indices), target_num=1200)

                        object_org_scaled =  pad_point_cloud_with_zeros(object_org_scaled.squeeze(0),12000).unsqueeze(0)
                        print('add')


                        data = {
                                    'contact_map_bool': contact_map_bool_base.unsqueeze(0).cuda(),
                                    'object_vertices': object_vertices.float().cuda(),
                                    'object_vertices_org': object_org_scaled.float().cuda(),
                                    #'offset': offset_batch.squeeze(0).squeeze(0),
                                    'movement_gt':movemant_gt_base.float().cuda(),
                                    'distance':torch.tensor(distance_base).unsqueeze(0).float().detach().cuda(),
                                    'normal':normal.float().cuda(),
                                    'mask_num':[mask_num],
                                    'contact_map_bool_l1': contact_map_bool_l1.unsqueeze(0).cuda(),
                                    #'object_vertices_l1': object_vertices_batch_l1.squeeze(0).float(),
                                    'object_vertices_org_l1': object_vertices_org_l1.float().cuda(),
                                    'distance_l1':distance_l1.float().detach().unsqueeze(0).cuda(),
                                    'normal_l1':normal_l1.float().cuda(),
                                    'mask_num_l1':[mask_num_l1],
                                    'index_l1':indices_l1.unsqueeze(0).cuda(),
                                    'smp_distances_l1':smp_distances_l1.float().cuda(),
                                    'contact_map_bool_l2': contact_map_bool_l2.unsqueeze(0).cuda(),
                                    #'object_vertices_l2': object_vertices_batch_l2.squeeze(0).float(),
                                    'object_vertices_org_l2': object_vertices_org_l2.float().cuda(),
                                    'distance_l2':distance_l2.float().detach().unsqueeze(0).cuda(),
                                    'normal_l2':normal_l2.float().cuda(),
                                    'mask_num_l2':[mask_num_l2],
                                    'index_l2':indices_l2.unsqueeze(0).cuda(),
                                    'smp_distances_l2':smp_distances_l2.float().cuda(),

                            }
                        object_pred ,movement= softnet(data)

                        from pytorch3d.ops import taubin_smoothing
                        from pytorch3d.structures import Meshes
                        verts=[object_pred.squeeze(0)[:point_num,:].to('cuda')]
                        #faces=[face[0]]
                        face = [torch.tensor(obj_mesh_TTA.faces).to('cuda')]
                        pytorch3d_mesh = Meshes(verts,face)
                        #pytorch3d_mesh = taubin_smoothing(meshes=pytorch3d_mesh, lambd=0.53, mu= -0.53, num_iter= 5)
                        obj_mesh_TTA.vertices = pytorch3d_mesh.verts_list()[0].cpu().detach().numpy()
                        mask = abs(contact_map_bool_base[:point_num]).cpu().detach().bool().squeeze(0)
                        #print(mask.size())
                        colors = np.ones((len(obj_mesh_TTA.vertices), 4)) * [1, 1, 1, 1]  # RGBA 白色
                        colors[abs(contact_map_bool_base[:point_num]).cpu().detach().bool().squeeze(0)] = [0, 0, 1, 1]  
                        colors[(contact_map_bool_base[:point_num]==-1).cpu().detach().bool().squeeze(0)] = [1, 0, 0, 1]  # RGBA 红色
                        obj_mesh_TTA.visual.vertex_colors = colors
                        #trimesh.Scene([obj_mesh, hand_mesh]).show()
                        obj_mesh_TTA.export("./vis_deform/"+'obj_deform_{}_{}'.format(i,j)+".ply")
                        hand_mesh_TTA.export("./vis_deform/"+'hand_deform_{}_{}'.format(i,j)+".ply")

                        hand = seal(hand_mesh_TTA)
                        # object=org_mesh
                        # trimesh.repair.fix_normals(object)
                        # object = trimesh.convex.convex_hull(object)
                        # vol, mesh_dist = intersection_eval(hand, object, res=0.001, visualize_flag=True)
                        # vol_list.append(vol)
                        # mesh_dist_list.append(mesh_dist)
                        
                        # penetr_vol=vol
                        penetr_vol = intersect_vox_soft(org_mesh_TTA, hand)
                        if  penetr_vol < 1e-8:
                                sample_contact = False
                        else:
                                sample_contact = True
                        # simulation displacement
                        vhacd_exe = "/root/Pycode/v-hacd-master/TestVHACD"
                        try:
                                simu_disp = run_simulation(hand_mesh_TTA.vertices, rh_faces.reshape((-1, 3)),
                                                        org_mesh_TTA.vertices, org_mesh_TTA.faces.reshape((-1, 3)),
                                                        vhacd_exe=vhacd_exe)
                                #print('run success')
                        except:
                                simu_disp = 0.10
                        penetr_vol_org_TTA = penetr_vol
                        simu_disp_org_TTA = simu_disp



                        # object=obj_mesh
                        # trimesh.repair.fix_normals(object)
                        # object = trimesh.convex.convex_hull(object)
                        # vol, mesh_dist = intersection_eval(hand, object, res=0.001, visualize_flag=True)
                        # vol_list.append(vol)
                        # mesh_dist_list.append(mesh_dist)
                        
                        # penetr_vol=vol
                        penetr_vol = intersect_vox_soft(obj_mesh_TTA, hand)
                        if  penetr_vol < 1e-8:
                                sample_contact = False
                        else:
                                sample_contact = True
                        # simulation displacement
                        vhacd_exe = "/root/Pycode/v-hacd-master/TestVHACD"
                        try:
                                simu_disp = run_simulation(hand_mesh_TTA.vertices, rh_faces.reshape((-1, 3)),
                                                        obj_mesh_TTA.vertices, obj_mesh_TTA.faces.reshape((-1, 3)),
                                                        vhacd_exe=vhacd_exe)
                                #print('run success')
                        except:
                                simu_disp = 0.10
                

                    total_penetr_vol_deform_TTA+=penetr_vol
                    total_simu_disp_deform_TTA+=simu_disp
                    total_penetr_vol_deform+=penetr_vol_deform
                    total_simu_disp_deform+=simu_disp_deform
                    total_penetr_vol+=penetr_vol_org
                    total_simu_disp+=simu_disp_org
                    total_penetr_vol_TTA+=penetr_vol_org_TTA
                    total_simu_disp_TTA+=simu_disp_org_TTA



                    total_num+=1
                    print("#############################################")
                    print(" total_num: ", total_num)
                    print(" vol cm3: ", total_penetr_vol/total_num)
                    print(" simu_disp: ", total_simu_disp/total_num)
                    print(" deform vol cm3: ", total_penetr_vol_deform/total_num)
                    print(" deform simu_disp: ", total_simu_disp_deform/total_num)


                    print(" TTA vol cm3: ", total_penetr_vol_TTA/total_num)
                    print(" TTA simu_disp: ", total_simu_disp_TTA/total_num)
                    print(" TTA deform vol cm3: ", total_penetr_vol_deform_TTA/total_num)
                    print(" TTA deform simu_disp: ", total_simu_disp_deform_TTA/total_num)
                    #print(" deform inter dist cm: ", np.mean(mesh_dist_list))
                # except:
                #     continue
