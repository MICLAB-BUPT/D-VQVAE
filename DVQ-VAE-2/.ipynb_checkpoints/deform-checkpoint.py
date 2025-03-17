import os
from torch import optim, nn, utils, Tensor
import lightning as L
from pytorch3d.structures import Meshes
import mano
from network.softNet import softNet
from torch.utils.data import DataLoader
import torch
import trimesh
import numpy as np
from utils import utils_loss
import math
softnet = softNet()
checkpoint = torch.load("/home/zhaozhe/Pycode/NIPS24/soft/lightningv6/lightning_logs/version_21/checkpoints/epoch=202-step=97237.ckpt", map_location=torch.device('cpu'))['state_dict']
softnet.load_state_dict(checkpoint)
softnet.eval()
softnet = softnet.to('cuda')
base_path = '/data/zz/HMDO'
seq = 'sequence01'
split = [#s1
            ['0000','0066','0162'],
            #s2
            ['0000','0091','0191'],
            #s3
            ['0000','0085','0149','0170'],
            #s4
            ['0000','0090'],
            #s5
            ['0000','0066','0187'],
            #s6
            ['0000','0137'],
            #s7
            ['0000','0101','0182'],
            #s8
            ['0000','0170','0181'],
            #s9
            ['0000','0026','0100','0173'],
            #s10
            ['0000','0068','0174'],
            #s11
            ['0000','0156'],
            #s12
            ['0000','0022','0178'],
            #s13
            ['0000','0060','0084','0111','0154','0184'],
        ]
for seq_i in range(0,len(split[int(seq[-2:])-1])-1):
    start_num = int(split[int(seq[-2:])-1][seq_i])
    end_num = int(split[int(seq[-2:])-1][seq_i+1])
    
    bias = math.floor((end_num - start_num-1)/10)
    print(bias)
    for seq_num in range(1,bias):
        for cam in sorted(os.listdir(os.path.join(base_path, seq,'img'))):
            
            for frame_i in range(0,10):
                frame_id = start_num + seq_num+frame_i*bias
                formatted_str = "{:04d}".format(frame_id)
                hand_mesh_pre= trimesh.load_mesh("/data/zz/HMDO/"+seq+"/hand_mesh_pre/"+cam+"/"+formatted_str+".ply")
                org_mesh = trimesh.load_mesh("/data/zz/HMDO/"+seq+"/object_mesh_pre/"+cam+"/"+formatted_str+".ply")
                hand_mesh_pre.export("./vis/"+str(frame_i)+'hand'+".ply")
                org_mesh.export("./vis/"+str(frame_i)+'obj'+".ply")
                hand_mesh_gt= trimesh.load_mesh("/data/zz/HMDO/"+seq+"/hand_mesh/"+formatted_str+".ply")
                org_mesh_gt = trimesh.load_mesh("/data/zz/HMDO/"+seq+"/object_mesh/"+formatted_str+".ply")
                hand_mesh_gt.export("./vis/"+str(frame_i)+'hand_gt'+".ply")
                org_mesh_gt.export("./vis/"+str(frame_i)+'obj_gt'+".ply")
                with torch.no_grad():
                    rh_mano = mano.load(model_path='./models/mano/MANO_RIGHT.pkl',
                                                model_type='mano',
                                                use_pca=False,
                                                num_pca_comps=51,
                                                batch_size=1,
                                                flat_hand_mean=True)
                rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3).contiguous() # [1, 1538, 3], face triangle indexes  
                prior_idx=[
                        697, 698, 699, 700, 712, 713, 714, 715, 737, 738, 739, 740, 741, 743, 744, 745, 746, 748, 749,
                        750, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768,
                        46, 47, 48, 49, 164, 165, 166, 167, 194, 195, 223, 237, 238, 280, 281, 298, 301, 317, 320, 323, 324, 325, 326,
                        327, 328, 329, 330, 331, 332, 333, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354,
                        355,
                        356, 357, 358, 359, 375, 376, 386, 387, 396, 397, 402, 403, 413, 429, 433, 434, 435, 436, 437, 438,
                        439, 440, 441, 442, 443, 444, 452, 453, 454, 455, 456, 459, 460, 461, 462, 463, 464, 465, 466, 467,
                        468, 469, 470, 471, 484, 485, 486, 496, 497, 506, 507, 513, 514, 524, 545, 546, 547, 548, 549,
                        550, 551, 552, 553, 555, 563, 564, 565, 566, 567, 570, 572, 573, 574, 575, 576, 577, 578,
                        580, 581, 582, 583, 600, 601, 602, 614, 615, 624, 625, 630, 631, 641, 663, 664, 665, 666, 667,
                        668, 670, 672, 680, 681, 682, 683, 684, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695,
                        73, 96, 98, 99, 772, 774, 775, 777
                        ]
                hand_vertices = hand_mesh_pre.vertices
                mesh_ = Meshes(verts= torch.tensor(hand_vertices).unsqueeze(0), faces=rh_faces)  
                hand_normal = mesh_.verts_normals_packed().view(-1, 778, 3)
                hand_normal_prior = hand_normal[0][prior_idx].unsqueeze(0)
                hand_vertices_prior = torch.tensor(hand_vertices[prior_idx]).float().unsqueeze(0)
                obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(torch.tensor(org_mesh.vertices).unsqueeze(0).float(), hand_vertices_prior)
                interior = utils_loss.get_interior(hand_normal_prior, hand_vertices_prior, torch.tensor(org_mesh.vertices).unsqueeze(0).float(), obj_nn_idx_recon).type(torch.bool)
                #interior = interior。float
                contact_map_bool = (obj_nn_dist_recon<1e-4).float()
                obj_nn_dist_recon[interior]= obj_nn_dist_recon[interior]*-1
                contact_map = obj_nn_dist_recon
                contact_map_bool[interior] = (contact_map_bool[interior]*-1).float()
                contact_map = contact_map *abs(contact_map_bool)
                print(contact_map_bool.sum())
                print(contact_map.size())
                data_dict = {}

                mesh_obj = Meshes(verts= torch.tensor(org_mesh.vertices).unsqueeze(0), faces=torch.tensor(org_mesh.faces).unsqueeze(0))
                gt_normal = mesh_obj.verts_normals_packed()#.view(-1, 778, 3)
                data_dict['grid_size'] = torch.tensor([0.001]).float().cuda()
                data_dict['offset'] = torch.tensor([torch.tensor(org_mesh.vertices).size(0)]).cuda()

                data_dict['coord'] = torch.tensor(org_mesh.vertices).float().cuda()
                data_dict['feat'] = torch.cat((contact_map_bool.unsqueeze(2),contact_map.unsqueeze(2),gt_normal.unsqueeze(0)),dim=2).squeeze(0).cuda().float()
                object_pred ,movement,_= softnet(data_dict)
                from pytorch3d.ops import taubin_smoothing
                from pytorch3d.structures import Meshes
                verts=[object_pred.squeeze(0).to('cuda')]
                #faces=[face[0]]
                face = [torch.tensor(org_mesh.faces).to('cuda')]
                pytorch3d_mesh = Meshes(verts,face)
                pytorch3d_mesh = taubin_smoothing(meshes=pytorch3d_mesh, lambd=0.53, mu= -0.53, num_iter= 5)
                org_mesh.vertices = pytorch3d_mesh.verts_list()[0].cpu().detach().numpy()
                org_mesh.export("./vis/"+str(frame_i)+'obj_deform'+".ply")
            break
        break
    break