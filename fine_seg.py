import torch

from scene.gaussian_model import GaussianModel
from scene import Scene
from gaussian_renderer import render

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams

from argparse import ArgumentParser, Namespace
from PIL import Image
import numpy as np
import os
import torch.nn.functional as F
import torchvision.transforms.functional as func
from seg_utils import conv2d_matrix, update, compute_ratios

from plyfile import PlyData, PlyElement
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_combined_args(parser : ArgumentParser):
    # cmdlne_string = sys.argv[1:]
    # cfgfile_string = "Namespace()"
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args()
    
    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)


    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

def porject_to_2d(viewpoint_camera, points3D):
    full_matrix = viewpoint_camera.full_proj_transform  # w2c @ K 
    # project to image plane
    if points3D.shape[-1] != 4:
        points3D = F.pad(input=points3D, pad=(0, 1), mode='constant', value=1)
    p_hom = (points3D @ full_matrix).transpose(0, 1)  # N, 4 -> 4, N   -1 ~ 1
    p_w = 1.0 / (p_hom[-1, :] + 0.0000001)
    p_proj = p_hom[:3, :] * p_w

    h = viewpoint_camera.image_height
    w = viewpoint_camera.image_width

    point_image = 0.5 * ((p_proj[:2] + 1) * torch.tensor([w, h]).unsqueeze(-1).to(p_proj.device) - 1) # image plane
    point_image = point_image.detach().clone()
    point_image = torch.round(point_image.transpose(0, 1))

    return point_image

def mask_inverse(xyz, viewpoint_camera, sam_mask):
    w2c_matrix = viewpoint_camera.world_view_transform
    # project to camera space
    xyz = F.pad(input=xyz, pad=(0, 1), mode='constant', value=1)
    p_view = (xyz @ w2c_matrix[:, :3]).transpose(0, 1)  # N, 3 -> 3, N
    depth = p_view[-1, :].detach().clone()
    valid_depth = depth >= 0

    h = viewpoint_camera.image_height
    w = viewpoint_camera.image_width
    

    if sam_mask.shape[0] != h or sam_mask.shape[1] != w:
        sam_mask = func.resize(sam_mask.unsqueeze(0), (h, w), antialias=True).squeeze(0).long()
    else:
        sam_mask = sam_mask.long()

    point_image = porject_to_2d(viewpoint_camera, xyz)
    point_image = point_image.long()

    # 判断x,y是否在图像范围之内
    valid_x = (point_image[:, 0] >= 0) & (point_image[:, 0] < w)
    valid_y = (point_image[:, 1] >= 0) & (point_image[:, 1] < h)
    valid_mask = valid_x & valid_y & valid_depth
    point_mask = torch.full((point_image.shape[0],), -1).to("cuda")

    point_mask[valid_mask] = sam_mask[point_image[valid_mask, 1], point_image[valid_mask, 0]]
    indices_mask = torch.where(point_mask == 1)[0]

    return point_mask, indices_mask

def ensemble(multiview_masks, threshold=0.7):
    # threshold = 0.7
    multiview_masks = torch.cat(multiview_masks, dim=1) 
    # vote_labels,_ = torch.mode(multiview_masks, dim=1) # at least 0.5
    # now set 1 as object mask instead of mode
    vote_labels = torch.ones(multiview_masks.shape[0]).to("cuda")
    # # select points with score > threshold 
    matches = torch.eq(multiview_masks, vote_labels.unsqueeze(1))
    # matches = torch.eq(multiview_masks, vote_labels)
    ratios = torch.sum(matches, dim=1) / multiview_masks.shape[1]
    
    ratios_mask = ratios > threshold
    # print("vote labels:",vote_labels[vote_labels == 1])
    # labels_mask = (vote_labels == 1) & ratios_mask
    labels_mask = ratios_mask
    # print("ratios:",labels_mask[labels_mask == True])
    indices_mask = torch.where(labels_mask)[0].detach().cpu()
    # inverse_indices_mask = torch.where(labels_mask!=1)[0].detach().cpu()
    return vote_labels, indices_mask

def save_gs(pc, indices_mask, save_path):
    
    xyz = pc._xyz.detach().cpu()[indices_mask].numpy()
    normals = np.zeros_like(xyz)
    f_dc = pc._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu()[indices_mask].numpy()
    f_rest = pc._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu()[indices_mask].numpy()
    opacities = pc._opacity.detach().cpu()[indices_mask].numpy()
    scale = pc._scaling.detach().cpu()[indices_mask].numpy()
    rotation = pc._rotation.detach().cpu()[indices_mask].numpy()
    obj_dc = pc._objects_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu()[indices_mask].numpy()

    dtype_full = [(attribute, 'f4') for attribute in pc.construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation,obj_dc), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(save_path)

def gaussian_decomp(gaussians, viewpoint_camera, input_mask, indices_mask):
    xyz = gaussians.get_xyz
    point_image = porject_to_2d(viewpoint_camera, xyz)

    conv2d = conv2d_matrix(gaussians, viewpoint_camera, indices_mask, device="cuda")
    height = viewpoint_camera.image_height
    width = viewpoint_camera.image_width
    index_in_all, ratios, dir_vector = compute_ratios(conv2d, point_image, indices_mask, input_mask, height, width)

    # print("size before decompose and index size:",gaussians.get_xyz.shape[0],index_in_all.shape[0])
    decomp_gaussians = update(gaussians, viewpoint_camera, index_in_all, ratios, dir_vector)
    # print("size after decompose:",decomp_gaussians.get_scaling.shape[0])

    return decomp_gaussians

parser = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)

parser.add_argument("-t","--gd_interval", default=10, type=int, help='interval of performing gaussian decomposition')
parser.add_argument("--single",action='store_true' ,help='whether decomposition for each component')

args = get_combined_args(parser)
print(args)
dataset = model.extract(args)
gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, load_iteration="_seg", shuffle=False)
# scene = Scene(dataset, gaussians, load_iteration="/iteration_30000", shuffle=False) # test seg with SAGA 

cameras = scene.getTrainCameras()


mask_path = os.path.join(args.source_path,"inpaint_object_mask_255")

file_extensions = set()
dirs = [] # save dir name
for filename in os.listdir(mask_path):
    if(os.path.isdir(os.path.join(mask_path,filename))):
        dirs.append(filename)
        continue # in case it is a subdirectory
    file_extension = os.path.splitext(filename)[1].lower()
    file_extensions.add(file_extension)

if (len(file_extensions) != 1):
    print("warninng: not just one type of extension")

ext = file_extension # acquire last one

xyz = gaussians.get_xyz

result_mask = torch.empty(0, dtype=torch.long)

if(args.single):
    dirs=[""]
for dir in dirs: # segment each part separately. or segment all of them as whole
    sam_masks = []
    multiview_masks = []
    for i, view in enumerate(cameras):
        image_name = view.image_name+ext
        img_path = os.path.join(mask_path,dir, image_name)
        img = Image.open(img_path)
        mask_array= np.asarray(img,dtype=np.uint8)
        mask_array = np.where(mask_array>127,255,0).astype(np.uint8)

        if len(mask_array.shape) != 2:
            mask_array = torch.from_numpy(mask_array).squeeze(-1).to("cuda")
        else:
            mask_array = torch.from_numpy(mask_array).to("cuda")

        mask_array = (mask_array/255).long()
        sam_masks.append(mask_array)

        point_mask, indices_mask = mask_inverse(xyz, view, mask_array)

        multiview_masks.append(point_mask.unsqueeze(-1))

    _, final_mask = ensemble(multiview_masks,threshold=0.45) # adjust threshold here. set to 0.7 can fail when too many camera views not set to objects


    save_path = os.path.join(scene.model_path, 'point_cloud/iteration_30000/point_cloud_seg_clear{}.ply'.format(dir))
    save_gs(gaussians, final_mask, save_path)    

    for i, view in enumerate(cameras):
        if args.gd_interval != -1 and i % args.gd_interval == 0:
            input_mask = sam_masks[i]
            gaussians = gaussian_decomp(gaussians, view, input_mask, final_mask.to('cuda'))

    save_gd_path = os.path.join(scene.model_path, 'point_cloud/iteration_30000/point_cloud_seg_gd{}.ply'.format(dir))
    save_gs(gaussians, final_mask, save_gd_path)
    result_mask = torch.cat((result_mask, final_mask), dim=0)

# # save left points
# result_mask = torch.unique(result_mask)
# length = xyz.shape[0]
# inverse_indices = torch.arange(length)
# inverse_mask = ~torch.isin(inverse_indices, result_mask)
# inverse_indices = inverse_indices[inverse_mask]

# save_others_path = os.path.join(scene.model_path, 'point_cloud/iteration_30000/point_cloud_seg_others.ply')
# save_gs(gaussians, inverse_indices, save_others_path)

#####show result 
# other_gaussians = GaussianModel(dataset.sh_degree)
# other_gaussians.load_ply(save_others_path)
# bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
# background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

# other_save_path = os.path.join(scene.model_path,'train/ours_others')
# if not os.path.exists(other_save_path):
#     os.mkdir(other_save_path)

# for idx in range(len(cameras)):
#     image_name = cameras[idx].image_name
#     view = cameras[idx]

#     render_pkg = render(view, other_gaussians, pipeline, background)
#     # get sam output mask
#     render_image = render_pkg["render"].permute(1, 2, 0).detach().cpu().numpy()
#     render_image = (255 * np.clip(render_image, 0, 1)).astype(np.uint8)
#     render_image = cv2.cvtColor(render_image, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(os.path.join(other_save_path, '{}.jpg'.format(image_name)), render_image)