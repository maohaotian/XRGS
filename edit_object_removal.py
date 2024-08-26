# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import json

import cv2
from sklearn.decomposition import PCA

from scipy.spatial import ConvexHull, Delaunay
from render import feature_to_rgb, visualize_obj, obj2mask_binary
import shutil
from fine_seg import ensemble,mask_inverse

def points_inside_convex_hull(point_cloud, mask, remove_outliers=True, outlier_factor=1.0):
    """
    Given a point cloud and a mask indicating a subset of points, this function computes the convex hull of the 
    subset of points and then identifies all points from the original point cloud that are inside this convex hull.
    
    Parameters:
    - point_cloud (torch.Tensor): A tensor of shape (N, 3) representing the point cloud.
    - mask (torch.Tensor): A tensor of shape (N,) indicating the subset of points to be used for constructing the convex hull.
    - remove_outliers (bool): Whether to remove outliers from the masked points before computing the convex hull. Default is True.
    - outlier_factor (float): The factor used to determine outliers based on the IQR method. Larger values will classify more points as outliers.
    
    Returns:
    - inside_hull_tensor_mask (torch.Tensor): A mask of shape (N,) with values set to True for the points inside the convex hull 
                                              and False otherwise.
    """

    # Extract the masked points from the point cloud
    masked_points = point_cloud[mask].cpu().numpy()

    # Remove outliers if the option is selected
    if remove_outliers:
        Q1 = np.percentile(masked_points, 25, axis=0)
        Q3 = np.percentile(masked_points, 75, axis=0)
        IQR = Q3 - Q1
        outlier_mask = (masked_points < (Q1 - outlier_factor * IQR)) | (masked_points > (Q3 + outlier_factor * IQR))
        filtered_masked_points = masked_points[~np.any(outlier_mask, axis=1)]
    else:
        filtered_masked_points = masked_points

    # Compute the Delaunay triangulation of the filtered masked points
    delaunay = Delaunay(filtered_masked_points)

    # Determine which points from the original point cloud are inside the convex hull
    points_inside_hull_mask = delaunay.find_simplex(point_cloud.cpu().numpy()) >= 0

    # Convert the numpy mask back to a torch tensor and return
    inside_hull_tensor_mask = torch.tensor(points_inside_hull_mask, device='cuda')

    return inside_hull_tensor_mask

def seg_with_mask(mask_dir,cameras,xyz):
    filenames = os.listdir(mask_dir)
    file_extension = os.path.splitext(filenames[0])[1].lower()
    multiview_masks = []
    # control the size according 100
    camera_size = len(cameras)
    skip_iteration = int(camera_size/100)
    for i,view in enumerate(cameras):
        if skip_iteration>1 and i % skip_iteration!= 0:
            continue
        image_name = view.image_name+file_extension
        img_path = os.path.join(mask_dir, image_name)
        img = Image.open(img_path)
        mask_array= np.asarray(img,dtype=np.uint8)
        mask_array = np.where(mask_array>127,255,0).astype(np.uint8)

        if len(mask_array.shape) != 2:
            mask_array = torch.from_numpy(mask_array).squeeze(-1).to("cuda")
        else:
            mask_array = torch.from_numpy(mask_array).to("cuda")

        mask_array = (mask_array/255).long()
        point_mask, indices_mask = mask_inverse(xyz, view, mask_array)

        multiview_masks.append(point_mask.unsqueeze(-1))
        del mask_array

    _, final_mask = ensemble(multiview_masks,threshold=0.5) #0.1
    return final_mask

def double_cut_seg(classifier,gaussians,selected_id,removal_thresh,mask_path,views):
    with torch.no_grad():
        # using fine seg
        logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
        prob_obj3d = torch.softmax(logits3d,dim=0)
        mask = prob_obj3d[selected_id, :, :] > removal_thresh            
        mask3d = mask.any(dim=0).squeeze()

        mask3d_convex = points_inside_convex_hull(gaussians._xyz.detach(),mask3d,outlier_factor=1.0)
        mask3d = torch.logical_or(mask3d,mask3d_convex)
        # print(mask3d,mask3d.shape)
        # print(fine_seg_mask,fine_seg_mask.shape)
        fine_seg_mask = seg_with_mask(mask_path,views,gaussians.get_xyz)
        mask3d[[i for i in range(len(mask3d)) if i not in fine_seg_mask]] = False
        mask3d = mask3d.float()[:,None,None]
        return mask3d

def removal_setup(opt, model_path, iteration, views, gaussians, pipeline, background, classifier, selected_obj_ids, cameras_extent, removal_thresh,source_path,support_faces):
    selected_obj_ids = torch.tensor(selected_obj_ids).cuda()
    support_faces = torch.tensor(support_faces).cuda()
    all_masks=[]
    seg_cloud_path = os.path.join(model_path, "point_cloud_seg") # all segmented objects, using mask to generate later

    for selected_id in support_faces:
        mask_path = os.path.join(source_path,"inpaint_object_mask_255",str(selected_id.item()))
        selected_id = selected_id.unsqueeze(0)
        mask3d = double_cut_seg(classifier,gaussians,selected_id,removal_thresh,mask_path,views)
        gaussians.save_selected_ply(os.path.join(seg_cloud_path, f"point_seg{str(selected_id.item())}.ply"), mask3d) #only need ply for support


    for selected_id in selected_obj_ids:
        mask_path = os.path.join(source_path,"inpaint_object_mask_255",str(selected_id.item()))
        selected_id = selected_id.unsqueeze(0)
        mask3d = double_cut_seg(classifier,gaussians,selected_id,removal_thresh,mask_path,views)
        gaussians.save_selected_ply(os.path.join(seg_cloud_path, f"point_seg{str(selected_id.item())}.ply"), mask3d)
        all_masks.append(mask3d)
            # print("mask3d:",mask3d,mask3d.shape)
    all_masks_tensor = torch.stack(all_masks, dim=0)
    # all_masks = torch.cat(all_masks, dim=0)
    # print("allmask:",all_masks_tensor,all_masks_tensor.shape)
    final_mask3d = all_masks_tensor.any(dim=0).float()
    # print("final mask:",final_mask3d,final_mask3d.shape)
    torch.save(final_mask3d,os.path.join(model_path,"final_mask3d.pth")) #

    point_cloud_path = os.path.join(model_path, "point_cloud_object_removal/iteration_{}".format(iteration))
    # save segmented gaussians
    gaussians.save_selected_ply(os.path.join(seg_cloud_path, "point_cloud.ply"), final_mask3d) # this is actually not necessary
    
    # fix some gaussians
    gaussians.removal_setup(opt,final_mask3d)
    
    
    # save gaussians
    # point_cloud_path = os.path.join(model_path, "point_cloud_object_removal/iteration_{}".format(iteration))
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    return gaussians

def save_mask_binary(model_path, name, iteration, views, selected_ids,  sub_dir = None):
    if sub_dir == None:
        mask_obj_path = os.path.join(model_path, name, "ours{}".format(iteration), "inpaint_object_mask_255")
    else:
        mask_obj_path = os.path.join(model_path, name, "ours{}".format(iteration), "inpaint_object_mask_255", sub_dir)
    for idx, view in enumerate(views):
        gt_objects = view.objects
        makedirs(mask_obj_path, exist_ok=True)
        mask_binary = obj2mask_binary(gt_objects.cpu().numpy().astype(np.uint8),selected_ids)
        mask_binary = np.where(mask_binary > 0, 255, 0).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
        Image.fromarray(mask_binary, mode="L").save(os.path.join(mask_obj_path, '{0:05d}'.format(idx) + ".png"))

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier):
    render_path = os.path.join(model_path, name, "ours{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours{}".format(iteration), "gt")
    colormask_path = os.path.join(model_path, name, "ours{}".format(iteration), "objects_feature16")
    gt_colormask_path = os.path.join(model_path, name, "ours{}".format(iteration), "gt_objects_color")
    pred_obj_path = os.path.join(model_path, name, "ours{}".format(iteration), "objects_pred")
    

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(colormask_path, exist_ok=True)
    makedirs(gt_colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        rendering_obj = results["render_object"]
        logits = classifier(rendering_obj)
        pred_obj = torch.argmax(logits,dim=0)
        pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))

        gt_objects = view.objects
        gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8))

        rgb_mask = feature_to_rgb(rendering_obj)
        Image.fromarray(rgb_mask).save(os.path.join(colormask_path, '{0:05d}'.format(idx) + ".png"))
        Image.fromarray(gt_rgb_mask).save(os.path.join(gt_colormask_path, '{0:05d}'.format(idx) + ".png"))
        Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, '{0:05d}'.format(idx) + ".png"))
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


    out_path = os.path.join(render_path[:-8],'concat')
    makedirs(out_path,exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*'DIVX') 
    size = (gt.shape[-1]*5,gt.shape[-2])
    fps = float(5) if 'train' in out_path else float(1)
    writer = cv2.VideoWriter(os.path.join(out_path,'result.mp4'), fourcc, fps, size)

    for file_name in sorted(os.listdir(gts_path)):
        gt = np.array(Image.open(os.path.join(gts_path,file_name)))
        rgb = np.array(Image.open(os.path.join(render_path,file_name)))
        gt_obj = np.array(Image.open(os.path.join(gt_colormask_path,file_name)))
        render_obj = np.array(Image.open(os.path.join(colormask_path,file_name)))
        pred_obj = np.array(Image.open(os.path.join(pred_obj_path,file_name)))

        result = np.hstack([gt,rgb,gt_obj,pred_obj,render_obj])
        result = result.astype('uint8')

        Image.fromarray(result).save(os.path.join(out_path,file_name))
        writer.write(result[:,:,::-1])

    writer.release()

def rename_to_images(dir,input_names, output_names):
    for i,name in enumerate(input_names):
        src_path = os.path.join(dir,name)
        image =  Image.open(src_path)
        tgt_name = output_names[i]
        tgt_path = os.path.join(dir,tgt_name)
        image.save(tgt_path)
        os.remove(src_path)

def move_to_data(model_path, name, iteration, data_path):
    # source dir must exsit
    source_dir = os.path.join(model_path, name, "ours{}".format(iteration), "inpaint_object_mask_255")
    in_names= sorted([name for name in os.listdir(source_dir) if not os.path.isdir(os.path.join(source_dir,name))])
    in_dirs=sorted([name for name in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir,name))])
    train_dir = os.path.join(data_path,'images')
    train_names = sorted(os.listdir(train_dir))
    # out_dir = os.path.join(data_path,"inpaint_object_mask_255")
    # makedirs(out_dir,exist_ok=True)
    out_dir = os.path.join(data_path,"inpaint_object_mask_255")
    
    for dir in in_dirs:
        if(os.path.exists(os.path.join(out_dir,dir))):
            shutil.rmtree(os.path.join(out_dir,dir))
        shutil.move(os.path.join(source_dir,dir),out_dir)
    
    for name in in_names:
        if(os.path.exists(os.path.join(out_dir,name))):
            os.remove(os.path.join(out_dir,name))
        shutil.move(os.path.join(source_dir,name),out_dir)

    # for whole masks
    rename_to_images(out_dir,in_names,train_names)
    # for single mask
    for dir in in_dirs:
        dir_name = os.path.join(out_dir,dir)
        sub_in_names = sorted(os.listdir(dir_name))
        rename_to_images(dir_name,sub_in_names,train_names)
    
    

def removal(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, opt : OptimizationParams, select_obj_id : int, removal_thresh : float, support_faces):
    # 1. load gaussian checkpoint
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    num_classes = dataset.num_classes
    print("Num classes: ",num_classes)
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    classifier.cuda()
    classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth")))
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    save_mask_binary(dataset.model_path,"train",'_object_removal/iteration_'+str(scene.loaded_iter), scene.getTrainCameras(), select_obj_id) # no need to save whole masks
    # 2. remove selected object
    for i in select_obj_id+support_faces:
        save_mask_binary(dataset.model_path,"train",'_object_removal/iteration_'+str(scene.loaded_iter), scene.getTrainCameras(), [i], str(i))

    move_to_data(dataset.model_path,"train",'_object_removal/iteration_'+str(scene.loaded_iter),dataset.source_path) #move mask to data dir

    gaussians = removal_setup(opt, dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier, select_obj_id, scene.cameras_extent, removal_thresh,dataset.source_path,support_faces)

    # 3. render new result
    scene = Scene(dataset, gaussians, load_iteration='_object_removal/iteration_'+str(scene.loaded_iter), shuffle=False)
    # additional: save removed objects
    # this is used for later inpainting

    # this is used for other modification
    

    with torch.no_grad():
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, classifier)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--config_file", type=str, default="config/object_removal/bear.json", help="Path to the configuration file")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Read and parse the configuration file
    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        exit(1)

    args.num_classes = config.get("num_classes", 200)
    args.removal_thresh = config.get("removal_thresh", 0.3)
    args.select_obj_id = config.get("select_obj_id", [34])
    args.support_faces = config.get("support_faces",[])

    # Initialize system state (RNG)
    safe_state(args.quiet)

    removal(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, opt.extract(args), args.select_obj_id, args.removal_thresh,args.support_faces)


