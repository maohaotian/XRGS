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
import cv2
from utils.loss_utils import masked_l1_loss
from random import randint
import lpips
import json
import sys

from render import feature_to_rgb, visualize_obj
from edit_object_removal import points_inside_convex_hull


def mask_to_bbox(mask):
    # Find the rows and columns where the mask is non-zero
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]
    
    return xmin, ymin, xmax, ymax

def crop_using_bbox(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    return image[:, ymin:ymax+1, xmin:xmax+1]

# Function to divide image into K x K patches
def divide_into_patches(image, K):
    B, C, H, W = image.shape
    patch_h, patch_w = H // K, W // K
    patches = torch.nn.functional.unfold(image, (patch_h, patch_w), stride=(patch_h, patch_w))
    patches = patches.view(B, C, patch_h, patch_w, -1)
    return patches.permute(0, 4, 1, 2, 3)

def finetune_inpaint(opt, model_path, iteration, views, gaussians, pipeline, background, classifier, selected_obj_ids, cameras_extent, removal_thresh, finetune_iteration,density_iteration,mask_path):
    # all_masks=[]
    # for selected_id in selected_obj_ids: # this is for delete
    # # get 3d gaussians idx corresponding to select obj id
    #     with torch.no_grad():
    #         logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
    #         prob_obj3d = torch.softmax(logits3d,dim=0)
    #         mask = prob_obj3d[[selected_id], :, :] > removal_thresh
    #         mask3d = mask.any(dim=0).squeeze()

    #         mask3d_convex = points_inside_convex_hull(gaussians._xyz.detach(),mask3d,outlier_factor=1.0)
    #         mask3d = torch.logical_or(mask3d,mask3d_convex)
    #         mask3d = mask3d.float()[:,None,None]
    #         all_masks.append(mask3d)
    # all_masks_tensor = torch.stack(all_masks, dim=0)
    # final_mask3d = all_masks_tensor.any(dim=0).float()
    final_mask3d = torch.load(os.path.join(model_path,"final_mask3d.pth"))

    # fix some gaussians
    gaussians.inpaint_setup(opt,final_mask3d)
    iterations = finetune_iteration
    progress_bar = tqdm(range(iterations), desc="Finetuning progress")
    LPIPS = lpips.LPIPS(net='vgg')
    for param in LPIPS.parameters():
        param.requires_grad = False
    LPIPS.cuda()

    # object_images = []   
    # # read mask from each sub directory
    # # follow the sequence as later
    # for dir in os.listdir(mask_path):
    #     dir_path = os.path.join(mask_path,dir)
    #     if(not os.path.isdir(dir_path)):
    #         continue
    #     mask_images = []
    #     for filename in os.listdir(dir_path):
    #         mask_image = torch.Tensor(np.array(Image.open(os.path.join(dir_path,filename))))
    #         mask_images.append(mask_image)
    #     object_images.append(mask_images)
    source_path = os.path.dirname(mask_path)
    image_truth_path = os.path.join(source_path,"images")
    
    # ignore if there is no mask ?（must!）
    for iteration in range(iterations):
        ineffective_flag = True #can't ignore part of the image
        while(ineffective_flag):
            ineffective_flag = False # add to jump the loop
            viewpoint_stack = views.copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            render_pkg = render(viewpoint_cam, gaussians, pipeline, background)
            image, viewspace_point_tensor, visibility_filter, radii, objects = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["render_object"]

            mask2d = viewpoint_cam.objects > 128 # here need each one to be dealt
            gt_image = viewpoint_cam.original_image.cuda()

            # use compicated method here. method1
            gt_image_path = os.path.join(image_truth_path,viewpoint_cam.image_name+".jpg")
            if(not os.path.exists(gt_image_path)):
                gt_image_path = os.path.join(image_truth_path,viewpoint_cam.image_name+".png")
            gt_image_array = np.array(Image.open(gt_image_path))
            gt_original_image = torch.Tensor(gt_image_array.astype(np.float32) / 255.0).permute(2,0,1).clamp(0.0,1.0).to("cuda")
            # end here
            
            # Ll1 = masked_l1_loss(image, gt_image, ~mask2d) # this can use `all mask`。

            K=2
            lpips_loss = 0

            bounding_box_mask = torch.zeros_like(mask2d,dtype=torch.bool).to("cuda")

            for selected_id in selected_obj_ids:
                single_mask_path = os.path.join(mask_path,str(selected_id),viewpoint_cam.image_name+".jpg")
                # print("single_path:",single_mask_path)
                mask_single = torch.Tensor(np.array(Image.open(single_mask_path)))
                mask_single_2d = mask_single > 128
                if(torch.all(mask_single_2d.eq(0))):
                    continue
                bbox = mask_to_bbox(mask_single_2d)
                # print("bbox:",bbox)
                if(not (bbox[2] - bbox[0] >=32 and bbox[3] - bbox[1] >=32)):
                    continue
                ineffective_flag = False
                # print("iteration name bbox:",iteration, viewpoint_cam.image_name, bbox)
                cropped_image = crop_using_bbox(image, bbox)
                cropped_gt_image = crop_using_bbox(gt_image, bbox)
                # K = 2
                rendering_patches = divide_into_patches(cropped_image[None, ...], K)
                gt_patches = divide_into_patches(cropped_gt_image[None, ...], K)
                # print("input size",rendering_patches.shape,gt_patches.shape)
                lpips_loss += LPIPS(rendering_patches.squeeze()*2-1,gt_patches.squeeze()*2-1).mean()
                bounding_box_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = True

            if(ineffective_flag):
                continue
            
            inverse_mask2d = ~mask2d
            inpaint_cal_mask = inverse_mask2d & bounding_box_mask
            original_cal_mask = ~bounding_box_mask

            # Image.fromarray(inpaint_cal_mask.cpu().numpy()).save("./test/{}_inpaint_cal_mask_{}.png".format(iteration,viewpoint_cam.image_name))
            # Image.fromarray(original_cal_mask.cpu().numpy()).save("./test/{}_original_mask_{}.png".format(iteration,viewpoint_cam.image_name))
            # Image.fromarray(bounding_box_mask.cpu().numpy()).save("./test/{}_bounding_{}.png".format(iteration,viewpoint_cam.image_name))
            # Ll1 = (masked_l1_loss(image, gt_image, inpaint_cal_mask) * inpaint_cal_mask.sum() + masked_l1_loss(image, gt_original_image, original_cal_mask) * original_cal_mask.sum())/(inpaint_cal_mask.sum() + original_cal_mask.sum())
            Ll1 = masked_l1_loss(image, gt_image, mask2d) + masked_l1_loss(image,gt_original_image,inverse_mask2d) #Ll1 is much better

            # using three level loss: method2 here.

            # print("lpips_loss:",lpips_loss)
            # lpips_loss = lpips_loss / len(selected_obj_ids)
            # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * lpips_loss
            loss = Ll1
            loss.backward()

            with torch.no_grad():
                if iteration < density_iteration : # too high leads to failure
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if  iteration % 300 == 0:
                        size_threshold = 20 
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, cameras_extent, size_threshold)
                    
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
                progress_bar.update(10)
    progress_bar.close()
    
    # save gaussians
    point_cloud_path = os.path.join(model_path, "point_cloud_object_inpaint/iteration_{}".format(iteration))
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    return gaussians




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


def inpaint(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, opt : OptimizationParams, select_obj_id : int, removal_thresh : float,  finetune_iteration: int, density_iteration:int):
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


    # 2. inpaint selected object
    gaussians = finetune_inpaint(opt, dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier, select_obj_id, scene.cameras_extent, removal_thresh, finetune_iteration,density_iteration, os.path.join(dataset.source_path,dataset.object_path))

    # 3. render new result
    dataset.object_path = 'object_mask'
    dataset.images = 'images'
    scene = Scene(dataset, gaussians, load_iteration='_object_inpaint/iteration_'+str(finetune_iteration-1), shuffle=False)
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
    args.images = config.get("images", "images")
    args.object_path = config.get("object_path", "object_mask")
    args.resolution = config.get("r", 1)
    args.lambda_dssim = config.get("lambda_dlpips", 0.5)
    args.finetune_iteration = config.get("finetune_iteration", 10_000)
    args.density_iteration = config.get("density_iteration", 5000)

    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    inpaint(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, opt.extract(args), args.select_obj_id, args.removal_thresh, args.finetune_iteration,args.density_iteration)