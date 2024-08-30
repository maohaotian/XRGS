import sys

sys.path.append("gaussian-splatting")

import argparse
import math
import cv2
import torch
import os
import numpy as np
import json
from tqdm import tqdm

# Gaussian splatting dependencies
from utils.sh_utils import eval_sh
from scene.gaussian_model import GaussianModel
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.cameras import Camera as GSCamera
from gaussian_renderer import render, GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov

# Particle filling dependencies
from particle_filling.filling import *

# Utils
from utils.decode_param import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *


ti.init(arch=ti.cuda, device_memory_GB=8.0)


class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

def particle_position_tensor_to_ply(position_tensor, filename):
    # position is (n,3)
    if os.path.exists(filename):
        os.remove(filename)
    position = position_tensor.clone().detach().cpu().numpy()
    num_particles = (position).shape[0]
    # print(f'num_particles: {num_particles}')
    position = position.astype(np.float32)
    with open(filename, "wb") as f:  # write binary
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_particles}
property float x
property float y
property float z
end_header
"""
        f.write(str.encode(header))
        f.write(position.tobytes())
        print("write", filename)

def load_checkpoint(model_path, sh_degree=3, iteration=-1):
    # Check if there's a .ply file directly under model_path
    ply_files = [f for f in os.listdir(model_path) if f.endswith('.ply')]
    
    if ply_files:
        # If there's a .ply file, use the first one found
        checkpt_path = os.path.join(model_path, ply_files[0])
    else:
        # If no .ply file is found, find checkpoint in point_cloud directory
        checkpt_dir = os.path.join(model_path, "point_cloud")
        if iteration == -1:
            iteration = searchForMaxIteration(checkpt_dir)
        checkpt_path = os.path.join(
            checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
        )

    # Load gaussians
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply_origin(checkpt_path)
    return gaussians


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--white_bg", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        AssertionError("Model path does not exist!")
    if not os.path.exists(args.config):
        AssertionError("Scene config does not exist!")
    if args.output_path is not None and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # load scene config
    print("Loading scene config...")
    (
        material_params,
        preprocessing_params,
    ) = decode_param_json(args.config)

    # load gaussians
    print("Loading gaussians...")
    model_path = args.model_path
    gaussians = load_checkpoint(model_path)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )

    # init the scene
    print("Initializing scene and pre-processing...")
    params = load_params_from_gs(gaussians, pipeline)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]


    # throw away low opacity kernels
    mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
    initial_mask = mask
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_screen_points = init_screen_points[mask, :]
    init_shs = init_shs[mask, :]


    # rorate and translate object
    # particle_position_tensor_to_ply(
    #     init_pos,
    #     args.output_path + "init_particles.ply",
    # )
    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )
    rotated_pos = apply_rotations(init_pos, rotation_matrices)

    # particle_position_tensor_to_ply(rotated_pos, args.output_path + "rotated_particles.ply")

    # select a sim area and save params of unslected particles
    unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (
        None,
        None,
        None,
        None,
    )
    if preprocessing_params["sim_area"] is not None:
        boundary = preprocessing_params["sim_area"]
        assert len(boundary) == 6
        mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool).to(device="cuda")
        for i in range(3):
            mask = torch.logical_and(mask, rotated_pos[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, rotated_pos[:, i] < boundary[2 * i + 1])

        unselected_pos = init_pos[~mask, :]
        unselected_cov = init_cov[~mask, :]
        unselected_opacity = init_opacity[~mask, :]
        unselected_shs = init_shs[~mask, :]

        rotated_pos = rotated_pos[mask, :]
        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]
        init_shs = init_shs[mask, :]

    transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos)
    transformed_pos = shift2center111(transformed_pos)

    # modify covariance matrix accordingly
    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov


    # particle_position_tensor_to_ply(
    #     transformed_pos,
    #     args.output_path + "transformed_particles.ply",
    # )

    # fill particles if needed
    gs_num = transformed_pos.shape[0]
    device = "cuda:0"
    filling_params = preprocessing_params["particle_filling"]

    water_mask = None
    mpm_init_pos, water_mask = fill_particles(
        pos=transformed_pos,
        opacity=init_opacity,
        cov=init_cov,
        grid_n=filling_params["n_grid"],
        max_samples=filling_params["max_particles_num"],
        grid_dx=material_params["grid_lim"] / filling_params["n_grid"],
        density_thres=filling_params["density_threshold"],
        search_thres=filling_params["search_threshold"],
        max_particles_per_cell=filling_params["max_partciels_per_cell"],
        search_exclude_dir=filling_params["search_exclude_direction"],
        ray_cast_dir=filling_params["ray_cast_direction"],
        boundary=filling_params["boundary"],
        smooth=filling_params["smooth"],
        water_grid_threshold = filling_params["water_grid_threshold"],
        water_complete_factor = filling_params["water_complete_factor"],
        water_complete_offset = filling_params["water_complete_offset"],
        water_complete_layer = filling_params["water_complete_layer"],
        exposure=filling_params["exposure"]
    )
    mpm_init_pos = mpm_init_pos.to(device=device)

    if gs_num == mpm_init_pos.shape[0]:
        print("Warning: None of particles are filled, please change the filling parameters.")
        sys.exit(1)

    xyz, features_dc, features_rest, scaling, rotation, opacity, normals = gaussians.get_attributes()
    scaling = scaling[initial_mask, :]
    rotation = rotation[initial_mask, :]
    shs, opacity, mpm_init_cov, scale, rotation = init_filled_particles(
         mpm_init_pos[:gs_num],
        init_shs,
        init_cov,
        init_opacity,
        mpm_init_pos[gs_num:],
        scaling,
        rotation,
        water_mask
    )
    # gs_num = mpm_init_pos.shape[0]

    mpm_init_pos = undo_all_transforms(
        mpm_init_pos, rotation_matrices, scale_origin, original_mean_pos
    )

    fill_particle_count = mpm_init_pos.shape[0]
    # 存储填充后的液体粒子
    water_scale = scale
    scale_factor = filling_params["water_scale_factor"]
    print("Water scale factor: ", scale_factor)
    water_scale = water_scale * scale_factor
    gaussians.save_filled_ply_by_attributes(args.output_path + "fill_water.ply",
        mpm_init_pos, shs, opacity, mpm_init_cov ,water_scale, rotation ,water_mask)

    # 存储填充后的杯子粒子
    fill_cup_mask = 1 - water_mask
    gaussians.save_filled_ply_by_attributes(args.output_path + "fill_cup.ply",
        mpm_init_pos, shs, opacity, mpm_init_cov ,scale, rotation ,fill_cup_mask)

    # 存储原始杯子粒子
    origin_cup_mask = fill_cup_mask.clone()
    origin_cup_mask[gs_num:] = 0
    gaussians.save_filled_ply_by_attributes(args.output_path + "origin_cup.ply",
        mpm_init_pos, shs, opacity, mpm_init_cov ,scale, rotation ,origin_cup_mask)

    # 存储原始杯子粒子和填充后的水粒子
    fill_water_origin_cup_mask = torch.zeros_like(water_mask)
    for i in range(fill_particle_count):
        if water_mask[i] == 1 or origin_cup_mask[i] == 1:
            fill_water_origin_cup_mask[i] = 1
    fill_water_origin_cup_scale = scale.clone()
    for i in range(fill_particle_count):
        if water_mask[i] == 1:
            fill_water_origin_cup_scale[i] *= scale_factor
    gaussians.save_filled_ply_by_attributes(args.output_path + "fill_water_origin_cup.ply",
        mpm_init_pos, shs, opacity, mpm_init_cov ,fill_water_origin_cup_scale, rotation ,fill_water_origin_cup_mask)

    # 存储填充后的杯子粒子和填充后的水粒子
    fill_water_fill_cup_mask = torch.zeros_like(water_mask)
    for i in range(fill_particle_count):
        fill_water_fill_cup_mask[i] = 1
    fill_water_fill_cup_scale = fill_water_origin_cup_scale.clone()
    gaussians.save_filled_ply_by_attributes(args.output_path + "fill_water_fill_cup.ply",
        mpm_init_pos, shs, opacity, mpm_init_cov ,fill_water_fill_cup_scale, rotation ,fill_water_fill_cup_mask)