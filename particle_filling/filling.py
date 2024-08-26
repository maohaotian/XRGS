import torch
import os
import numpy as np
import taichi as ti
import mcubes
import csv

# 1. densify grids
# 2. identify grids whose density is larger than some threshold
# 3. filling grids with particles
# 4. identify and fill internal grids


@ti.func
def compute_density(index, pos, opacity, cov, grid_dx):
    gaussian_weight = 0.0
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                node_pos = (index + ti.Vector([i, j, k])) * grid_dx
                dist = pos - node_pos
                gaussian_weight += ti.exp(-0.5 * dist.dot(cov @ dist))

    return opacity * gaussian_weight / 8.0


@ti.kernel
def densify_grids(
    init_particles: ti.template(),
    opacity: ti.template(),
    cov_upper: ti.template(),
    grid: ti.template(),
    grid_density: ti.template(),
    grid_dx: float,
):
    for pi in range(init_particles.shape[0]):
        pos = init_particles[pi]
        x = pos[0]
        y = pos[1]
        z = pos[2]
        i = ti.floor(x / grid_dx, dtype=int)
        j = ti.floor(y / grid_dx, dtype=int)
        k = ti.floor(z / grid_dx, dtype=int)
        ti.atomic_add(grid[i, j, k], 1)
        cov = ti.Matrix(
            [
                [cov_upper[pi][0], cov_upper[pi][1], cov_upper[pi][2]],
                [cov_upper[pi][1], cov_upper[pi][3], cov_upper[pi][4]],
                [cov_upper[pi][2], cov_upper[pi][4], cov_upper[pi][5]],
            ]
        )
        sig, Q = ti.sym_eig(cov)
        sig[0] = ti.max(sig[0], 1e-8)
        sig[1] = ti.max(sig[1], 1e-8)
        sig[2] = ti.max(sig[2], 1e-8)
        sig_mat = ti.Matrix(
            [[1.0 / sig[0], 0, 0], [0, 1.0 / sig[1], 0], [0, 0, 1.0 / sig[2]]]
        )
        cov = Q @ sig_mat @ Q.transpose()
        r = 0.0
        for idx in ti.static(range(3)):
            if sig[idx] < 0:
                sig[idx] = ti.sqrt(-sig[idx])
            else:
                sig[idx] = ti.sqrt(sig[idx])

            r = ti.max(r, sig[idx])

        r = ti.ceil(r / grid_dx, dtype=int)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if (
                        i + dx >= 0
                        and i + dx < grid_density.shape[0]
                        and j + dy >= 0
                        and j + dy < grid_density.shape[1]
                        and k + dz >= 0
                        and k + dz < grid_density.shape[2]
                    ):
                        density = compute_density(
                            ti.Vector([i + dx, j + dy, k + dz]),
                            pos,
                            opacity[pi],
                            cov,
                            grid_dx,
                        )
                        ti.atomic_add(grid_density[i + dx, j + dy, k + dz], density)


@ti.kernel
def fill_dense_grids(
    grid: ti.template(),
    grid_density: ti.template(),
    grid_dx: float,
    density_thres: float,
    new_particles: ti.template(),
    start_idx: int,
    max_particles_per_cell: int,
) -> int:
    new_start_idx = start_idx
    for i, j, k in grid_density:
        if grid_density[i, j, k] > density_thres:
            if grid[i, j, k] < max_particles_per_cell:
                diff = max_particles_per_cell - grid[i, j, k]
                grid[i, j, k] = max_particles_per_cell
                tmp_start_idx = ti.atomic_add(new_start_idx, diff)

                for index in range(tmp_start_idx, tmp_start_idx + diff):
                    di = ti.random()
                    dj = ti.random()
                    dk = ti.random()
                    new_particles[index] = ti.Vector([i + di, j + dj, k + dk]) * grid_dx
    return new_start_idx


@ti.func
def collision_search(
    grid: ti.template(), grid_density: ti.template(), index, dir_type, size, threshold
) -> bool:
    dir = ti.Vector([0, 0, 0])
    if dir_type == 0:
        dir[0] = 1
    elif dir_type == 1:
        dir[0] = -1
    elif dir_type == 2:
        dir[1] = 1
    elif dir_type == 3:
        dir[1] = -1
    elif dir_type == 4:
        dir[2] = 1
    elif dir_type == 5:
        dir[2] = -1

    flag = False
    index += dir
    i, j, k = index
    while ti.max(i, j, k) < size and ti.min(i, j, k) >= 0:
        if grid_density[index] > threshold:
            flag = True
            break
        index += dir
        i, j, k = index

    return flag


@ti.func
def collision_times(
    grid: ti.template(), grid_density: ti.template(), index, dir_type, size, threshold
) -> int:
    dir = ti.Vector([0, 0, 0])
    times = 0
    if dir_type > 5 or dir_type < 0:
        times = 1
    else:
        if dir_type == 0:
            dir[0] = 1
        elif dir_type == 1:
            dir[0] = -1
        elif dir_type == 2:
            dir[1] = 1
        elif dir_type == 3:
            dir[1] = -1
        elif dir_type == 4:
            dir[2] = 1
        elif dir_type == 5:
            dir[2] = -1

        state = grid[index] > 0
        index += dir
        i, j, k = index
        while ti.max(i, j, k) < size and ti.min(i, j, k) >= 0:
            new_state = grid_density[index] > threshold
            if new_state != state and state == False:
                times += 1
            state = new_state
            index += dir
            i, j, k = index

    return times


@ti.kernel
def internal_filling(
    grid: ti.template(),
    grid_density: ti.template(),
    grid_dx: float,
    new_particles: ti.template(),
    start_idx: int,
    max_particles_per_cell: int,
    water_grid_indices: ti.template(),
    water_grid_count: ti.template(),
    exclude_dir: int,
    ray_cast_dir: int,
    threshold: float,
) -> int:
    new_start_idx = start_idx
    count = 0
    for i, j, k in grid:
        # if grid[i, j, k] == 0:
        if grid[i, j, k] < max_particles_per_cell:
            five_collision_hit = True
            six_collision_hit = True
            for dir_type in ti.static(range(6)):
                if dir_type != exclude_dir:
                    hit_test = collision_search(
                        grid=grid,
                        grid_density=grid_density,
                        index=ti.Vector([i, j, k]),
                        dir_type=dir_type,
                        size=grid.shape[0],
                        threshold=threshold,
                    )
                    five_collision_hit = five_collision_hit and hit_test
                    six_collision_hit = six_collision_hit and hit_test
                else :
                    hit_test = collision_search(
                        grid=grid,
                        grid_density=grid_density,
                        index=ti.Vector([i, j, k]),
                        dir_type=dir_type,
                        size=grid.shape[0],
                        threshold=threshold,
                    )
                    six_collision_hit = six_collision_hit and hit_test


            # 液面上方杯子内部
            if not six_collision_hit and five_collision_hit:
                pass


            # 液面内部
            if six_collision_hit and five_collision_hit:
                hit_times = collision_times(
                    grid=grid,
                    grid_density=grid_density,
                    index=ti.Vector([i, j, k]),
                    dir_type=ray_cast_dir,
                    size=grid.shape[0],
                    threshold=threshold,
                )

                if ti.math.mod(hit_times, 2) == 1:
                    diff = max_particles_per_cell - grid[i, j, k]
                    grid[i, j, k] = max_particles_per_cell
                    tmp_start_idx = ti.atomic_add(new_start_idx, diff)
                    for index in range(tmp_start_idx, tmp_start_idx + diff):
                        di = ti.random()
                        dj = ti.random()
                        dk = ti.random()
                        new_particles[index] = (
                            ti.Vector([i + di, j + dj, k + dk]) * grid_dx
                        )        
                    water_grid_indices[count] = ti.Vector([i, j, k])
                    ti.atomic_add(count, 1)

            # 杯子外面
            if not six_collision_hit and not five_collision_hit:
                pass

    water_grid_count[0] = count

    return new_start_idx


@ti.kernel
def assign_particle_to_grid(pos: ti.template(), grid: ti.template(), grid_dx: float):
    for pi in range(pos.shape[0]):
        p = pos[pi]
        i = ti.floor(p[0] / grid_dx, dtype=int)
        j = ti.floor(p[1] / grid_dx, dtype=int)
        k = ti.floor(p[2] / grid_dx, dtype=int)
        ti.atomic_add(grid[i, j, k], 1)


@ti.kernel
def compute_particle_volume(
    pos: ti.template(), grid: ti.template(), particle_vol: ti.template(), grid_dx: float
):
    for pi in range(pos.shape[0]):
        p = pos[pi]
        i = ti.floor(p[0] / grid_dx, dtype=int)
        j = ti.floor(p[1] / grid_dx, dtype=int)
        k = ti.floor(p[2] / grid_dx, dtype=int)
        particle_vol[pi] = (grid_dx * grid_dx * grid_dx) / grid[i, j, k]


@ti.kernel
def assign_particle_to_grid(
    pos: ti.template(),
    grid: ti.template(),
    grid_dx: float,
):
    for pi in range(pos.shape[0]):
        p = pos[pi]
        i = ti.floor(p[0] / grid_dx, dtype=int)
        j = ti.floor(p[1] / grid_dx, dtype=int)
        k = ti.floor(p[2] / grid_dx, dtype=int)
        ti.atomic_add(grid[i, j, k], 1)

@ti.kernel
def create_particle_mask(
    water_grid_indices: ti.template(),  # 记录了需要提取的grid对应index
    pos: ti.template(),                      # 粒子的位置信息
    grid_dx: float,                           # 网格单元的大小
    mask: ti.template(),                       # 输出的mask数组
    water_grid_count: ti.template()       # 记录了需要提取的grid的数量
):
    for idx in range(water_grid_count[0]):
        grid_index = water_grid_indices[idx]
        for pi in range(pos.shape[0]):
            pos_grid_index = ti.Vector([
                ti.floor(pos[pi][0] / grid_dx, dtype=int),
                ti.floor(pos[pi][1] / grid_dx, dtype=int),
                ti.floor(pos[pi][2] / grid_dx, dtype=int)
            ])
            
            if (pos_grid_index[0] == grid_index[0] and
                pos_grid_index[1] == grid_index[1] and
                pos_grid_index[2] == grid_index[2]):
                mask[pi] = 1  # 标记为1，表示这个粒子在需要提取的网格中

@ti.kernel
def copy_effective_indices(src: ti.template(), dst: ti.template(), count: int):
    for i in range(count):
        dst[i] = src[i]



def get_particle_volume(pos, grid_n: int, grid_dx: float, unifrom: bool = False):
    ti_pos = ti.Vector.field(n=3, dtype=float, shape=pos.shape[0])
    ti_pos.from_torch(pos.reshape(-1, 3))

    grid = ti.field(dtype=int, shape=(grid_n, grid_n, grid_n))
    particle_vol = ti.field(dtype=float, shape=pos.shape[0])

    assign_particle_to_grid(ti_pos, grid, grid_dx)
    compute_particle_volume(ti_pos, grid, particle_vol, grid_dx)

    if unifrom:
        vol = particle_vol.to_torch()
        vol = torch.mean(vol).repeat(pos.shape[0])
        return vol
    else:
        return particle_vol.to_torch()


def fill_particles(
    pos,
    opacity,
    cov,
    grid_n: int,
    max_samples: int,
    grid_dx: float,
    density_thres=2.0,
    search_thres=1.0,
    max_particles_per_cell=1,
    search_exclude_dir=4,
    ray_cast_dir=5,
    boundary: list = None,
    smooth: bool = False,
):
    from torch import nn

    pos_clone = pos.clone()
    print(f"pos size in fill: {pos_clone.shape}")

    pos_np = pos_clone.detach().cpu().numpy()
        
    min_x = np.min(pos_np[:, 0])
    max_x = np.max(pos_np[:, 0])
    min_y = np.min(pos_np[:, 1])
    max_y = np.max(pos_np[:, 1])
    min_z = np.min(pos_np[:, 2])
    max_z = np.max(pos_np[:, 2])

    print("min_x:", min_x)
    print("max_x:", max_x)
    print("min_y:", min_y)
    print("max_y:", max_y)
    print("min_z:", min_z)
    print("max_z:", max_z)

    if boundary is not None:
        assert len(boundary) == 6
        mask = torch.ones(pos_clone.shape[0], dtype=torch.bool).cuda()
        max_diff = 0.0
        for i in range(3):
            mask = torch.logical_and(mask, pos_clone[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, pos_clone[:, i] < boundary[2 * i + 1])
            max_diff = max(max_diff, boundary[2 * i + 1] - boundary[2 * i])

        pos = pos[mask]
        opacity = opacity[mask]
        cov = cov[mask]

        grid_dx = max_diff / grid_n
        new_origin = torch.tensor([boundary[0], boundary[2], boundary[4]]).cuda()
        pos = pos - new_origin

    print(f"pos.shape[0] is: {pos.shape[0]}")
    ti_pos = ti.Vector.field(n=3, dtype=float, shape=pos.shape[0])
    ti_opacity = ti.field(dtype=float, shape=opacity.shape[0])
    ti_cov = ti.Vector.field(n=6, dtype=float, shape=cov.shape[0])
    ti_pos.from_torch(pos.reshape(-1, 3))
    ti_opacity.from_torch(opacity.reshape(-1))
    ti_cov.from_torch(cov.reshape(-1, 6))

    grid = ti.field(dtype=int, shape=(grid_n, grid_n, grid_n))
    grid_density = ti.field(dtype=float, shape=(grid_n, grid_n, grid_n))
    particles = ti.Vector.field(n=3, dtype=float, shape=max_samples)
    fill_num = 0

    # compute density_field
    densify_grids(ti_pos, ti_opacity, ti_cov, grid, grid_density, grid_dx)


    # fill dense grids
    print("Calling fill_dense_grids with parameters:")
    print(f"grid: {grid.shape}")
    print(f"grid_density: {grid_density.shape}")
    # print("grid_density values:")
    # grid_density_np = grid_density.to_numpy()  
    # for i in range(grid_density_np.shape[0]):
    #     for j in range(grid_density_np.shape[1]):
    #         for k in range(grid_density_np.shape[2]):
    #             print(f"grid_density[{i}, {j}, {k}] = {grid_density_np[i, j, k]}")
    print(f"grid_dx: {grid_dx}")
    print(f"density_thres: {density_thres}")
    print(f"max_particles_per_cell: {max_particles_per_cell}")
    # fill_num = fill_dense_grids(
    #     grid,
    #     grid_density,
    #     grid_dx,
    #     density_thres,
    #     particles,
    #     0,
    #     max_particles_per_cell,
    # )

    # smooth density_field
    if smooth:
        df = grid_density.to_numpy()
        smoothed_df = mcubes.smooth(df, method="constrained", max_iters=500).astype(
            np.float32
        )
        grid_density.from_numpy(smoothed_df)
        print("smooth finished")

    # fill internal grids
    water_grid_indices = ti.Vector.field(n=3, dtype=int, shape=grid_n ** 3)
    water_grid_count = ti.field(dtype=int, shape=1)
    fill_num = internal_filling(
        grid,
        grid_density,
        grid_dx,
        particles,
        fill_num,
        max_particles_per_cell,
        water_grid_indices,
        water_grid_count,
        exclude_dir=search_exclude_dir,  # 0: x, 1: -x, 2: y, 3: -y, 4: z, 5: -z direction
        ray_cast_dir=ray_cast_dir,  # 0: x, 1: -x, 2: y, 3: -y, 4: z, 5: -z direction
        threshold=search_thres,
    )
    print("after internal grids: ", fill_num)
    
    # put new particles together with original particles
    particles_tensor = particles.to_torch()[:fill_num].cuda()
    if boundary is not None:
        particles_tensor = particles_tensor + new_origin
    particles_tensor = torch.cat([pos_clone, particles_tensor], dim=0)

    # create mask for particles
    particles_all = ti.Vector.field(n=3, dtype=float, shape=particles_tensor.shape[0])
    particles_all.from_torch(particles_tensor.reshape(-1, 3))
    mask = ti.field(dtype=int, shape=particles_tensor.shape[0])
    create_particle_mask(water_grid_indices, particles_all, grid_dx, mask, water_grid_count)
    water_mask = mask.to_torch().cuda()


    #### debug ####
    print(f"particles all count: {particles_all.shape[0]}")
    print(f"water grid indices count: {water_grid_count[0]}")
    mask_count = 0
    for i in range(water_mask.shape[0]):
        if water_mask[i] == 1:
            mask_count += 1
    print(f"Number of total indices grid count: {water_mask.shape[0]}")
    print(f"Number of water grid count: {mask_count}")

    # 保存 particles_all 为 CSV 文件
    particles_all_np = particles_all.to_numpy()
    np.savetxt("particles_all.csv", particles_all_np.reshape(-1, 3), delimiter=",", header="x,y,z")

    # 保存 water_grid_indices 为 CSV 文件
    water_grid_indices_np = water_grid_indices.to_numpy()
    water_grid_indices_np = water_grid_indices_np[:water_grid_count[0]]
    np.savetxt("water_grid_indices.csv", water_grid_indices_np.reshape(-1, 3), delimiter=",", header="i,j,k")

    # 保存 mask 为 CSV 文件
    mask_np = mask.to_numpy()
    np.savetxt("mask.csv", mask_np, delimiter=",", header="mask")

    return particles_tensor, water_mask


@ti.kernel
def get_attr_from_closest(
    ti_pos: ti.template(),
    ti_shs: ti.template(),
    ti_opacity: ti.template(),
    ti_cov: ti.template(),
    ti_new_pos: ti.template(),
    ti_new_shs: ti.template(),
    ti_new_opacity: ti.template(),
    ti_new_cov: ti.template(),
    ti_scale: ti.template(),
    ti_new_scale: ti.template(),
    ti_rotation: ti.template(),
    ti_new_rotation: ti.template()
):
    for pi in range(ti_new_pos.shape[0]):
        p = ti_new_pos[pi]
        min_dist = 1e10
        min_idx = -1
        for pj in range(ti_pos.shape[0]):
            dist = (p - ti_pos[pj]).norm()
            if dist < min_dist:
                min_dist = dist
                min_idx = pj
        ti_new_shs[pi] = ti_shs[min_idx]
        ti_new_opacity[pi] = ti_opacity[min_idx]
        ti_new_cov[pi] = ti_cov[min_idx]
        ti_new_scale[pi] = ti_scale[min_idx]
        ti_new_rotation[pi] = ti_rotation[min_idx]
    
    # # test
    # for index in range(ti_new_shs.shape[0]):
    #     ti_new_shs[index] = ti_shs[0]

    # for index in range(ti_shs.shape[0]):
    #     ti_shs[index] = ti_shs[1]


def init_filled_particles(pos, shs, cov, opacity, new_pos, scaling, rotation):
    # print(f"pos shape is : {pos.shape}")
    # print(f"shs shape is : {shs.shape}")
    # print(f"cov shape is : {cov.shape}")
    # print(f"opacity shape is : {opacity.shape}")
    # print(f"scaling shape is : {scaling.shape}")
    # print(f"rotation shape is : {rotation.shape}")


    shs = shs.reshape(pos.shape[0], -1)
    ti_pos = ti.Vector.field(n=3, dtype=float, shape=pos.shape[0])
    ti_cov = ti.Vector.field(n=6, dtype=float, shape=cov.shape[0])
    ti_shs = ti.Vector.field(n=shs.shape[1], dtype=float, shape=shs.shape[0])
    ti_opacity = ti.field(dtype=float, shape=opacity.shape[0])
    ti_pos.from_torch(pos.reshape(-1, 3))
    ti_cov.from_torch(cov.reshape(-1, 6))
    ti_shs.from_torch(shs)
    ti_opacity.from_torch(opacity.reshape(-1))
    ti_scale = ti.Vector.field(n=3, dtype=float, shape=pos.shape[0])
    ti_rotation = ti.Vector.field(n=4,dtype=float, shape=pos.shape[0])
    ti_scale.from_torch(scaling.reshape(-1,3))
    ti_rotation.from_torch(rotation.reshape(-1,4))


    new_shs = torch.mean(shs, dim=0).repeat(new_pos.shape[0], 1).cuda()
    ti_new_pos = ti.Vector.field(n=3, dtype=float, shape=new_pos.shape[0])
    ti_new_shs = ti.Vector.field(n=shs.shape[1], dtype=float, shape=new_pos.shape[0])
    ti_new_opacity = ti.field(dtype=float, shape=new_pos.shape[0])
    ti_new_cov = ti.Vector.field(n=6, dtype=float, shape=new_pos.shape[0])
    ti_new_pos.from_torch(new_pos.reshape(-1, 3))
    ti_new_shs.from_torch(new_shs)
    ti_new_scale = ti.Vector.field(n=3, dtype=float, shape=new_pos.shape[0])
    ti_new_rotation = ti.Vector.field(n=4, dtype=float, shape=new_pos.shape[0])

    get_attr_from_closest(
        ti_pos,
        ti_shs,
        ti_opacity,
        ti_cov,
        ti_new_pos,
        ti_new_shs,
        ti_new_opacity,
        ti_new_cov,
        ti_scale,
        ti_new_scale,
        ti_rotation,
        ti_new_rotation
    )

    shs_tensor = ti_new_shs.to_torch().cuda()
    opacity_tensor = ti_new_opacity.to_torch().cuda()
    cov_tensor = ti_new_cov.to_torch().cuda()
    scale_tensor = ti_new_scale.to_torch().cuda()
    rotation_tensor = ti_new_rotation.to_torch().cuda()

    shs_tensor = torch.cat([shs, shs_tensor], dim=0)
    shs_tensor = shs_tensor.view(shs_tensor.shape[0], -1, 3)
    opacity_tensor = torch.cat([opacity, opacity_tensor.reshape(-1, 1)], dim=0)
    cov_tensor = torch.cat([cov, cov_tensor], dim=0)
    scale_tensor = torch.cat([scaling, scale_tensor], dim=0)
    rotation_tensor = torch.cat([rotation,rotation_tensor],dim=0)
    return shs_tensor, opacity_tensor, cov_tensor, scale_tensor, rotation_tensor
