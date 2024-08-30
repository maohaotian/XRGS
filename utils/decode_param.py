import json

def decode_param_json(json_file):
    f = open(json_file)
    sim_params = json.load(f)
    material_params = {}
    # material parameters

    if "grid_lim" in sim_params.keys():
        material_params["grid_lim"] = sim_params["grid_lim"]
    else:
        material_params["grid_lim"] = 2.0

    if "n_grid" in sim_params.keys():
        material_params["n_grid"] = sim_params["n_grid"]
    else:
        material_params["n_grid"] = 50


    # preprocessing_params
    preprocessing_params = {}
    if "opacity_threshold" in sim_params.keys():
        preprocessing_params["opacity_threshold"] = sim_params["opacity_threshold"]
    else:
        preprocessing_params["opacity_threshold"] = 0.02

    if "rotation_degree" in sim_params.keys():
        preprocessing_params["rotation_degree"] = sim_params["rotation_degree"]
    else:
        preprocessing_params["rotation_degree"] = []

    if "rotation_axis" in sim_params.keys():
        preprocessing_params["rotation_axis"] = sim_params["rotation_axis"]
    else:
        preprocessing_params["rotation_axis"] = []

    if "sim_area" in sim_params.keys():
        preprocessing_params["sim_area"] = sim_params["sim_area"]
    else:
        preprocessing_params["sim_area"] = None

    if "particle_filling" in sim_params.keys():
        preprocessing_params["particle_filling"] = sim_params["particle_filling"]
        filling_params = preprocessing_params["particle_filling"]
        if not "n_grid" in filling_params.keys():
            filling_params["n_grid"] = material_params["n_grid"] * 4

        if not "density_threshold" in filling_params.keys():
            filling_params["density_threshold"] = 5.0

        if not "search_threshold" in filling_params.keys():
            filling_params["search_threshold"] = 3.0

        if not "max_particles_num" in filling_params.keys():
            filling_params["max_particles_num"] = 2000000

        if not "max_partciels_per_cell" in filling_params.keys():
            filling_params["max_partciels_per_cell"] = 1

        if not "search_exclude_direction" in filling_params.keys():
            filling_params["search_exclude_direction"] = 5

        if not "ray_cast_direction" in filling_params.keys():
            filling_params["ray_cast_direction"] = 4

        if not "boundary" in filling_params.keys():
            filling_params["boundary"] = None

        if not "smooth" in filling_params.keys():
            filling_params["smooth"] = False
        
        if not "visualize" in filling_params.keys():
            filling_params["visualize"] = False
        
        if not "water_grid_threshold" in filling_params.keys():
            filling_params["water_grid_threshold"] = 1.0

        if not "water_complete_factor" in filling_params.keys():
            filling_params["water_complete_factor"] = 1.0

        if not "water_complete_offset" in filling_params.keys():
            filling_params["water_complete_offset"] = [0.0, 0.0, 0.0]

        if not "water_scale_factor" in filling_params.keys():
            filling_params["water_scale_factor"] = 1.0

        if not "water_complete_layer" in filling_params.keys():
            filling_params["water_complete_layer"] = 0

        if not "exposure" in filling_params.keys():
            filling_params["exposure"] = False
    else:
        preprocessing_params["particle_filling"] = None

   

    return material_params, preprocessing_params


