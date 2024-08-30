# 代码说明
## 文件说明
- 填充逻辑在``XRGS/particle_filling/filling.py``
- 整个填充过程在``XRGS/filling_process.py``
- 输出位置会写入多个文件，``fill_water.ply``是填充后的液体粒子，``fill_cup.ply``是填充后的杯子粒子，可以理解为边界粒子，调``water_complete_factor``这个参数可以调整杯壁的厚度，``origin_cup.ply``是原始杯子粒子，``fill_water_origin_cup.ply``是原始杯子粒子和填充后的水粒子，``fill_water_fill_cup.ply``是填充后的杯子粒子和填充后的水粒子。不同应用场景可以选择渲染不同的文件。

## 代码运行
- 代码运行需要指定输入文件，配置文件，输出位置
- 代码运行指令：在XRGS目录下
``CUDA_VISIBLE_DEVICES=1 python filling_process.py --model_path particle_filling/models/teatime/ --output_path particle_filling/outputs/teatime/ --config particle_filling/configs/teatime_config.json ``

## 配置文件参数说明
- ``opacity_threshold``：填充前用作透明度过滤，透明度低于此值的被过滤。
- ``rotation_degree``代表旋转角度，``rotation_axis``代表旋转轴，0对应x轴，1对应y轴，2对应z轴。填充前会对读入的ply先进行旋转。
- ``particle_filling``：填充的基本参数
    - ``n_grid``：网格化一维的gird个数，模型最终网格化成gird*grid*grid。
    - ``density_threshold``：密度填充的阈值，低于此密度的网格将进行填充。（当前代码逻辑中没用到）
    - ``search_threshold``：填充粒子时做射线检测时的阈值。
    - ``search_exclude_direction``：做五次射线检测时不检测的方向。
    0: x, 1: -x, 2: y, 3: -y, 4: z, 5: -z
    - ``ray_cast_direction``：用作碰撞次数检测的方向。
    - ``max_particles_num``：填充后的模型最大粒子数。
    - ``max_partciels_per_cell``：填充过程中每个grid中最大粒子数，填充时每个要填充的grid都会填充到这个数值。
    - ``smooth``：射线填充前是否做平滑处理。
    - ``water_grid_threshold``：寻找液体部分时射线检测的阈值。
    - ``water_complete_factor``：液体部分体积的收缩参数。
    - ``water_complete_offset``：液体部分体积的位移参数。当把物体转正后，竖直方向是k轴。这里前两维分别代表ij两个方向的位移，第三维代表k方向上的削去部分。比如第三维填2，则会直接削去最底下两层grid，而非整体向上移动两个grid。
    - ``water_complete_layer``：控制在找到最大和最小的层高值后，向内延伸多少层来确定四个边界点的范围(需要大于等于1)
    - ``water_scale_factor``：液体部分高斯核scale的调整参数，大于1时高斯核变小，小于1时高斯核变大。


## 需要优化的地方
- ``filling.py``里的325行，``complete_water_grid_indices``函数直接暴力改成串行，性能较差。