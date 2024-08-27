# 代码说明
- 填充逻辑在``XRGS/particle_filling/filling.py``
- 整个填充过程在``XRGS/filling_process.py``
- 代码运行需要指定输入文件，配置文件，输出位置
- 输出位置会写入多个文件，filled_by_attribute.ply是最终文件，其余是中间文件
- 代码运行指令：在XRGS目录下
``CUDA_VISIBLE_DEVICES=1 python filling_process.py --model_path particle_filling/models/teatime/ --output_path particle_filling/outputs/teatime/ --config particle_filling/configs/teatime_config.json ``
- 配置文件中的``particle_filling``说明了填充的基本参数
- 填充前会对读入的ply先进行旋转，旋转参数写在配置文件中。``rotation_degree``代表旋转角度，``rotation_axis``代表旋转轴，0对应x轴，1对应y轴，2对应z轴。


# TODO List
- 修改颜色填充逻辑，改成根据上部分的粒子颜色对填充粒子赋颜色值
- 填充范围由配置文件手动设置，改成根据bound box判断，比如设置成bound box * 0.9，但是杯子有把手之后会影响判断

# 问题
- 把ply转正后并填充后，需要转回去吗？
