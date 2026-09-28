import bpy
from mathutils import Vector
import numpy as np

# obj = bpy.context.active_object
# print([v.index for v in obj.data.vertices if v.select])

# the indexs of 5 fingers' top surface [28586, 177779, 206387, 216350, 252101]

TARGET_POINTS = {
    "TI": 44961,
    "II": 113232,
    "MI": 13298,
    "RI": 216551,
    "PI": 63630,

    #    "TM":334975,
    #    "IM":89881,
    #    "MM":50601,
    #    "RM":155671,
    #    "PM":3641,

    #    "IP":150199,
    #    "MP":189927,
    #    "RP":346805,
    #    "PP":272841,
    #    "TMetacarpal":85343,

}


def get_skin_point_positions(mesh_obj, point):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    mesh_data = eval_obj.to_mesh()
    positions = None
    vidx = point
    if vidx < len(mesh_data.vertices):
        local_co = mesh_data.vertices[vidx].co
        world_co = mesh_obj.matrix_world @ local_co
        positions = [round(world_co.x, 4), round(world_co.y, 4), round(world_co.z, 4)]

    eval_obj.to_mesh_clear()
    return positions


def create_sphere_at_coords(
        coords,  # 三维坐标，格式：[x, y, z]
        sphere_name="Point_Sphere",  # 小球名称
        radius=0.01,  # 小球半径（可根据模型比例调整）
        segments=16,  # 小球分段数（数值越大越光滑）
        rings=8  # 小球环数
):
    """
    在指定世界坐标位置创建UV球体
    :param coords: 世界坐标 [x, y, z]
    :param sphere_name: 球体对象名称
    :param radius: 球体半径
    :param segments: 经度方向分段数
    :param rings: 纬度方向环数
    :return: 创建的球体对象
    """
    # 1. 删除同名对象（避免重复创建报错）
    if sphere_name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[sphere_name])

    # 2. 创建UV球体（默认在世界原点(0,0,0)）
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        segments=segments,
        ring_count=rings,
        enter_editmode=False,  # 直接在对象模式创建
        align='WORLD',  # 对齐世界坐标系
        location=(0, 0, 0)  # 先创建在原点，后续再移动
    )

    # 3. 获取新建的球体对象，重命名并移动到指定坐标
    sphere_obj = bpy.context.active_object
    sphere_obj.name = sphere_name
    # 设置球体的世界坐标（核心步骤：移动到指定位置）
    sphere_obj.location = (coords[0], coords[1], coords[2])

    return sphere_obj


def create_spheres_from_positions(
        positions_list,  # 坐标列表，格式：[[x1,y1,z1], [x2,y2,z2], ...]
        base_name="Point_Sphere",  # 小球基础名称（会自动加序号，如Point_Sphere_0）
        radius=0.01,
        segments=16,
        rings=8
):
    """
    批量生成小球，每个坐标对应一个小球
    :param positions_list: 三维坐标列表
    :param base_name: 小球基础名称，自动追加序号区分
    :param radius: 小球半径
    :param segments: 经度分段数
    :param rings: 纬度环数
    :return: 生成的小球对象列表
    """
    created_spheres = []  # 存储生成的所有小球对象
    for idx, coords in enumerate(positions_list):
        # 为每个小球生成唯一名称（基础名+序号），避免重名
        sphere_name = f"{base_name}_{idx}"
        # 调用单个生成函数创建小球
        sphere = create_sphere_at_coords(
            coords=coords,
            sphere_name=sphere_name,
            radius=radius,
            segments=segments,
            rings=rings
        )
        created_spheres.append(sphere)
    return created_spheres


MESH_NAME = "Hand_2019"

mesh_obj = bpy.data.objects.get(MESH_NAME)

points_id = [[252018, 63636, 54613, 13934, 208340, 208255, 113193, 113265, 134415, 34070],
             [62732, 155821, 155686, 248554, 202351, 13040, 367582, 89830, 177402, 235478],
             [244704, 272851, 13719, 346863, 192152, 12400, 144602, 355405, 234820, 335001],
             [69389, 161179, 94704, 254580, 152413, 96630, 93538, 107681, 21275, 339301],
             [255810, 162317, 162538, 10575, 118237, 106679, 364057, 362020, 141258, 60339],
             [55892, 105987, 7664, 163193, 6800, 363459, 365548, 141042, 140907, 140750],
             [163681, 32837, 105347, 119668, 130139, 32977, 363113, 92933, 140558, 140349],
             [7719, 129554, 105024, 173832, 44050, 174237, 33113, 131095, 140626, 131291],
             [163468, 119312, 26577, 26665, 365935, 363263, 362748, 140855, 182956, 9081],
             [256008, 96556, 118829, 14235, 96511, 6819, 91130, 90820, 238734, 238368],
             [292884, 55584, 255165, 69513, 161812, 117814, 107123, 91317, 337857, 84852],
             [272674, 218889, 346550, 254226, 187646, 187450, 150461, 150278, 234680, 85306],
             [248665, 352607, 352540, 244952, 50114, 198135, 358884, 145137, 59271, 84269],
             [251828, 40081, 216697, 252263, 206432, 206134, 110571, 27946, 134198, 11517]]

point_list = np.array(points_id).flatten()
position_list = []

for point in point_list:
    position_list.append(get_skin_point_positions(mesh_obj, point))

spheres = create_spheres_from_positions(position_list, "Skin_Points", 0.01)