import open3d as o3d
import numpy as np
import time

# --- 1. 准备您的坐标点 ---
# 在实际应用中，您将从手套中加载这些点。
# 这里我们使用一个示例点云（一个甜甜圈的形状）来演示。
# 您的点云将是 (N, 3) 的Numpy数组。
print("正在生成示例点云...")
torus = o3d.geometry.TriangleMesh.create_torus(torus_radius=1.0, tube_radius=0.4, radial_resolution=20, tubular_resolution=30)
points = np.asarray(torus.vertices)
# 模拟抓握时可能产生的噪声
points += 0.01 * np.random.randn(points.shape[0], 3)
# 您的坐标点看起来像这样:
# points = np.array([
#     [x1, y1, z1],
#     [x2, y2, z2],
#     ...
# ])
print(f"总共加载了 {len(points)} 个坐标点。")

# --- 2. 将Numpy数组转换为Open3D点云对象 ---
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
# （可选，但推荐）为点云计算法线
# 虽然Alpha Shape不强制要求，但有法线可以帮助某些可视化和后续处理
pcd.estimate_normals()
print("点云已转换并计算了法线。")

# --- 3. 核心：计算 Alpha Shape ---
# 这是最关键的一步：选择一个合适的 alpha 值
# 这个值需要您根据点云的稀疏程度进行试验。
# 一个好的起点可能是您点云平均间距的 2-5 倍。
# 对于这个甜甜圈示例，0.3 是个合适的值。
alpha = 0.3
print(f"正在使用 alpha = {alpha} 计算 Alpha Shape...")
# 调用 create_from_point_cloud_alpha_shape 函数
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

# Alpha Shape 算法可能会产生一些非流形的边或未连接的三角面片。
# 我们可以清理一下模型，让它更适合渲染。
mesh.compute_vertex_normals()
print("Alpha Shape 计算完成，已生成 3D 网格模型。")

# --- 4. 将模型保存为 .obj 文件 ---
# 您可以将这个文件导入 Blender, MeshLab 或任何3D建模软件
output_filename = "reconstructed_object.obj"
o3d.io.write_triangle_mesh(output_filename, mesh)
print(f"模型已成功保存为: {output_filename}")

# --- 5. 在 Python 中直接可视化 ---
print("正在启动Python可视化窗口...")
print("按 'W' 键可以切换显示三角形线框（您要的'三角形形态'）")
print("按 'Q' 键关闭窗口。")

# 5.1 可视化光滑着色的物体
# o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=False)

# 5.2 可视化您要求的“三角形在物体表面的形态”（即线框模式）
# mesh_show_wireframe=True 将只显示三角形的边缘
o3d.visualization.draw_geometries(
    [mesh],
    mesh_show_wireframe=True,  # <-- 这个参数是关键
    window_name="Alpha Shape - 三角形线框"
)

# 5.3 同时显示点云和重建的线框
print("显示点云和重建结果的对比...")
# 给点云上个色，方便区分
pcd.paint_uniform_color([1, 0.706, 0]) # 黄色
o3d.visualization.draw_geometries(
    [pcd, mesh],
    mesh_show_wireframe=True,
    window_name="点云 与 Alpha Shape"
)

print("流程结束。")