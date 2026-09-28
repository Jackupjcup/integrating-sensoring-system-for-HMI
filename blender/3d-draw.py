import sys
import os
import numpy as np
import serial
import re
import time
import pandas as pd
import math
from mathutils import Vector
import threading
from collections import deque
from threading import Thread, Event
import bpy
import bmesh

blend_dir = os.path.abspath(bpy.path.abspath("//"))
modules_dir = os.path.join(blend_dir, "modules")
if modules_dir not in sys.path:
    sys.path.append(modules_dir)

from IMUProcessing.MPU6050_utils import load_config, IMUData
from PalmPoints.palm_points_class import PalmPointsClass
from ThreadingCollect_class import ThreadingCollect
from utils import init_datetime


def get_skin_point_positions(mesh_obj, point_id):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    mesh_data = eval_obj.to_mesh()
    positions = None
    vidx = point_id
    if vidx < len(mesh_data.vertices):
        local_co = mesh_data.vertices[vidx].co
        world_co = mesh_obj.matrix_world @ local_co
        positions = [round(world_co.x, 4), round(world_co.y, 4), round(world_co.z, 4)]

    eval_obj.to_mesh_clear()
    return positions


class RealTimeQuatRotation(bpy.types.Operator):
    bl_idname = "object.real_time_quat_rotation"
    bl_label = "Real Time Quaternion Rotation"

    timer = None

    def execute(self, context):
        return self.invoke(context, None)

    def invoke(self, context, event):

        # 启动数据采集线程
        threading_collect.stop_event.clear()
        self._threads = [
            Thread(target=threading_collect.collecting)
        ]
        for t in self._threads:
            t.daemon = True
            t.start()

        # Init Mesh
        MESH_NAME = "Hand_2019"
        self.mesh_obj = bpy.data.objects.get(MESH_NAME)
        self.mouse_id = threading_collect.palm_points.blender_points_id[7]
        self.pinch_index_idx = [6, 7, 136, 137]
        self.pinch_middle_idx = [4, 5, 134, 135]
        self.pinch_thumb_idx = [8, 9, 138, 139]

        # Init Grease Pencil
        gpencil_data = bpy.data.grease_pencils.new("GPencil")
        gpencil = bpy.data.objects.new(gpencil_data.name, gpencil_data)
        bpy.context.collection.objects.link(gpencil)
        gp_layer = gpencil_data.layers.new("line")
        self.gp_frame = gp_layer.frames.new(bpy.context.scene.frame_current)
        self.is_drawing = False

        # ===== 初始化颜色材质 =====
        self.color_index = 0
        self.colors = [
            (1.0, 0.0, 0.0, 1.0),   # 红色
            (0.0, 1.0, 0.0, 1.0),   # 绿色
            (0.0, 0.0, 1.0, 1.0),   # 蓝色
            (1.0, 1.0, 0.0, 1.0),   # 黄色
            (1.0, 1.0, 1.0, 1.0),   # 白色
        ]

        self.materials = []
        for i, color in enumerate(self.colors):
            mat = bpy.data.materials.new(name=f"GP_Material_{i}")
            bpy.data.materials.create_gpencil_data(mat)
            mat.grease_pencil.color = color  # RGBA 四元组
            gpencil.data.materials.append(mat)
            self.materials.append(mat)

        self.current_finger = None  # 'index' 或 'middle'
        # ===========================

        self.gp_stroke = self.gp_frame.strokes.new()
        self.gp_stroke.line_width = 30
        self.gp_stroke.material_index = self.color_index
        self.gp_stroke.points.add(1)  # 先添加点
        init_position = get_skin_point_positions(self.mesh_obj, self.mouse_id)
        self.gp_stroke.points[0].co = Vector((init_position[0], init_position[1], init_position[2]))

        # 获取对象
        bpy.ops.object.mode_set(mode="POSE")
        self.wrist = bpy.data.objects.get("Armature").pose.bones.get('wrist')
        self.forearm = bpy.data.objects.get("Armature").pose.bones.get('ForeArm')
        self.upperarm = bpy.data.objects.get("Armature").pose.bones.get('UpperArm')
        if self.wrist is None:
            self.report({'ERROR'}, "wrist对象未找到")
            return {'CANCELLED'}
        if self.forearm is None:
            self.report({'ERROR'}, "ForeArm对象未找到")
            return {'CANCELLED'}
        if self.upperarm is None:
            self.report({'ERROR'}, "UpperArm对象未找到")
            return {'CANCELLED'}

        # 设置旋转模式
        self.wrist.rotation_mode = 'QUATERNION'
        self.forearm.rotation_mode = 'QUATERNION'
        self.upperarm.rotation_mode = 'QUATERNION'

        # 添加模态定时器
        wm = context.window_manager
        self.timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER':
            try:
                if imu_1.Quat is not None:
                    self.wrist.rotation_quaternion = imu_1.Quat
                if imu_2.Quat is not None:
                    self.forearm.rotation_quaternion = imu_2.Quat
                if imu_3.Quat is not None:
                    self.upperarm.rotation_quaternion = imu_3.Quat

                self.draw_to_here()

            except StopIteration:
                self.finish(context)
                return {'FINISHED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self.finish(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def draw_to_here(self):
        index_record = []
        thumb_record = []
        middle_record = []
        threshold = 300
        for idx in self.pinch_index_idx:
            index_record.append(threading_collect.palm_points.latest_pressure_data[idx])
        for idx in self.pinch_thumb_idx:
            thumb_record.append(threading_collect.palm_points.latest_pressure_data[idx])
        for idx in self.pinch_middle_idx:
            middle_record.append(threading_collect.palm_points.latest_pressure_data[idx])

        # 检测当前手势
        index_pinch = np.sum(index_record) > threshold and np.sum(thumb_record) > threshold
        middle_pinch = np.sum(middle_record) > threshold and np.sum(thumb_record) > threshold

        if self.is_drawing:
            position = get_skin_point_positions(self.mesh_obj, self.mouse_id)

            self.gp_stroke.points.add(1)
            self.gp_stroke.points[-1].co = Vector((position[0], position[1], position[2]))

            # 从食指切换到中指 或 从中指切换到食指 时停止当前笔画
            if (self.current_finger == 'index' and middle_pinch and not index_pinch) or \
               (self.current_finger == 'middle' and index_pinch and not middle_pinch):
                self.is_drawing = False
        else:
            # 食指捏合开始绘制
            if index_pinch:
                # 如果从另一个手指切换过来，切换颜色
                if self.current_finger != 'index':
                    self.color_index = (self.color_index + 1) % len(self.colors)
                self.current_finger = 'index'

                self.gp_stroke = self.gp_frame.strokes.new()
                self.gp_stroke.line_width = 30
                self.gp_stroke.material_index = self.color_index
                self.gp_stroke.points.add(1)  # 先添加点
                init_position = get_skin_point_positions(self.mesh_obj, self.mouse_id)
                self.gp_stroke.points[0].co = Vector((init_position[0], init_position[1], init_position[2]))
                self.is_drawing = True

            # 中指捏合开始绘制
            elif middle_pinch:
                # 如果从另一个手指切换过来，切换颜色
                if self.current_finger != 'middle':
                    self.color_index = (self.color_index + 1) % len(self.colors)
                self.current_finger = 'middle'

                self.gp_stroke = self.gp_frame.strokes.new()
                self.gp_stroke.line_width = 30
                self.gp_stroke.material_index = self.color_index
                self.gp_stroke.points.add(1)  # 先添加点
                init_position = get_skin_point_positions(self.mesh_obj, self.mouse_id)
                self.gp_stroke.points[0].co = Vector((init_position[0], init_position[1], init_position[2]))
                self.is_drawing = True

    def finish(self, context):
        threading_collect.stop_event.set()
        for t in self._threads:
            t.join(timeout=0.1)
        wm = context.window_manager
        wm.event_timer_remove(self.timer)
        if threading_collect.serial and threading_collect.serial.is_open:
            threading_collect.serial.close()
        print("Function Finish")

        timestamp = init_datetime()[: 19]

        for i in range(1, 4):
            imu_save_dir = os.path.join(
                "D:/Blender4.2/blender4.2_workspace/modules/DataCSV",
                timestamp,
                f"imu{i}"
            )
            os.makedirs(imu_save_dir, exist_ok=True)
            imu_raw_data_file_path = os.path.join(imu_save_dir, "imu_raw_data.csv")
            Quat_file_path = os.path.join(imu_save_dir, "Quat.csv")

            threading_collect.imu_dic[f"imu_{i}"].imu_data_df.insert(loc=0,
                                                                     column="dt",
                                                                     value=np.array(threading_collect.imu_dt_list))
            threading_collect.imu_dic[f"imu_{i}"].Quat_df.insert(loc=0,
                                                                 column="dt",
                                                                 value=np.array(threading_collect.imu_dt_list))

            threading_collect.imu_dic[f"imu_{i}"].imu_data_df.to_csv(imu_raw_data_file_path, index=False)
            threading_collect.imu_dic[f"imu_{i}"].Quat_df.to_csv(Quat_file_path, index=False)

        palm_pressure_data_file_path = os.path.join("D:/Blender4.2/blender4.2_workspace/modules/DataCSV", timestamp,
                                                    "PalmPressure_data.csv")
        threading_collect.palm_points.collected_points_df.insert(loc=0,
                                                                 column="dt",
                                                                 value=np.array(threading_collect.palm_points_dt_list))

        threading_collect.palm_points.collected_points_df.to_csv(palm_pressure_data_file_path, index=False)

        # Reset the attitudes
        self.wrist.rotation_quaternion = np.array([1, 0, 0, 0])
        self.forearm.rotation_quaternion = np.array([1, 0, 0, 0])
        self.upperarm.rotation_quaternion = np.array([1, 0, 0, 0])

        print("all datas have been collected")


def register():
    bpy.utils.register_class(RealTimeQuatRotation)


def unregister():
    bpy.utils.unregister_class(RealTimeQuatRotation)


if __name__ == "__main__":
    imu_config = load_config("D:/Blender4.2/blender4.2_workspace/modules/IMUProcessing/config.json")
    # 配置串口
    ser = serial.Serial(
        port=imu_config['serial']['port'],  # 根据实际情况修改端口号
        baudrate=imu_config['serial']['baud_rate'],  # 根据实际情况修改波特率
        timeout=imu_config['serial']['timeout']
    )

    # Initiate IMU Params
    imu_config_path = "D:/Blender4.2/blender4.2_workspace/modules/IMUProcessing/config.json"
    imu_1 = IMUData('mpu6050_1', imu_config_path)
    imu_2 = IMUData('mpu6050_2', imu_config_path)
    imu_3 = IMUData('mpu6050_3', imu_config_path)

    # Initiate Palm Points
    palm_points = PalmPointsClass(14, 10, "D:/Blender4.2/blender4.2_workspace/modules/PalmPoints/PalmPointsConfig.json")

    threading_collect = ThreadingCollect(ser=ser,
                                         imu_1=imu_1,
                                         imu_2=imu_2,
                                         imu_3=imu_3,
                                         palm_points=palm_points,
                                         duration=500)

    #    threading_collect.palm_points.latest_pressure_data = np.zeros(140, )

    register()
    # 运行操作符
    bpy.ops.object.real_time_quat_rotation()
