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
from PalmPoints_and_FingerBend.palm_points_class import PalmPointsClass
from ThreadingCollect_class import ThreadingCollect


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
        self.pinch_index_idx = [7, 8, 137, 138]
        self.pinch_thumb_idx = [9, 10, 139, 140]

        # Init Grease Pencil
        gpencil_data = bpy.data.grease_pencils.new("GPencil")
        gpencil = bpy.data.objects.new(gpencil_data.name, gpencil_data)
        bpy.context.collection.objects.link(gpencil)
        gp_layer = gpencil_data.layers.new("line")
        self.gp_frame = gp_layer.frames.new(bpy.context.scene.frame_current)
        self.is_drawing = False
        self.

        self.gp_stroke = self.gp_frame.strokes.new()
        self.gp_stroke.line_width = 40
        self.gp_stroke.points.add(1)
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
        if self.is_drawing:
            position = get_skin_point_positions(self.mesh_obj, self.mouse_id)

            self.gp_stroke.points.add(1)
            self.gp_stroke.points[-1].co = Vector((position[0], position[1], position[2]))
        else:
            index_record = []
            thumb_record = []
            for idx in self.pinch_index_idx:
                index_record.append(threading_collect.palm_points.latest_pressure_data[idx])
            for idx in self.pinch_thumb_idx:
                thumb_record.append(threading_collect.palm_points.latest_pressure_data[idx])
            if np.sum(index_record) > 20 and np.sum(thumb_record) > 20:
                self.gp_stroke = self.gp_frame.strokes.new()
                self.gp_stroke.line_width = 40
                self.gp_stroke.points.add(1)
                init_position = get_skin_point_positions(self.mesh_obj, self.mouse_id)
                self.gp_stroke.points[0].co = Vector((init_position[0], init_position[1], init_position[2]))
                self.is_drawing = True
            else:
                pass

    def finish(self, context):
        threading_collect.stop_event.set()
        for t in self._threads:
            t.join(timeout=0.1)
        wm = context.window_manager
        wm.event_timer_remove(self.timer)
        if threading_collect.serial and threading_collect.serial.is_open:
            threading_collect.serial.close()
        print("Function Finish")


def register():
    bpy.utils.register_class(RealTimeQuatRotation)


def unregister():
    bpy.utils.unregister_class(RealTimeQuatRotation)


if __name__ == "__main__":
    imu_config = load_config("D:/blender_workplace/modules/IMUProcessing/config.json")
    # 配置串口
    ser = serial.Serial(
        port=imu_config['serial']['port'],  # 根据实际情况修改端口号
        baudrate=imu_config['serial']['baud_rate'],  # 根据实际情况修改波特率
        timeout=imu_config['serial']['timeout']
    )

    # Initiate IMU Params
    imu_config_path = "D:/blender_workplace/modules/IMUProcessing/config.json"
    imu_1 = IMUData('mpu6050_1', imu_config_path)
    imu_2 = IMUData('mpu6050_2', imu_config_path)
    imu_3 = IMUData('mpu6050_3', imu_config_path)

    # Initiate Palm Points
    palm_points = PalmPointsClass(14, 10, "/PalmPoints_and_FingerBend/PalmPointsConfig.json")

    threading_collect = ThreadingCollect(ser=ser,
                                         imu_1=imu_1,
                                         imu_2=imu_2,
                                         imu_3=imu_3,
                                         palm_points=palm_points,
                                         duration=500)

    register()
    # 运行操作符
    bpy.ops.object.real_time_quat_rotation()





