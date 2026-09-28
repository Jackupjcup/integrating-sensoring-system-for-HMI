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
from utils import init_datetime
from illustration_based_on_data.DataBase_utils import load_data, imu_and_pressure_interval


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


def reset_all_colors(spheres_collection):
    if spheres_collection:
        for obj in spheres_collection.objects:
            if "Skin_Points_" in obj.name:
                obj.color = (0.5, 0.5, 0.5, 1.0)


def get_color(value):
    factor = value / MAX_PRESSURE_VALUE
    factor = max(0, min(1, factor))
    return (factor, 0.0, 1.0 - factor, 1.0)


class RealTimeQuatRotation(bpy.types.Operator):
    bl_idname = "object.real_time_quat_rotation"
    bl_label = "Real Time Quaternion Rotation"

    timer = None

    def execute(self, context):
        return self.invoke(context, None)

    def invoke(self, context, event):

        # Init Mesh
        MESH_NAME = "Hand_2019"
        self.mesh_obj = bpy.data.objects.get(MESH_NAME)

        # Init COUNT
        self.imu_count = 0
        self.palm_points_count = 0

        # Init Pressure Sensor Spheres
        self.spheres_collection = bpy.data.collections.get('points_display')
        if self.spheres_collection:
            self.sphere_map = {}
            for idx, obj in enumerate(self.spheres_collection.objects):
                if f"Skin_Points_{idx}" in obj.name:
                    try:
                        self.sphere_map[f"Skin_Points_{idx}"] = obj
                    except (IndexError, ValueError):
                        continue
        else:
            print("No spheres collection found")

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

        self.start_time = time.time()
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER':
            try:
                now = imu_interval * self.imu_count
                if now > (self.palm_points_count + 1) * palm_points_interval:
                    self.palm_points_count += 1

                imu_1.Quat = imu_1.Quat_df.iloc[self.imu_count].values
                imu_2.Quat = imu_2.Quat_df.iloc[self.imu_count].values
                imu_3.Quat = imu_3.Quat_df.iloc[self.imu_count].values
                palm_points.latest_pressure_data = palm_points.collected_points_df.iloc[self.palm_points_count].values

                if imu_1.Quat is not None:
                    self.wrist.rotation_quaternion = imu_1.Quat
                if imu_2.Quat is not None:
                    self.forearm.rotation_quaternion = imu_2.Quat
                if imu_3.Quat is not None:
                    self.upperarm.rotation_quaternion = imu_3.Quat
                positions_list = []

                for i, val in enumerate(palm_points.latest_pressure_data):
                    if i < len(self.sphere_map):
                        color = get_color(val)
                        self.sphere_map[f"Skin_Points_{i}"].color = color
                        positions_list.append(get_skin_point_positions(self.mesh_obj, i))
                        self.sphere_map[f"Skin_Points_{i}"].location = positions_list[-1]
                self.imu_count += 1

            except StopIteration:
                self.finish(context)
                return {'FINISHED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self.finish(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def finish(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self.timer)
        print("Function Finish")
        reset_all_colors(self.spheres_collection)
        timestamp = init_datetime()[: 19]


def register():
    bpy.utils.register_class(RealTimeQuatRotation)


def unregister():
    bpy.utils.unregister_class(RealTimeQuatRotation)


if __name__ == "__main__":
    imu_config = load_config("D:/blender_workplace/modules/IMUProcessing/config.json")

    # Initiate IMU Params
    imu_config_path = "D:/blender_workplace/modules/IMUProcessing/config.json"
    imu_1 = IMUData('mpu6050_1', imu_config_path)
    imu_2 = IMUData('mpu6050_2', imu_config_path)
    imu_3 = IMUData('mpu6050_3', imu_config_path)

    # Initiate Palm Points
    ROWS, COLS = 14, 10
    TOTAL_NUM = ROWS * COLS
    palm_points = PalmPointsClass(ROWS, COLS, "/PalmPoints_and_FingerBend/PalmPointsConfig.json")
    MAX_PRESSURE_VALUE = 2000

    # Load Data
    data_csv_dic = "D:/Blender4.2/blender4.2_workspace/modules/DataCSV"
    log_time = "2026-01-11_16_09_00_bottle"

    imu_1_Quat_fp = os.path.join(data_csv_dic, log_time, "imu1/Quat.csv")
    imu_2_Quat_fp = os.path.join(data_csv_dic, log_time, "imu2/Quat.csv")
    imu_3_Quat_fp = os.path.join(data_csv_dic, log_time, "imu3/Quat.csv")
    pressure_fp = os.path.join(data_csv_dic, log_time, "PalmPressure_data.csv")

    imu_1.Quat_df = load_data(imu_1_Quat_fp)
    imu_2.Quat_df = load_data(imu_2_Quat_fp)
    imu_3.Quat_df = load_data(imu_3_Quat_fp)
    palm_points.collected_points_df = load_data(pressure_fp)

    imu_interval, palm_points_interval = imu_and_pressure_interval(imu_1_Quat_fp, pressure_fp)

    register()
    # 运行操作符
    bpy.ops.object.real_time_quat_rotation()





