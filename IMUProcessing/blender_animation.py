import numpy as np
import serial
import re
import time
import pandas as pd
import math
import threading
from collections import deque
from threading import Thread, Event
import bpy
from imu_threading import ThreeIMUCollector
from MPU6050_utils import IMUData, load_config


# 线程安全的数据缓冲区
imu_buffer = deque(maxlen=10)
quat_buffer = deque(maxlen=1)
# stop_event = Event()

three_imu_buffer = {'imu_1':imu_buffer, 'imu_2':imu_buffer, 'imu_3':imu_buffer}
three_quat_buffer = {'imu_1':quat_buffer, 'imu_2':quat_buffer, 'imu_3':quat_buffer}


class RealTimeQuatRotation(bpy.types.Operator):
    bl_idname = "object.real_time_quat_rotation"
    bl_label = "Real Time Quaternion Rotation"

    timer = None

    def execute(self, context):
        return self.invoke(context, None)

    def invoke(self, context, event):

        # 启动数据采集线程
        three_imu_collector.stop_event.clear()
        self._threads = [
            Thread(target=three_imu_collector.collect_imu_raw_data)
        ]
        for t in self._threads:
            t.daemon = True
            t.start()

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

            except StopIteration:
                self.finish(context)
                return {'FINISHED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self.finish(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def finish(self, context):
        three_imu_collector.stop_event.set()
        for t in self._threads:
            t.join(timeout=0.1)
        wm = context.window_manager
        wm.event_timer_remove(self.timer)
        if three_imu_collector.serial and three_imu_collector.serial.is_open:
            three_imu_collector.serial.close()
        print("Function Finish")


def register():
    bpy.utils.register_class(RealTimeQuatRotation)


def unregister():
    bpy.utils.unregister_class(RealTimeQuatRotation)


if __name__ == "__main__":
    config = load_config()
    # 配置串口
    ser = serial.Serial(
        port=config['serial']['port'],  # 根据实际情况修改端口号
        baudrate=config['serial']['baud_rate'],  # 根据实际情况修改波特率
        timeout=config['serial']['timeout']
    )

    # Initiate IMU Params
    COUNT_NUM = 0
    imu_1 = IMUData('mpu6050_1')
    imu_2 = IMUData('mpu6050_2')
    imu_3 = IMUData('mpu6050_3')

    three_imu_collector = ThreeIMUCollector(ser, imu_1, imu_2, imu_3, count_num=COUNT_NUM, duration=100)

    register()
    # 运行操作符
    bpy.ops.object.real_time_quat_rotation()





