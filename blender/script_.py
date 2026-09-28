import numpy as np
import re
import time
import pandas as pd
import math
import socket
import json
from collections import deque
# from threading import Thread, Event
import bpy
import bmesh
import threading
import queue

data_queue = queue.Queue()

def socket_listener(sock, q):

    while True:
        try:
            data, addr = sock.recvfrom(2048)

            if data:
                angles = json.loads(data.decode("utf-8"))
                q.put(angles)

        except Exception:
            pass

'''Blender Parameters'''
fingers = {
    "thumb": ["TMetacarpal", "TM", "TI"],
    "index": ["IP", "IM", "II"],
    "middle": ["MP", "MM", "MI"],
    "ring": ["RP", "RM", "RI"],
    "pinky": ["PP", "PM", "PI"]
}
armature = bpy.data.objects["Armature"]
bpy.context.view_layer.objects.active = armature
bpy.ops.object.mode_set(mode='POSE')

MAX_ANGLE = 180


class RealTimeHandPose(bpy.types.Operator):
    bl_idname = "object.real_time_hand_pose"
    bl_label = "Real Time Hand Pose"

    timer = None
    sock = None

    def execute(self, context):
        return self.invoke(context, None)

    def invoke(self, context, event):

        # 获取对象
        for f_chain in fingers.values():
            for bone_name in f_chain:
                if bone_name != "TMetacarpal":
                    bone = armature.pose.bones[bone_name]
                    if bone is None:
                        self.report({'ERROR'}, f"{bone_name}对象未找到")
                    else:
                        bone.rotation_mode = "XYZ"
                        bone.rotation_euler = (0.0, 0.0, 0.0)

        # Initiate Socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 5005))
        self.sock.setblocking(True)  # 设为非阻塞模式

        self.thread = threading.Thread(
            target=socket_listener,
            args=(self.sock, data_queue),
            daemon=True
        )
        self.thread.start()

        # 添加模态定时器
        wm = context.window_manager
        self.timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER':
            try:
                if not data_queue.empty():
                    angles = data_queue.get()
                # if data:
                #     angles = json.loads(data.decode("utf-8"))
                #     print(angles)

                    c = 0
                    for f_chain in fingers.values():
                        for bone_name in f_chain:
                            if bone_name != "TMetacarpal":
                                bone = armature.pose.bones[bone_name]
                                target_z_rad = -math.radians(MAX_ANGLE - angles[c])
                                current_euler = bone.rotation_euler
                                bone.rotation_euler = (current_euler.x, current_euler.y, target_z_rad)
                                c += 1

            except BlockingIOError:
                pass

            except StopIteration:
                self.finish(context)
                return {'FINISHED'}
            except Exception as e:
                print(f"Receive Error: {e}")

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self.finish(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def finish(self, context):
        #        for t in self._threads:
        #            t.join(timeout=0.1)
        wm = context.window_manager
        wm.event_timer_remove(self.timer)
        print("Function Finish")


def register():
    bpy.utils.register_class(RealTimeHandPose)


def unregister():
    bpy.utils.unregister_class(RealTimeHandPose)


if __name__ == "__main__":
    register()
    # 运行操作符
    bpy.ops.object.real_time_hand_pose()







