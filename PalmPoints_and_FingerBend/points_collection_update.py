import bpy
import math

class RealTimeController(bpy.types.Operator):
    """一个可以扩展的实时控制操作符"""
    bl_idname = "wm.realtime_controller"
    bl_label = "可拓展实时控制器"

    # Blender 会把这个操作符注册为系统事件循环的一部分
    _timer = None         # 计时器对象
    _running_time = 0.0   # 已运行的时间，用于调试或动画计算

    def invoke(self, context, event):
        print("操作符启动：创建计时器并进入事件循环...")
        wm = context.window_manager

        # 1. 创建计时器 (0.02s ~ 50FPS)
        self._timer = wm.event_timer_add(0.02, window=context.window)

        # 2. 将当前操作符加入“模态事件循环”
        wm.modal_handler_add(self)

        # 3. 初始化状态 (例如姿态缓冲、输入队列等)
        self._running_time = 0.0

        print("✅ 计时器创建成功，开始运行！")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        # ESC 退出
        if event.type == 'ESC':
            print("🔴 检测到 ESC，正在停止操作符...")
            return self.cancel(context)

        # 只在计时器事件触发时更新
        if event.type == 'TIMER':
            self._running_time += 0.02
            self.update_scene(context)

        # 允许其他事件通过（旋转视图、缩放等）
        return {'PASS_THROUGH'}

    def update_scene(self, context):
        """
        这里写“每帧执行一次”的逻辑。
        后续你可以在这里加上：
        - 手势输入映射
        - 外部数据同步
        - 动画/物理模拟
        """

        # 获取对象
        obj = context.view_layer.objects.active or context.scene.objects.get("Cube")
        if not obj:
            print("⚠️ 未找到 'Cube' 或激活对象，跳过更新")
            return

        # 示例：让物体做轻微上下波动运动（比单调上升更直观）
        obj.location.z = 1.0 + 0.5 * math.sin(self._running_time * 2)
        obj.location.x = 1.0 + 0.5 * math.sin(self._running_time * 2)
        print(f"⏱️ 运行时间={self._running_time:.2f}s | Cube Z={obj.location.z:.2f}")

    def cancel(self, context):
        print("🧹 停止并移除计时器...")
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
        print("✅ 操作符已停止。")
        return {'CANCELLED'}

def register():
    bpy.utils.register_class(RealTimeController)
    print("✅ RealTimeController 已注册")


def unregister():
    bpy.utils.unregister_class(RealTimeController)
    print("♻️ RealTimeController 已注销")

if __name__ == "__main__":
    # 防止重复注册报错
    try:
        unregister()
    except Exception:
        pass

    register()

    # 启动
    bpy.ops.wm.realtime_controller('INVOKE_DEFAULT')
