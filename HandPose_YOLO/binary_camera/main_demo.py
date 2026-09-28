"""
双摄像头实时人体姿态估计演示程序
运行前请确保已完成摄像头标定
"""

import cv2
import numpy as np
import argparse
import time
from dual_camera_pose_system import DualCameraPoseSystem


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="双摄像头实时人体姿态估计")
    parser.add_argument("--cam1", type=int, default=0, help="摄像头1索引 (默认: 0)")
    parser.add_argument("--cam2", type=int, default=1, help="摄像头2索引 (默认: 1)")
    parser.add_argument("--calibration", type=str, default="calibration.json",
                       help="标定文件路径 (默认: calibration.json)")
    parser.add_argument("--model", type=str, default="yolo26n-pose.pt",
                       help="YOLO模型路径 (默认: yolo26n-pose.pt)")
    parser.add_argument("--width", type=int, default=1280, help="摄像头宽度 (默认: 1280)")
    parser.add_argument("--height", type=int, default=720, help="摄像头高度 (默认: 720)")
    parser.add_argument("--fps", type=int, default=30, help="摄像头FPS (默认: 30)")
    parser.add_argument("--no-smoothing", action="store_true", help="禁用时间平滑")
    parser.add_argument("--show-3d", action="store_true", help="显示3D可视化窗口")
    parser.add_argument("--record", action="store_true", help="录制结果视频")
    parser.add_argument("--output-dir", type=str, default="output", help="输出目录")

    return parser.parse_args()


def setup_cameras(cam1_idx, cam2_idx, width, height, fps):
    """设置并打开两个摄像头"""
    print(f"打开摄像头: cam1={cam1_idx}, cam2={cam2_idx}")
    print(f"分辨率: {width}x{height}, FPS: {fps}")

    cap1 = cv2.VideoCapture(cam1_idx)
    cap2 = cv2.VideoCapture(cam2_idx)

    # 设置摄像头参数
    for i, cap in enumerate([cap1, cap2], 1):
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 {i}")
            return None, None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # 验证设置
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))

        print(f"摄像头{i}实际设置: {actual_width}x{actual_height}, {actual_fps}FPS")

    return cap1, cap2


def setup_video_writers(output_dir, width, height, fps):
    """设置视频录制器"""
    import os
    from datetime import datetime

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    writer1 = cv2.VideoWriter(
        os.path.join(output_dir, f"cam1_{timestamp}.avi"),
        cv2.VideoWriter_fourcc(*'XVID'),
        fps,
        (width, height)
    )

    writer2 = cv2.VideoWriter(
        os.path.join(output_dir, f"cam2_{timestamp}.avi"),
        cv2.VideoWriter_fourcc(*'XVID'),
        fps,
        (width, height)
    )

    return writer1, writer2


def create_3d_visualization_window():
    """创建3D可视化窗口（简化版，使用2D投影）"""
    # 这里可以扩展为真正的3D可视化，暂时使用空白图像
    return np.zeros((480, 640, 3), dtype=np.uint8)


def draw_3d_visualization(frame_3d, points_3d, quaternions):
    """绘制3D可视化（简化版）"""
    # 清空画布
    frame_3d.fill(0)

    if points_3d is None or quaternions is None:
        cv2.putText(frame_3d, "等待数据...", (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame_3d

    height, width = frame_3d.shape[:2]

    # 定义关键点颜色
    colors = {
        "right_shoulder": (0, 255, 0),
        "right_elbow": (0, 200, 255),
        "right_wrist": (0, 100, 255),
        "left_shoulder": (255, 0, 0),
        "left_elbow": (255, 100, 0),
        "left_wrist": (255, 200, 0)
    }

    # 转换3D坐标到2D屏幕坐标（简单的正交投影）
    points_2d = {}
    scale = 100  # 缩放因子
    offset_x, offset_y = width // 2, height // 2

    for name, point in points_3d.items():
        if name in colors:
            # 简单投影：忽略Y轴（深度），使用X和Z
            x = int(point[0] * scale + offset_x)
            y = int(-point[2] * scale + offset_y)  # Z轴向上

            # 限制在图像范围内
            x = max(10, min(width - 10, x))
            y = max(10, min(height - 10, y))

            points_2d[name] = (x, y)

    # 绘制点
    for name, (x, y) in points_2d.items():
        color = colors.get(name, (255, 255, 255))
        cv2.circle(frame_3d, (x, y), 8, color, -1)
        cv2.putText(frame_3d, name.split('_')[1][0], (x-5, y+5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 绘制骨架连线
    skeleton = [
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "left_shoulder")
    ]

    for start, end in skeleton:
        if start in points_2d and end in points_2d:
            cv2.line(frame_3d, points_2d[start], points_2d[end], (255, 255, 255), 2)

    # 显示四元数信息
    y_offset = 30
    for name in ["right_upper_arm", "right_elbow"]:
        if name in quaternions:
            q = quaternions[name]
            text = f"{name}: [{q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f}]"
            cv2.putText(frame_3d, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            y_offset += 20

    return frame_3d


def main():
    """主函数"""
    args = parse_arguments()

    print("=" * 60)
    print("双摄像头实时人体姿态估计系统")
    print("=" * 60)
    print(f"摄像头: {args.cam1}, {args.cam2}")
    print(f"标定文件: {args.calibration}")
    print(f"模型: {args.model}")
    print(f"分辨率: {args.width}x{args.height}")
    print(f"时间平滑: {'禁用' if args.no_smoothing else '启用'}")
    print("=" * 60)

    # 1. 设置摄像头
    cap1, cap2 = setup_cameras(args.cam1, args.cam2, args.width, args.height, args.fps)

    if cap1 is None or cap2 is None:
        return

    # 2. 初始化姿态估计系统
    print("初始化姿态估计系统...")
    system = DualCameraPoseSystem(args.calibration, args.model)

    # 3. 设置视频录制（如果启用）
    writer1, writer2 = None, None
    if args.record:
        print("启用视频录制...")
        writer1, writer2 = setup_video_writers(args.output_dir, args.width, args.height, args.fps)

    # 4. 设置3D可视化窗口（如果启用）
    frame_3d = None
    if args.show_3d:
        frame_3d = create_3d_visualization_window()

    print("\n控制说明:")
    print("  'q' - 退出程序")
    print("  's' - 保存当前帧的3D数据")
    print("  'p' - 暂停/继续")
    print("  'd' - 切换调试显示")
    print("=" * 60)
    print("开始实时处理...")

    # 主循环
    frame_count = 0
    paused = False
    show_debug = False
    last_print_time = time.time()

    try:
        while True:
            if not paused:
                # 读取帧
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()

                if not ret1 or not ret2:
                    print("摄像头读取失败")
                    break

                # 处理帧
                points_3d, quaternions, vis_data = system.process_frame(
                    frame1, frame2, apply_smoothing=not args.no_smoothing
                )

                # 可视化
                vis_frame1, vis_frame2 = system.draw_visualization(
                    vis_data, points_3d if show_debug else None, quaternions if show_debug else None
                )

                # 显示结果
                cv2.imshow("Camera 1 - 姿态估计", vis_frame1)
                cv2.imshow("Camera 2 - 姿态估计", vis_frame2)

                # 3D可视化
                if args.show_3d and points_3d is not None:
                    frame_3d = draw_3d_visualization(frame_3d, points_3d, quaternions)
                    cv2.imshow("3D Visualization", frame_3d)

                # 录制视频
                if args.record and writer1 is not None and writer2 is not None:
                    writer1.write(vis_frame1)
                    writer2.write(vis_frame2)

                # 定期打印结果
                current_time = time.time()
                if current_time - last_print_time > 2.0:  # 每2秒打印一次
                    if points_3d is not None:
                        system.print_results(points_3d, quaternions, frame_count)
                    last_print_time = current_time

                frame_count += 1

            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("用户退出")
                break
            elif key == ord('p'):
                paused = not paused
                print(f"{'已暂停' if paused else '已继续'}")
            elif key == ord('s'):
                # 保存当前帧数据
                if points_3d is not None:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    np.savez(f"frame_{timestamp}.npz",
                            points_3d=points_3d,
                            quaternions=quaternions,
                            frame_count=frame_count)
                    print(f"已保存帧数据到 frame_{timestamp}.npz")
            elif key == ord('d'):
                show_debug = not show_debug
                print(f"调试显示: {'开启' if show_debug else '关闭'}")

    except KeyboardInterrupt:
        print("\n程序被中断")

    finally:
        # 清理资源
        print("\n正在清理资源...")

        cap1.release()
        cap2.release()

        if writer1 is not None:
            writer1.release()
        if writer2 is not None:
            writer2.release()

        cv2.destroyAllWindows()

        # 显示性能统计
        stats = system.get_performance_stats()
        print("\n性能统计:")
        print(f"  处理帧数: {stats['frames_processed']}")
        print(f"  平均FPS: {stats['fps']:.1f}")
        print(f"  平均处理时间: {stats['avg_time_ms']:.1f}ms")
        print(f"  标定误差: {stats['calibration_error']:.4f}")

        print("程序结束")


if __name__ == "__main__":
    main()