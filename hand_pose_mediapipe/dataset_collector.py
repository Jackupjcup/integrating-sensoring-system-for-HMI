#!/usr/bin/env python
"""
手指角度收集器测试脚本

这个脚本演示如何使用FingerAngleCollector类来收集手指角度数据和视频
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# from bend_angle_collect import FingerAngleCollector
from AllData_collect_2 import FingerAngleAndPressureSensorCollector


def main():
    """
    主函数：演示FingerAngleCollector的使用
    """
    print("=" * 60)
    print("手指角度收集器测试")
    print("=" * 60)

    # 配置参数
    handpose_config = {
        'record_dir': 'record',  # 记录目录
        'camera_id': 0,  # 摄像头ID（0通常是默认摄像头）
        'use_threading': True,  # 使用线程模式（推荐）
        'num_hands': 2,  # 检测的最大手部数量
        'model_asset_path': 'hand_landmarker.task',  # MediaPipe模型文件
        'video_codec': 'XVID',  # 视频编码器
        'video_fps': 30.0,  # 视频帧率
    }

    print("配置参数:")
    for key, value in handpose_config.items():
        print(f"  {key}: {value}")
    print()

    # 创建收集器实例
    print("初始化收集器...")
    collector = FingerAngleAndPressureSensorCollector(**handpose_config)

    print("\n使用说明:")
    print("1. 按 'r' 键开始/停止录制")
    print("2. 按 'q' 键退出程序")
    print("3. 录制时，数据将保存到record目录下的子目录中")
    print("4. 每个录制会话包含:")
    print("   - angles_data.csv: 角度数据（时间戳和角度值）")
    print("   - pressure_data.csv: 压力传感器矩阵数据")
    print("   - video.avi: 录制的视频")
    print("   - session_info.json: 会话信息")
    print()

    # 运行收集器
    try:
        collector.run()
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n测试完成")

    # 显示录制信息
    record_dir = os.path.abspath('record')
    if os.path.exists(record_dir):
        print(f"\n录制的数据保存在: {record_dir}")
        sessions = [d for d in os.listdir(record_dir)
                   if os.path.isdir(os.path.join(record_dir, d)) and d.startswith('session_')]

        if sessions:
            print(f"找到 {len(sessions)} 个录制会话:")
            for session in sorted(sessions)[-3:]:  # 显示最近3个会话
                session_path = os.path.join(record_dir, session)
                print(f"  - {session}")

                # 检查文件
                files = os.listdir(session_path) if os.path.exists(session_path) else []
                for file in files:
                    file_path = os.path.join(session_path, file)
                    size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                    print(f"    * {file} ({size:,} bytes)")
        else:
            print("没有找到录制会话（可能没有进行录制）")
    else:
        print(f"记录目录不存在: {record_dir}")


if __name__ == "__main__":
    main()