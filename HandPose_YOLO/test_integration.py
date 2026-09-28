"""
系统集成测试脚本
测试各个模块的功能和集成
"""

import cv2
import numpy as np
import json
import os
from camera_calibrator import CameraCalibrator, NumpyEncoder
from dual_camera_pose_system import DualCameraPoseSystem


def test_calibration_module():
    """测试标定模块"""
    print("=" * 60)
    print("测试标定模块")
    print("=" * 60)

    # 1. 创建测试标定板图像（模拟）
    print("1. 创建测试标定板图像...")
    pattern_size = (9, 6)
    square_size = 0.025

    # 创建虚拟标定器
    calibrator = CameraCalibrator(pattern_type="checkerboard",
                                 pattern_size=pattern_size,
                                 square_size=square_size)

    # 创建虚拟标定数据
    K1_test = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
    D1_test = np.zeros((1, 5), dtype=np.float32)
    K2_test = np.array([[1050, 0, 650], [0, 1050, 350], [0, 0, 1]], dtype=np.float32)
    D2_test = np.zeros((1, 5), dtype=np.float32)
    R_test = np.eye(3, dtype=np.float32)
    T_test = np.array([0.3, 0.0, 0.0], dtype=np.float32).reshape(3, 1)

    calibrator.camera1_params = {"K": K1_test, "D": D1_test, "error": 0.1}
    calibrator.camera2_params = {"K": K2_test, "D": D2_test, "error": 0.12}
    calibrator.extrinsic_params = {"R": R_test, "T": T_test, "error": 0.08}

    # 2. 测试保存和加载标定参数
    print("2. 测试保存/加载标定参数...")
    test_calib_file = "test_calibration.json"

    # 保存
    success = calibrator.save_calibration(test_calib_file)
    if success and os.path.exists(test_calib_file):
        print(f"  ✓ 标定参数已保存到 {test_calib_file}")

        # 加载
        calibrator2 = CameraCalibrator(pattern_type="checkerboard",
                                      pattern_size=pattern_size,
                                      square_size=square_size)
        if calibrator2.load_calibration(test_calib_file):
            print("  ✓ 标定参数加载成功")

            # 验证加载的数据
            K1_loaded = calibrator2.camera1_params["K"]
            if np.allclose(K1_test, K1_loaded, rtol=1e-3):
                print("  ✓ 内参矩阵正确")
            else:
                print("  ✗ 内参矩阵不匹配")
        else:
            print("  ✗ 标定参数加载失败")
    else:
        print("  ✗ 标定参数保存失败")

    # 清理
    if os.path.exists(test_calib_file):
        os.remove(test_calib_file)

    print("标定模块测试完成\n")


def test_pose_system_with_mock_data():
    """使用模拟数据测试姿态估计系统"""
    print("=" * 60)
    print("测试姿态估计系统（模拟数据）")
    print("=" * 60)

    # 1. 创建测试标定文件
    print("1. 创建测试标定数据...")
    test_calib = {
        "calibration_date": "2026-03-23",
        "pattern_type": "checkerboard",
        "pattern_size": [9, 6],
        "square_size": 0.025,
        "camera1": {
            "K": [[1000, 0, 640], [0, 1000, 360], [0, 0, 1]],
            "D": [0, 0, 0, 0, 0],
            "error": 0.1
        },
        "camera2": {
            "K": [[1050, 0, 650], [0, 1050, 350], [0, 0, 1]],
            "D": [0, 0, 0, 0, 0],
            "error": 0.12
        },
        "extrinsic": {
            "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "T": [0.3, 0.0, 0.0],
            "error": 0.08
        }
    }

    with open("test_calib_temp.json", "w") as f:
        json.dump(test_calib, f, cls=NumpyEncoder, indent=2)

    # 2. 初始化系统
    print("2. 初始化姿态估计系统...")
    try:
        system = DualCameraPoseSystem("test_calib_temp.json", "yolo26n-pose.pt")
        print("  ✓ 系统初始化成功")
    except Exception as e:
        print(f"  ✗ 系统初始化失败: {e}")
        # 清理
        if os.path.exists("test_calib_temp.json"):
            os.remove("test_calib_temp.json")
        return

    # 3. 测试四元数计算函数
    print("3. 测试四元数计算函数...")

    # 测试向量到四元数转换
    test_vector = np.array([1, 0, 0])
    test_reference = np.array([0, 0, 1])

    quat = system._vector_to_quaternion(test_vector, test_reference)
    if quat is not None and len(quat) == 4:
        print(f"  ✓ 向量到四元数转换成功: {quat}")

        # 验证四元数范数应为1
        norm = np.linalg.norm(quat)
        if abs(norm - 1.0) < 0.001:
            print(f"  ✓ 四元数范数正确: {norm:.4f}")
        else:
            print(f"  ✗ 四元数范数不正确: {norm:.4f}")
    else:
        print("  ✗ 向量到四元数转换失败")

    # 4. 测试相对旋转计算
    print("4. 测试相对旋转计算...")
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0, 1, 0])

    rel_quat = system._relative_rotation(vec1, vec2)
    if rel_quat is not None:
        print(f"  ✓ 相对旋转计算成功: {rel_quat}")
    else:
        print("  ✗ 相对旋转计算失败")

    # 5. 测试肢体四元数计算（模拟数据）
    print("5. 测试肢体四元数计算...")
    mock_points_3d = {
        "right_shoulder": np.array([0, 0, 1]),
        "right_elbow": np.array([0.2, 0, 0.8]),
        "right_wrist": np.array([0.4, 0, 0.6]),
        "left_shoulder": np.array([-0.1, 0, 1]),
        "left_elbow": np.array([-0.3, 0, 0.8]),
        "left_wrist": np.array([-0.5, 0, 0.6])
    }

    quaternions = system.calculate_limb_quaternions(mock_points_3d)
    if quaternions:
        print(f"  ✓ 肢体四元数计算成功，得到 {len(quaternions)} 个四元数")
        for name, q in quaternions.items():
            print(f"    {name}: [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]")
    else:
        print("  ✗ 肢体四元数计算失败")

    # 6. 测试时间平滑
    print("6. 测试时间平滑...")
    smoothed_points, smoothed_quats = system.apply_temporal_smoothing(
        mock_points_3d, quaternions, smoothing_factor=0.3
    )

    if smoothed_points and smoothed_quats:
        print(f"  ✓ 时间平滑成功，平滑后点: {len(smoothed_points)}，四元数: {len(smoothed_quats)}")
    else:
        print("  ✗ 时间平滑失败")

    # 清理
    if os.path.exists("test_calib_temp.json"):
        os.remove("test_calib_temp.json")

    print("姿态估计系统测试完成\n")


def test_with_real_images():
    """使用真实图像测试（如果存在）"""
    print("=" * 60)
    print("测试真实图像处理")
    print("=" * 60)

    # 检查是否存在测试图像
    test_images = ["img.png", "hand_with_keypoints.jpg"]
    available_images = []

    for img_name in test_images:
        if os.path.exists(img_name):
            available_images.append(img_name)

    if len(available_images) == 0:
        print("未找到测试图像，跳过真实图像测试")
        return

    print(f"找到测试图像: {available_images}")

    # 使用第一张图像作为测试
    test_image = available_images[0]
    img = cv2.imread(test_image)

    if img is None:
        print(f"无法读取图像: {test_image}")
        return

    print(f"图像尺寸: {img.shape}")

    # 创建一个简单的标定文件用于测试
    simple_calib = {
        "camera1": {
            "K": [[1000, 0, img.shape[1]//2],
                  [0, 1000, img.shape[0]//2],
                  [0, 0, 1]],
            "D": [0, 0, 0, 0, 0],
            "error": 0.1
        },
        "camera2": {
            "K": [[1050, 0, img.shape[1]//2],
                  [0, 1050, img.shape[0]//2],
                  [0, 0, 1]],
            "D": [0, 0, 0, 0, 0],
            "error": 0.12
        },
        "extrinsic": {
            "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "T": [0.3, 0.0, 0.0],
            "error": 0.08
        }
    }

    calib_file = "test_real_image_calib.json"
    with open(calib_file, "w") as f:
        json.dump(simple_calib, f, cls=NumpyEncoder, indent=2)

    try:
        # 初始化系统
        system = DualCameraPoseSystem(calib_file, "yolo26n-pose.pt")

        # 使用同一图像模拟双摄像头输入（实际应用应使用两个不同图像）
        points_3d, quaternions, vis_data = system.process_frame(
            img, img.copy(), apply_smoothing=False
        )

        if points_3d is not None:
            print("✓ 成功处理图像，检测到3D关键点")
            print(f"  关键点数量: {len(points_3d)}")

            # 可视化
            vis_frame1, vis_frame2 = system.draw_visualization(vis_data, points_3d, quaternions)

            # 保存结果
            output_file = "test_output.jpg"
            cv2.imwrite(output_file, vis_frame1)
            print(f"  结果已保存到: {output_file}")

            # 显示（如果可能）
            try:
                cv2.imshow("测试结果", vis_frame1)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
            except:
                pass
        else:
            print("✗ 未能检测到3D关键点")
            print("  可能原因: YOLO未检测到人体，或三角测量失败")

    except Exception as e:
        print(f"✗ 处理图像时出错: {e}")

    finally:
        # 清理
        if os.path.exists(calib_file):
            os.remove(calib_file)

    print("真实图像测试完成\n")


def run_all_tests():
    """运行所有测试"""
    print("开始系统集成测试")
    print("=" * 60)

    test_calibration_module()
    test_pose_system_with_mock_data()
    test_with_real_images()

    print("=" * 60)
    print("所有测试完成")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()