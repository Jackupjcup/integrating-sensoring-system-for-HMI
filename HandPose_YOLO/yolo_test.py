import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from ultralytics import YOLO
from caluculate_quaternions import calculate_arm_quaternion



# 1. 加载模型（yolo11n.pt 是最新的轻量级模型，速度最快，适合实时检测）
# 第一次运行会自动下载该模型文件
# model = YOLO('best.pt')
# model = YOLO('train5/weights/best.pt')
model = YOLO('yolo26n-pose.pt')
# 2. 打开摄像头 (0 通常是内置摄像头，1 或 2 可能是外接 USB 摄像头)
cap = cv2.VideoCapture(0)


# 检查摄像头是否正常打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("按下 'q' 键退出程序")

while True:
    # 逐帧读取
    ret, frame = cap.read()
    if not ret:
        break

    # 3. 对当前帧进行推理
    # stream=True 模式更节省显存/内存
    results = model(frame, stream=False, verbose=False)
    # print(results)
    # keypoints = results[0].keypoints.data.cpu().numpy()[0]  # 转numpy并取第一个人体
    #
    # # 计算左右手臂的旋转四元数
    # right_quat = calculate_arm_quaternion(keypoints, side="right")
    # left_quat = calculate_arm_quaternion(keypoints, side="left")

    # print("=== 手臂旋转四元数（w, x, y, z）===")
    # print(f"右臂：{right_quat}")
    # print(f"左臂：{left_quat}")

    # 4. 绘制识别结果并显示
    for r in results:
        # 使用 model 自带的 plot() 方法直接在图片上绘制边框和标签
        annotated_frame = r.plot()

        # 显示窗口
        cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    # 5. 退出机制：按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()





