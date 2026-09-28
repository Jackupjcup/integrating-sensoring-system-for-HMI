import numpy as np
import serial
import time
from palm_points_class import PalmPointsClass
# cd C:\Users\Admin\PycharmProjects\smart glove\MPU6050
# python imu_transition.py

# 配置串口
ser = serial.Serial(
    port='COM3',  # 根据实际情况修改端口号
    baudrate=460800,  # 根据实际情况修改波特率
    timeout=1
)


if __name__ == "__main__":
    ROWS =14
    COLS = 10
    palm_points = PalmPointsClass(ROWS, COLS)

    Count = 0
    START_TIME = time.time()
    animation_duration = 5
    is_capturing = False
    temp_matrix = []


    print("collecting points...")
    while time.time() - START_TIME < animation_duration:
        try:
            # 读取一行并解码
            line_bytes = ser.readline()
            line = line_bytes.decode('utf-8', errors='ignore').strip()
        except Exception:
            continue

        if not line:
            continue

        # --- 状态机解析 ---
        if line == "FRAME_START":
            temp_matrix = []
            is_capturing = True

        elif line == "FRAME_END":
            is_capturing = False
            # 检查是否收齐了 14 行
            if len(temp_matrix) == ROWS:
                # print(f"has been collected {ROWS} rows")
                try:
                    # 转换为 numpy 数组并发送
                    final_data = np.array(temp_matrix, dtype=float)
                    # print(final_data.shape)
                    palm_points.transform_latest_pressure_data(final_data)
                except Exception as e:
                    print(f"数据转换错误: {e}")

        elif is_capturing:
            # 解析具体数据行 "0 0 11 33..."
            try:
                vals = [int(x) for x in line.split() if x.isdigit()]
                if len(vals) == COLS:
                    temp_matrix.append(vals)
            except ValueError:
                pass


    print(palm_points.collected_points_df.shape)
    palm_points.save_collected_points_df()






