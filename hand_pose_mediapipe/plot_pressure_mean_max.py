"""
对每个session的dataset.csv中的pressure_data (n*14*10)和angles (n*14)，
计算每帧pressure的平均值和最大值，以及angles的平均值（乘180转换为角度），
绘制时间序列曲线，保存到各自的session目录中。

图使用双y轴：左边为角度(0-180)，右边为压力值。
三条折线：pressure max, pressure min (mean), angles
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RECORD_DIR = os.path.join(os.path.dirname(__file__), "record")


def parse_numpy_str(val):
    """解析numpy数组字符串表示，返回shape=(14,10)的数组。

    numpy数组字符串格式如 '[[ -1.5  2.3]\n [ 3.4  -5.6]]'，
    数字间用空格分隔而非逗号，需要正则处理。
    """
    # 合并多行为单行
    val = val.replace("\n", " ").replace("\r", " ").strip()

    # 用正则将数字间的空格替换为逗号
    # 匹配: 数字(包括负号和小数点) + 空格 + 数字(包括负号和小数点)
    # 在中间插入逗号
    val = re.sub(r'(?<=\d|\.)\s+(?=-?\d)', ', ', val)
    val = re.sub(r'(?<=\])\s+(?=\[)', ', ', val)

    # 现在可以用 ast.literal_eval
    import ast
    try:
        arr = ast.literal_eval(val)
        return np.array(arr, dtype=np.float64)
    except Exception as e:
        return None


def load_pressure_from_csv(csv_path):
    """从dataset.csv加载pressure_data，返回shape=(n,14,10)的数组。"""
    df = pd.read_csv(csv_path)
    if "pressure" not in df.columns:
        return None
    pressures = []
    for val in df["pressure"]:
        arr = parse_numpy_str(val)
        if arr is None:
            continue
        if arr.shape == (14, 10):
            pressures.append(arr)
        else:
            # 尝试reshape
            flat = arr.flatten()
            if len(flat) == 140:
                pressures.append(flat.reshape(14, 10))
            else:
                continue
    if not pressures:
        return None
    return np.stack(pressures)  # shape: (n, 14, 10)


def load_angles_from_csv(csv_path):
    """从dataset.csv加载angles数据，返回shape=(n,14)的数组。"""
    df = pd.read_csv(csv_path)
    if "angles" not in df.columns:
        return None
    angles_list = []
    for val in df["angles"]:
        arr = parse_numpy_str(val)
        if arr is None:
            continue
        angles_list.append(arr.flatten())
    if not angles_list:
        return None
    return np.stack(angles_list)  # shape: (n, 14)


def plot_pressure_stats(pressure_3d, session_dir, session_name, angles_2d=None):
    """绘制每帧平均值和最大值的时间曲线，以及angles平均值，保存到session目录。"""
    n = pressure_3d.shape[0]
    mean_per_frame = pressure_3d.reshape(n, -1).mean(axis=1)
    max_per_frame = pressure_3d.reshape(n, -1).max(axis=1)
    frames = np.arange(n)

    fig, ax1 = plt.subplots(figsize=(14, 5))

    # 右边y轴: 压力值
    ax1.plot(frames, mean_per_frame, linewidth=0.5, color="steelblue", label="pressure min")
    ax1.plot(frames, max_per_frame, linewidth=0.5, color="firebrick", label="pressure max")
    ax1.set_ylabel("Pressure", color="black")
    ax1.set_xlabel("Frame Index")
    ax1.tick_params(axis="y", labelcolor="black")

    # 左边y轴: 角度 0-180
    if angles_2d is not None:
        # 取angles与pressure帧数对齐的部分
        n_angles = min(n, len(angles_2d))
        angles_mean = angles_2d[:n_angles].mean(axis=1) * 180  # 归一化值转角度
        ax2 = ax1.twinx()
        ax2.plot(frames[:n_angles], angles_mean, linewidth=0.5, color="green", label="angles")
        ax2.set_ylabel("Angle (deg)", color="green")
        ax2.set_ylim(0, 180)
        ax2.tick_params(axis="y", labelcolor="green")

    ax1.set_title(f"{session_name} — Frame-wise Pressure & Angles")

    # 合并两个轴的legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    if angles_2d is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(session_dir, "pressure_mean_max_curve.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    if not os.path.isdir(RECORD_DIR):
        print(f"Record directory not found: {RECORD_DIR}")
        return

    sessions = sorted(
        d for d in os.listdir(RECORD_DIR)
        if d.startswith("session_") and os.path.isdir(os.path.join(RECORD_DIR, d))
    )
    print(len(sessions))

    processed = 0
    skipped = 0
    for session_name in sessions[-3:]:
        session_dir = os.path.join(RECORD_DIR, session_name)
        csv_path = os.path.join(session_dir, "dataset.csv")
        if not os.path.isfile(csv_path):
            skipped += 1
            continue

        print(f"Processing {session_name} ...")
        pressure_3d = load_pressure_from_csv(csv_path)
        if pressure_3d is None:
            print(f"  Skipped (no valid pressure data)")
            skipped += 1
            continue

        print(f"  Loaded pressure data: {pressure_3d.shape}")

        # 加载 angles 数据
        angles_2d = load_angles_from_csv(csv_path)
        if angles_2d is not None:
            print(f"  Loaded angles data: {angles_2d.shape}")
        else:
            print(f"  No angles data found")

        plot_pressure_stats(pressure_3d, session_dir, session_name, angles_2d)
        processed += 1

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}")


if __name__ == "__main__":
    main()