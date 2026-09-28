"""
对每个session的dataset.csv中的pressure_data (n*14*10)，
计算每帧的平均值和最大值，绘制时间序列曲线，保存到各自的session目录中。
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


def plot_pressure_stats(pressure_3d, session_dir, session_name):
    """绘制每帧平均值和最大值的时间曲线，保存到session目录。"""
    n = pressure_3d.shape[0]
    # 每帧: 展平为(14*10,)后计算
    mean_per_frame = pressure_3d.reshape(n, -1).mean(axis=1)
    max_per_frame = pressure_3d.reshape(n, -1).max(axis=1)
    frames = np.arange(n)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(frames, mean_per_frame, linewidth=0.5, color="steelblue", label="Mean")
    ax.plot(frames, max_per_frame, linewidth=0.5, color="firebrick", label="Max")
    ax.set_ylabel("Pressure")
    ax.set_xlabel("Frame Index")
    ax.set_title(f"{session_name} — Frame-wise Mean & Max Pressure")
    ax.legend()
    ax.grid(True, alpha=0.3)

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

    processed = 0
    skipped = 0
    for session_name in sessions:
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
        plot_pressure_stats(pressure_3d, session_dir, session_name)
        processed += 1

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}")


if __name__ == "__main__":
    main()