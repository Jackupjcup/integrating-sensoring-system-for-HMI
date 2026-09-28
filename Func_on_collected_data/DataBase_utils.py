import pandas as pd
import numpy as np
from IMUProcessing.MPU6050_utils import load_config

imu_config = load_config(r"D:\blender_workplace\modules\IMUProcessing\config.json")
palm_points_config = load_config(r"/PalmPoints_and_FingerBend\PalmPointsConfig.json")


def load_data(pressure_data_file_path):
    pressure_data = pd.read_csv(pressure_data_file_path)
    if pressure_data.columns[0] == "dt":
        pressure_data = pressure_data.drop(columns=["dt"])
    return pressure_data

def imu_and_pressure_interval(imu_fp, pressure_fp):
    imu_data = pd.read_csv(imu_fp)
    pressure_data = pd.read_csv(pressure_fp)

    if imu_data.columns[0] == "dt":
        imu_interval = np.mean(imu_data["dt"].values)
    else:
        imu_interval = imu_config["imu_interval"]
    if pressure_data.columns[0] == "dt":
        pressure_interval = np.mean(pressure_data["dt"].values)
    else:
        pressure_interval = palm_points_config["Blender"]["pressure_interval"]

    return imu_interval, pressure_interval

