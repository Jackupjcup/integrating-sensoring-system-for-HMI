import ahrs
import numpy as np
import pandas as pd
from MPU6050_utils import gyr_to_quat


imu_df = pd.read_csv(r'../DataCSV/imu_raw_data.csv')

Quat = np.array([1, 0, 0, 0])
Quat_list = np.zeros((imu_df.shape[0], 4))
acc_np = np.array(imu_df.iloc[:, 0: 3])
gyro_np = np.array(imu_df.iloc[:, 3: 6])


attitude_ahrs = ahrs.filters.Madgwick(acc=acc_np, gyr=gyro_np)
attitude_ahrs_ = pd.DataFrame(attitude_ahrs.Q)
# attitude_ahrs_.to_csv('DataCSV/Quat_Madgwick.csv', index=False)
madgwick = ahrs.filters.Madgwick(frequency=100)

for i in range(imu_df.shape[0]):
    Quat = madgwick.updateIMU(q=Quat, acc=acc_np[i], gyr=gyro_np[i])
    Quat_list[i] = Quat


# # Old Method
# for i in range(imu_df.shape[0]):
#     Quat = gyr_to_quat(Quat, gyr_np=gyro_np[i])
#     Quat_list[i] = Quat



# attitude_math = pd.DataFrame(Quat_list)
# attitude_math.to_csv('DataCSV/Quat_math.csv', index=False)





