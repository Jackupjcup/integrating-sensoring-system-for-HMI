import numpy as np
from collections import deque


class HandAngleFilter:

    def __init__(
        self,
        median_window=3,
        ema_alpha=0.4,
        max_velocity=200  # degree per second
    ):

        self.median_window = median_window
        self.ema_alpha = ema_alpha
        self.max_velocity = max_velocity

        self.median_buffers = {}
        self.prev_angles = None
        self.prev_time = None


    def update(self, angles_dict, timestamp):

        # ---------- 1 Median filter ----------
        median_filtered = {}

        for finger, joints in angles_dict.items():

            if finger not in self.median_buffers:
                self.median_buffers[finger] = {}

            median_filtered[finger] = {}

            for joint, angle in joints.items():

                if joint not in self.median_buffers[finger]:
                    self.median_buffers[finger][joint] = deque(
                        maxlen=self.median_window
                    )

                buf = self.median_buffers[finger][joint]

                if angle is not None:
                    buf.append(angle)

                if len(buf) == 0:
                    median_filtered[finger][joint] = None
                else:
                    median_filtered[finger][joint] = float(np.median(buf))


        # ---------- 2 EMA filter ----------
        if self.prev_angles is None:

            ema_filtered = median_filtered

        else:

            ema_filtered = {}

            for finger in median_filtered:

                ema_filtered[finger] = {}

                for joint in median_filtered[finger]:

                    current = median_filtered[finger][joint]
                    prev = self.prev_angles[finger][joint]

                    if current is None:
                        ema_filtered[finger][joint] = prev
                    else:
                        ema_filtered[finger][joint] = (
                            self.ema_alpha * current
                            + (1 - self.ema_alpha) * prev
                        )


        # ---------- 3 Velocity limiter ----------
        if self.prev_angles is not None and self.prev_time is not None:

            dt = timestamp - self.prev_time

            if dt > 0:

                for finger in ema_filtered:

                    for joint in ema_filtered[finger]:

                        current = ema_filtered[finger][joint]
                        prev = self.prev_angles[finger][joint]

                        if current is None or prev is None:
                            continue

                        max_delta = self.max_velocity * dt

                        delta = current - prev

                        if abs(delta) > max_delta:
                            ema_filtered[finger][joint] = prev + np.sign(delta) * max_delta


        self.prev_angles = ema_filtered
        self.prev_time = timestamp

        return ema_filtered



if __name__ == "__main__":
    angle_filter = HandAngleFilter(
        median_window=3,
        ema_alpha=0.4,
        max_velocity=300
    )

    # timestamp = time.perf_counter()
    #
    # angles = calculate_finger_angles_mediapipe(hand_landmarks)
    #
    # angles_filtered = angle_filter.update(angles, timestamp)






