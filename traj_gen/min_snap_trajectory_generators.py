import sys
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped

import numpy as np


class xyzMinDerivTrajectory:

    def __init__(self, xyz_waypoints: np.ndarray):
        self.t_idx = 0  # index of current segment
        self.wps = xyz_waypoints

    def add_point(self, x, y, z):
        self.wps = np.concatenate((self.wps, [[x, y, z]]), axis=0)

    def eval(self, position: np.ndarray, pos_target: np.ndarray, rotation: np.ndarray,
             treshhold: float, transform) -> np.ndarray:
        des_pos = self.wps[self.t_idx]
        if np.linalg.norm(pos_target - position) < treshhold:
            print("NOWY PUNKT ======================================================", flush=True)
            self.t_idx = (self.t_idx + 1) % len(self.wps)
            des_pos = self.wps[self.t_idx, :]

        return des_pos
