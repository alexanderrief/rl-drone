import os
import numpy as np
import cv2
import pybullet as p

from gymnasium.spaces import Box
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from src.utils.vision_utils import compute_center

class VisualFollowMarkerEnv(BaseAviary):
    """
    A drone learns to fly to a stationary marker using visual input.
      - Observation: (cx, cy, depth) from camera image, shape (3,)
      - Action:      relative position offset [dx, dy, dz]
      - Reward:      negative distance to marker
    """

    def __init__(
        self,
        drone_model      : DroneModel        = DroneModel.CF2X,
        num_drones       : int               = 1,
        initial_xyzs     : np.ndarray | None = None,
        initial_rpys     : np.ndarray | None = None,
        physics          : Physics           = Physics.PYB,
        pyb_freq         : int               = 240,
        ctrl_freq        : int               = 24,
        gui              : bool              = False,
        record           : bool              = False,
        obstacles        : bool              = True,
        vision_attributes: bool              = True,
        output_folder    : str               = "results",
    ):
        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 0.5]])
                    
        self.marker_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rng = np.random.default_rng()

        self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

        self._sphere_urdf = os.path.join(os.path.dirname(__file__), "../assets/urdf/sphere.urdf")
        self.marker_id = None

        self._lastObs = -np.ones(3, dtype=np.float32)
        super().__init__(
            drone_model       = drone_model,
            num_drones        = num_drones,
            initial_xyzs      = initial_xyzs,
            initial_rpys      = initial_rpys,
            physics           = physics,
            pyb_freq          = pyb_freq,
            ctrl_freq         = ctrl_freq,
            gui               = gui,
            record            = record,
            obstacles         = obstacles,
            vision_attributes = vision_attributes,
            user_debug_gui    = False,
            output_folder     = output_folder,
        )
        self.IMG_RES = np.array([32, 24])
        
    def _addObstacles(self):
        self.marker_id = p.loadURDF(
            self._sphere_urdf,
            basePosition=self.marker_pos.tolist(),
            useFixedBase=True,
            physicsClientId=self.CLIENT,
        )

    def _computeObs(self)-> np.ndarray:
        """
        Build the 3-dim observation vector for the agent.
        Returns:
        - np.ndarray, shape (3,), dtype = float32
        - Concatenated of:
          - cx    (1,) : horizontal centroid of red marker in camera image [px]
          - cy    (1,) : vertical centroid of red marker in camera image [px]
          - depth (1,) : depth at centroid from PyBullet depth map [0, 1]
        - Returns zeros if no red marker is visible.
        """
        rgb, dep, _ = self._getDroneImages(0)
        hsv =  cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2HSV)
        lower = np.array([0, 100, 100])    # untere Grenze
        upper = np.array([10, 255, 255])   # obere Grenze
        mask = cv2.inRange(hsv, lower, upper)
        center = compute_center(mask)
        
        h, w = dep.shape
        if center is None:
            self._lastObs = -np.ones(3, dtype=np.float32)
        else:
            cx, cy = center
            cx_norm = (cx - w / 2) / (w / 2)
            cy_norm = (cy - h / 2) / (h / 2)
            depth = float(dep[cy, cx])
            self._lastObs = np.array([cx_norm, cy_norm, depth], dtype=np.float32)
        return self._lastObs

        

    def _computeReward(self) -> float:
        """
        Compute the scalar reward for the current timestep.
        Returns:
        - float (64-bit)
        - -2.0 if marker is not visible (small penalty to encourage searching).
        - -(cx_norm^2 + cy_norm^2) - depth otherwise:
            rewards centering the marker in the image and approaching it.
            Reward is maximized (-> 0) when marker is centered and close.
        """
        if self._lastObs[0] == -1:
            return -2.0
        cx_norm, cy_norm, depth = self._lastObs
        return float(-(cx_norm ** 2 + cy_norm ** 2) - depth)

    def _computeTerminated(self) -> bool:
        """
        Check whether the episode should terminate early.
        Returns:
        - bool
        - True if the drone leaves the allowed flight volume:
          - z < 0.05  : drone has crashed (below ground)
          - z > 3.0   : drone has flown too high
          - |x| > 2.5 : drone left horizontal boundary
          - |y| > 2.5 : drone left horizontal boundary
        """
        state = self._getDroneStateVector(0)
        x, y, z = state[0], state[1], state[2]
        
        if z < 0.05 or z > 3.0:
            return True
        if abs(x) > 2.5 or abs(y) > 2.5:
            return True
        return False
    
    def _computeTruncated(self) -> bool:
        return False
    
    def _computeInfo(self) -> dict:
        return {}
    
    def reset(self, seed: int | None = None, options: dict | None = None):
        """
        Reset the environment and randomize the marker position.
        Args:
        - seed    (int,  optional) : RNG seed passed to the base class.
        - options (dict, optional) : additional options passed to the base class.
        Returns:
        - obs  (np.ndarray) : initial observation after reset.
        - info (dict)       : info from the base class.
        """
        self.marker_pos = self.rng.uniform(
            low  = [0.0, 0.0, 0.3],
            high = [ 0.5,  0.5, 1],
        ).astype(np.float32)
        self.ctrl.reset()
        obs, info = super().reset(seed=seed, options=options)  # type: ignore[arg-type]
        return obs, info
    
    def step(self, action: np.ndarray):
        """
        Advance the simulation by one control step.
        Args:
        - action (np.ndarray, shape (3,)) : relative position offset [dx, dy, dz] [m].
        Returns:
        - obs        (np.ndarray) : new observation vector, shape (3,).
        - reward     (float)      : scalar reward for this step.
        - terminated (bool)       : True if the drone left the flight volume.
        - truncated  (bool)       : True if episode was cut short (always False here).
        - info       (dict)       : info from the base class.
        """
        obs, _, terminated, truncated, info = super().step(action)
        reward     = self._computeReward()
        terminated = self._computeTerminated()
        return obs, reward, terminated, truncated, info
    
    def _actionSpace(self):
        """
        Define the continuous action space.
        Returns:
        - Box, shape (3,), dtype = float32
        - Each action is a relative position offset [dx, dy, dz] [m].
        """
        return Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)
    
    def _observationSpace(self):
        """
        Define the continuous observation space.
        Returns:
        - Box, shape (3,), dtype = float32
        - Mirrors the output of _computeObs(): (cx, cy, depth).
        """
        return Box(low=np.full(3, -np.inf), high=np.full(3, np.inf), dtype=np.float32)
    
    def _preprocessAction(self, action: np.ndarray):
        """
        Convert a relative position offset into motor RPMs via PID control.
        Args:
        - action (np.ndarray, shape (3,)) : relative position offset [dx, dy, dz] [m].
        Returns:
        - rpm (np.ndarray, shape (4,)) : target RPMs for each rotor.
        """
        state       = self._getDroneStateVector(0)
        destination = state[0:3] + action
        next_pos    = self._calculateNextStep(
            current_position=state[0:3],
            destination=destination,
            step_size=0.05, #type: ignore[arg-type]
        )
        rpm, _, _ = self.ctrl.computeControl(
            control_timestep=self.CTRL_TIMESTEP,
            cur_pos=state[0:3],
            cur_quat=state[3:7],
            cur_vel=state[10:13],
            cur_ang_vel=state[13:16],
            target_pos=next_pos,
        )
        return rpm

