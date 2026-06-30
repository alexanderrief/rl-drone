import os
import numpy as np
from gymnasium.spaces import Box
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class FollowMarkerEnv(BaseAviary):
    """
    A drone learns to fly to a stationary marker using ground truth.
      - Observation: relative position (drone - marker) + drone velocity, shape (6,)
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
        ctrl_freq        : int               = 30,
        gui              : bool              = False,
        record           : bool              = False,
        obstacles        : bool              = False,
        vision_attributes: bool              = False,
        output_folder    : str               = "results",
    ):
        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 0.5]])
                    
        self.marker_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rng = np.random.default_rng()

        self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

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

    def _computeObs(self) -> np.ndarray:
        """
        Build the 6-dim observation vector for the agent.
        Returns:
        - np.ndarray, shape (6,), dtype = float32
        - Concatenated of:
          - rel_pos (3,) : drone position - marker position [m]
          - vel     (3,) : drone velocity [m/s]
        """
        state = self._getDroneStateVector(0)
        rel_pos = (state[0:3] - self.marker_pos).astype(np.float32)
        vel = state[10:13].astype(np.float32)
        return np.concatenate([rel_pos, vel])

    def _computeReward(self) -> float:
        """
        Compute the scalar reward for the current timestep.
        Returns:
        - float (64-bit)
        - Negative distance to marker, penalized by lateral approach direction.
        Reward is maximized when the drone reaches the marker (reward -> 0).
        """
        state = self._getDroneStateVector(0)

        pos      = state[0:3]
        velocity = state[10:13]
        dist     = np.linalg.norm(pos - self.marker_pos)
        speed    = np.linalg.norm(velocity)

        if speed > 1e-3:
            displacement      = pos - self.marker_pos
            vel_dir           = velocity / speed
            projection        = np.dot(displacement, vel_dir)
            approach_weight   = 0.1 * (1 - np.exp(-dist * 10))
            return float(-(dist + approach_weight * projection))
        else:
            return float(-dist)

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
            low  = [-2.0, -2.0, 0.3],
            high = [ 2.0,  2.0, 2.5],
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
        - obs        (np.ndarray) : new observation vector, shape (6,).
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
        - Box, shape (6,), dtype = float32
        - Unbounded; mirrors the output of _computeObs().
        """
        return Box(low=np.full(6, -np.inf), high=np.full(6, np.inf), dtype=np.float32)
    
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

