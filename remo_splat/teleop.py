# Simple scripts for teleop the robot, it used the joy node of ros to get the joystick data
#
import os

import numpy as np
import pyjoystick
import spatialgeometry as sg
import spatialmath as sm
from pyjoystick.sdl2 import Joystick, Key, run_event_loop


class Teleop:
    def __init__(self, env, initial_pose=sm.SE3()):
        self.speed = np.zeros(6)
        self.axis = sg.Axes(0.1, pose=initial_pose)
        self.initial_pose = initial_pose
        env.add(self.axis)
        repeater = pyjoystick.HatRepeater(
            first_repeat_timeout=0.5, repeat_timeout=0.03, check_timeout=0.01
        )

        mngr = pyjoystick.ThreadEventManager(
            event_loop=run_event_loop,
            handle_key_event=self.handle_key_event,
            button_repeater=repeater,
        )
        mngr.start()

    def get_pose(self):
        return sm.SE3(self.axis.T)

    def handle_key_event(self, key):
        if key.keytype == Key.BUTTON:
            if key.number == 3:
                self.speed[2] = 1 * key.value
            elif key.number == 0:
                self.speed[2] = -1 * key.value
            elif key.number == 1:
                self.speed[1] = 1 * key.value
            elif key.number == 2:
                self.speed[1] = -1 * key.value
        elif key.keytype == Key.HAT:
            if key.value == Key.HAT_UP:
                self.speed[0] = 1
            elif key.value == Key.HAT_DOWN:
                self.speed[0] = -1
            elif key.value == Key.HAT_LEFT:
                self.speed[5] = 1
            elif key.value == Key.HAT_RIGHT:
                self.speed[5] = -1
            elif key.number == Key.HAT_CENTERED:
                self.speed[0] = 0
                self.speed[5] = 0

        elif key.keytype == Key.AXIS:
            if key.number == 3:  # [Left rigth]
                self.speed[3] = 1 * key.value
            elif key.number == 4:  # [Up down]
                self.speed[4] = -1 * key.value
        self.axis.v = self.speed

    def step(self) -> bool:
        return False


class TeleopRecorder(Teleop):
    def __init__(self, env, initial_pose=sm.SE3(), scene_name="scene", exp_name="exp"):
        super().__init__(env, initial_pose)
        self.record = False
        self.recorded = []
        self.id = 0
        self.scene_name = scene_name
        self.exp_name = exp_name
        os.makedirs(f"data/trajs/{self.scene_name}/{self.exp_name}", exist_ok=True)

    def step(self):
        if self.record:
            self.recorded.append(self.axis.T)

    def handle_key_event(self, key):
        if key.keytype == Key.BUTTON:
            if key.number == 4:
                self.record = not self.record
                if self.record:
                    print("Recording")
                else:
                    print("Stop recording")
                    self._save_recorded_data()

        super().handle_key_event(key)

    def _save_recorded_data(self):
        filename = f"data/trajs/{self.scene_name}/{self.exp_name}/{self.id}.npy"
        np.save(filename, self.recorded)
        print(f"Data saved in {filename}")
        self.recorded = []
        self.id += 1
        self.axis.T = self.initial_pose
        self.axis.v = np.zeros(6)


class ReplayTeleop(Teleop):
    """
    Replay the recorded data
    """

    def __init__(
        self,
        env,
        scene_name,
        exp_name,
        n_interpolate=1,
        recorder_rate=0.03,
        query_rate=0.03,
    ):
        self.folder = os.path.join("data/trajs", scene_name, exp_name)
        filename = os.path.join(self.folder, "0.npy")
        self.load_poses(filename, n_interpolate)
        super().__init__(env, self.poses[0])
        self._traj = 0  # Traj id
        self._step = 0  # Current step
        self._inc_step = query_rate / recorder_rate  # Interpolation of poses
        print(self._inc_step)
        self.n_interpolate = n_interpolate

    def load_poses(self, filename, n_interpolate: int):
        self.data = np.load(filename)
        self.poses = [sm.SE3(A) for A in self.data]

    def step(self):
        """
        Get next pose
        Returns
        -------
        bool
            True if the recording is done
        """
        if self._step < len(self.poses) - 1:
            i = int(self._step)
            self.axis.T = self.poses[i].interp(self.poses[i + 1], self._step - i)
            self._step += self._inc_step
        else:
            return True

        return self._step >= len(self.poses)

    def reset_traj(self):
        self._step = 0

    def load_trajectory(self, id: int):
        """
        Load a specific trajectory
        """
        filename = os.path.join(self.folder, f"{id}.npy")
        self.load_poses(filename, self.n_interpolate)
        self._traj = id
        self._step = 0

    def step_trajectory(self):
        """
        Step to the next saved trajectory
        Returns
        -------
        bool
            True if there is not more trajectories in the folder
        """
        print(" Loading: ", f"{self.folder}/{self._traj}.npy")
        if os.path.exists(f"{self.folder}/{self._traj}.npy"):
            self.load_poses(f"{self.folder}/{self._traj}.npy")
            self._traj += 1
            self._step = 0
            return False
        else:
            return True

    def handle_key_event(self, key):
        if key.keytype == Key.BUTTON:
            if key.number == 3:
                if self._inc_step == 0:
                    self._inc_step = 1
                    print("Play")
                else:
                    self._inc_step = 0
                    print("Pause")
            elif key.number == 0:
                if key.value == 1:
                    self._inc_step = 1
                    print("Forward")
                else:
                    self._inc_step = 0
                    print("Pause")
            elif key.number == 1:
                if key.value == 1:
                    self._inc_step = -1
                    print("Backward")
                else:
                    self._inc_step = 0
                    print("Pause")
        return


if __name__ == "__main__":
    import swift

    env = swift.Swift()
    env.launch(realtime=True, browser="chromium")
    # a = TeleopRecorder(env)
    b = ReplayTeleop(env, "data/scene/exp")

    while True:
        env.step()
        if b.step():
            if b.step_trajectory():
                break
