from typing import Dict, Tuple
import numpy as np
from UnderdogEnvs.robots import Robot1, Robot1Obs
from UnderdogEnvs.tasks import CheetaState
from UnderdogEnvs.tasks.robot1 import CheeFricSpeed


class TeacherDatacollect:
    def __init__(self, robot: Robot1, max_time_step: int = 1000,
                 normalization: bool = False, action_range: float = 15.0, rew_func_list=None, speed_func=None,
                 yaw_rate_func=None, fric_func=None):
        self.base_task = CheeFricSpeed(robot, max_time_step, normalization, action_range, rew_func_list, speed_func,
                                       yaw_rate_func, fric_func)
        self.noncheat_obs_buffer = None
        self.obs = None

    def is_done(self, obs: np.ndarray) -> Tuple[bool, Dict]:
        return self.base_task.is_done(obs)

    def compute_reward(self, obs: Robot1Obs, state: CheetaState, *args) -> Tuple[float, Dict]:
        return self.base_task.compute_reward(obs, state, *args)

    def set_action(self, action: np.ndarray) -> Dict:
        return self.base_task.set_action(action)

    def reset(self):
        return self.base_task.reset()

    def update_commands(self):
        return self.base_task.update_commands()

    def get_observations(self, obs_data_struct: Robot1Obs, *args):
        obs_info = {}
        if self.base_task.ep_step_num == 1:
            self.obs, obs_info = self.base_task.get_observations(obs_data_struct, *args)
            self.noncheat_obs_buffer = np.full((10, 45), self.obs[:45])
        else:
            obs_info['old_cheatobs'] = self.obs
            self.obs, obs_info1 = self.base_task.get_observations(obs_data_struct, *args)
            obs_info = {**obs_info1, **obs_info}
            obs_info['old_nocheatobsbuff'] = self.noncheat_obs_buffer
            obs_info['old_nocheat'] = self.noncheat_obs_buffer[-1]
            self.noncheat_obs_buffer = np.delete(self.noncheat_obs_buffer, 0, axis=0)
            self.noncheat_obs_buffer = np.r_[self.noncheat_obs_buffer, [self.obs[:45]]]

        # Teacher data collection
        return self.obs, obs_info

    def get_observation_space(self):
        # data collection
        return self.base_task.get_observation_space()

    def get_action_space(self):
        return self.base_task.get_action_space()
