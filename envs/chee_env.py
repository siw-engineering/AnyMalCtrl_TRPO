from typing import Tuple, Dict
from gym import Env
import numpy
from UnderdogEnvs.utils import Logger


class CheetahEnv(Env):
    """OpenAI Gym environment for Mini Cheetah, utilises continuous action space."""

    def __init__(self, robot_cls: type, task_cls: type, robot_kwargs: Dict = None, task_kwargs: Dict = None,
                 enable_logging=True):
        if task_kwargs is None:
            task_kwargs = {}
        if robot_kwargs is None:
            robot_kwargs = {}
        self.__robot = robot_cls(**robot_kwargs)
        self.robot = self.__robot
        self.task = task_cls(self.__robot, **task_kwargs)
        self.enable_logging = enable_logging
        self.action_space = self.task.get_action_space()
        self.observation_space = self.task.get_observation_space()
        self.__episode_num = 0
        self.__cumulated_episode_reward = 0
        self.__step_num = 0
        self.__cum_step_num = 0
        self.__last_done_info = None
        if enable_logging:
            self.__logger = Logger(db_file="minicheetah_data.db")
        else:
            self.__logger = None
        # self.reset()
        '''

    def __del__(self):
        if self.enable_logging:
            del self.__logger
            '''

    def step(self, action: numpy.ndarray) -> Tuple[numpy.ndarray, float, bool, dict]:

        # print("AAACTIONSSS ", action)
        if type(action) != numpy.ndarray:
            raise RuntimeError("Action type not numpy array")
        action_info = self.task.set_action(action)
        old_command_obs = self.__robot.get_observations()
        done, done_info = self.task.is_done(old_command_obs)
        state = done_info['state']
        # We compute reward using old commands as observation to be fair
        reward, reward_info = self.task.compute_reward(old_command_obs, state)
        self.task.update_commands()
        new_command_obs = self.__robot.get_observations()
        # Return observation for model with new commands to generate proper action
        obs, obs_info = self.task.get_observations(new_command_obs)
        info: dict = {**reward_info, **done_info, **action_info, **obs_info}
        self.__cumulated_episode_reward += reward
        self.__step_num += 1
        self.__last_done_info = done_info
        log_kwargs = {
            'episode_num': self.__episode_num,
            'step_num': self.__step_num,
            'state': state,
            'action': action,
            'reward': reward,
            'cum_reward': self.__cumulated_episode_reward,
            **info
        }
        if self.enable_logging:
            self.__logger.store(**log_kwargs)

        # print(f"Reward for step {self.__step_num}: {reward}, \t cumulated reward: {self.__cumulated_episode_reward}")
        return obs, reward, done, info

    def get_cum_rew(self):
        return self.__cumulated_episode_reward

    def reset(self):
        if self.__last_done_info is not None:
            print(f'Episode {self.__episode_num: <6}     Reward: {float(self.__cumulated_episode_reward):.9f}     '
                  f'Reason: {self.__last_done_info["state"]:<35}      Timesteps: {self.__step_num:<4}')
        else:
            print(f'Episode {self.__episode_num: <6}     Reward: {self.__cumulated_episode_reward:.9f}     '
                  f'total timesteps: {self.__step_num:<4}')
        self.task.reset()
        self.task.update_commands()
        obs = self.__robot.get_observations()
        np_obs, _ = self.task.get_observations(obs)
        self.__cum_step_num += self.__step_num
        self.__step_num = 0
        self.__last_done_info = None
        self.__episode_num += 1
        self.__cumulated_episode_reward = 0
        return np_obs

    def close(self):
        print('Closing ' + self.__class__.__name__ + ' environment.')

    def render(self, mode='human'):
        self.robot.render()
        pass
