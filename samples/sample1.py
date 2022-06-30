import UnderdogEnvs
from UnderdogEnvs.robots import Robot1
from UnderdogEnvs.envs import CheetahEnv
from UnderdogEnvs.tasks.robot1 import CheeFricSpeed
import gym
import numpy as np
from reward_functions import LinVelReward, ang_vel_reward, rpy_reward, action_reward, fall_penalty, \
    joint_limit_penalty


def speed_func(ep_num: int, timestep: int):
    return [1.0, 0.0, 0.0]


def fric_func(ep_num: int, timestep: int):
    return (np.zeros((12,)), np.zeros((12,)))


robot_kwargs = {
    'spawn_orientation': [0, 0.0, 0],
    'spawn_position': [0, 0, 0.3]
}

task_kwargs = {
    'max_time_step': 1000,
    'normalization': True,
    'action_range': 15.0,
    'rew_func_list': [LinVelReward(), ang_vel_reward, rpy_reward, action_reward, fall_penalty,
                      joint_limit_penalty],
    'speed_func': speed_func,
    'fric_func': fric_func
}
if __name__ == '__main__':
    env: CheetahEnv = gym.make('cheetahAssist-v0', task_cls=CheeFricSpeed, task_kwargs=task_kwargs,
                               robot_kwargs=robot_kwargs)
    env.reset()
    for _ in range(1000):
        env.step(np.full((12,), 0.0))
        env.render()
    print(f"Cum rew: {env.get_cum_rew()}")
