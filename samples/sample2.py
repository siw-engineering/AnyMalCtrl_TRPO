import UnderdogEnvs
from UnderdogEnvs.robots import Robot2, Robot2Obs
from UnderdogEnvs.envs import CheetahEnv
from UnderdogEnvs.tasks.robot2.task1 import Task1
import gym
import numpy as np
#from gym.wrappers import Monitor
from reward_function_robot2 import lin_vel_reward, ang_vel_reward, yaw_rate_reward, rpy_reward, action_reward, fall_penalty, joint_limit_penalty
import math

def rew_func(action, obs: Robot2Obs, state):
    return obs.joint_positions.sum(), {}


def fric_func(ep_num: int, timestep: int):
    return (np.zeros((12,)), np.zeros((12,)))

def cmd_fun(ep_num: int, timestep: int):
    cmd_direction = 0
    #cmd_in_direction = cmd_direction + timestep
    #print("CMD DIRECTION ", cmd_in_direction)

    horizontal_direction  = ((math.cos(cmd_direction),math.sin(cmd_direction)))
    return (horizontal_direction, cmd_direction)




robot_kwargs = {
    'spawn_orientation': [0, 0.0, 0],
    'spawn_position': [0, 0, 0.3],
    'terrain_type': "Plane"
}

task_kwargs = {
    'max_time_step': 1000,
    'rew_func_list': [ lin_vel_reward, ang_vel_reward, rpy_reward, action_reward, fall_penalty, joint_limit_penalty],
    'fric_func': fric_func,
    'cmd_fun' : cmd_fun
}
if __name__ == '__main__':
    env: CheetahEnv = gym.make('cheetahPosCtrl-v0', robot_cls=Robot2, task_cls=Task1, task_kwargs=task_kwargs,
                               robot_kwargs=robot_kwargs)

    env.reset()
    for _ in range(3):
        for _ in range(400):
            # env.step(env.action_space.sample())
            env.step(np.full((16,), 0.0))
            env.render()

        env.reset()
    # print(f"Cum rew: {env.get_cum_rew()}")
