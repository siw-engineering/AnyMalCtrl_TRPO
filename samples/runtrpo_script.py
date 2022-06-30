import os
import pickle
import random
import UnderdogEnvs
from UnderdogEnvs.robots import Robot2, Robot2Obs
from UnderdogEnvs.envs import CheetahEnv
from UnderdogEnvs.tasks.robot2 import task1
import gym
import numpy as np
#from gym.wrappers import Monitor
from reward_function_robot2 import lin_vel_reward, ang_vel_reward, yaw_rate_reward, rpy_reward, action_reward, fall_penalty, joint_limit_penalty
from trpo import trpo


def get_fric_vals():
    fric_file_path = 'fric.pkl'
    if os.path.exists(fric_file_path):
        with open(fric_file_path, 'rb') as f:
            return pickle.load(f)
    else:
        with open(fric_file_path, 'wb') as f:
            fric_list = []
            for i in range(5):
                fric = (np.random.uniform(0.0, 1.0, (12,)), np.random.uniform(1.0, 2.0, (12,)))
                fric_list.append(fric)
            for i in range(5):
                fric = (np.random.uniform(2.0, 3.0, (12,)), np.random.uniform(0.0, 0.8, (12,)))
                fric_list.append(fric)
            pickle.dump(fric_list, f)
            return fric_list


def speed_func(ep_num: int, step_num):
    speed_list = np.array([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [-0.6, 0.0, 0.0]])
    speed = speed_list[ep_num % len(speed_list)]
    return speed

def fric_func(ep_num: int, timestep: int):
    return (np.zeros((12,)), np.zeros((12,)))


robot_kwargs = {
    'spawn_orientation': [0, 0.0, 0],
    'spawn_position': [0, 0, 0.3],
    'terrain_type': "Plane"
}

task_kwargs = {
    'max_time_step': 1000,
    'rew_func_list': [ lin_vel_reward, ang_vel_reward, rpy_reward, action_reward, fall_penalty, joint_limit_penalty],
    'fric_func': fric_func
}


if __name__ == '__main__':
    env: CheetahEnv = gym.make('cheetahPosCtrl-v0', robot_cls=Robot2, task_cls=task1, task_kwargs=task_kwargs,
                               robot_kwargs=robot_kwargs)
    env.reset()

    logger_kwargs = dict(output_dir='data/exp1', exp_name='exp1')

    run_kwargs_base = {'env_fn': env.get_env,
                       'logger_kwargs': logger_kwargs,
                       'epochs': 50,

                       # 'load_model_file': '/home/siwflhc/Desktop/forward_task_docker/aws/210417/addition3lessstrict/data/jostationaryboundary/pyt_save/model.pt',

                       'ac_kwargs': dict(hidden_sizes=[512, 256, 128]),  # 256 256

                       }
    run_kwargs_normal = {'env_fn': env.get_env,
                         'logger_kwargs': logger_kwargs,
                         'epochs': 50,  # 3~5

                         'ac_kwargs': dict(hidden_sizes=[512, 256, 128]),  # 256 256

                         }

    trpo(**run_kwargs)



    print(f"Finished task ")

# td3(env_fn=env_fn, logger_kwargs=logger_kwargs, epochs=100, random_exploration=exploration().get_threshold, num_test_episodes=0, save_checkpoint_path="/opt/run")

# # --- Testing Script ---
# from spinup.utils.test_policy import load_policy_and_env, run_policy
# fpath = 'Mdata/HyQP2'
# env, get_action = load_policy_and_env(fpath,itr ='last',deterministic=False)
# run_policy(env_fn, get_action, max_ep_len=0, num_episodes=10, render=True)
