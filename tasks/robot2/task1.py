import math
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from gym.spaces import Box
from UnderdogEnvs.robots import Robot2, Robot2Obs
from UnderdogEnvs.tasks import CheetaState
from UnderdogEnvs.utils import quaternion_to_euler
from DartRobots.DartRobotsPy import get_height
import quaternion
from numpy import random

@dataclass
class RobotCmd:
    desired_direction: Tuple[float, float]
    desired_turning_direction: float


class Task1:
    def __init__(self, robot: Robot2, max_time_step: int = 1000, normalization: bool = False,
                 rew_func_list=None,
                 fric_func=None, cmd_func=None, terrain_type=None):
        if rew_func_list is None:
            rew_func_list = []
        self.joint_velocity_buffer =  np.zeros(12,)
        self.joint_velocity_history = np.zeros(24,)

        self.target_foot_buffer =  np.zeros(12,)
        self.target_foot_history =  np.zeros(24,)

        self.des_joint_positoin_buffer =  np.zeros(12,)
        self.joint_position_error_history =  np.zeros(24,)


        self.initial_Step = False
        if fric_func is None:
            def fric_func(episode_num, step_num):
                return (
                    np.zeros((12,), dtype=np.float64),
                    np.zeros((12,), dtype=np.float64))
        if cmd_func is None:
            def cmd_func(episode_num, step_num):
                return RobotCmd((0.0, 0.0), 0.0)
        self.action = np.random.sample(12)*0.6
        self.robot = robot
        self._max_time_step = max_time_step
        self.rew_func_list = rew_func_list
        self.fric_func = fric_func
        self.cmd_func1 = RobotCmd((0.0, 0.0), 0.0)
        self.normalization = normalization
        self.cmd_func = cmd_func
        print(f'-------------------------------Setting task parameters-------------------------------')
        print('max_time_step: %8d               # Maximum time step before stopping the episode' % self._max_time_step)
        print(f'-------------------------------------------------------------------------------------')

        self._max_time_step = max_time_step
        self.__episode_num: int = 0
        self.ep_step_num: int = 0
        self.X_speed, self.input_speed = 0.0, 0.0
        self.lower_limit_count = 0
        self.upper_limit_count = 0
        self.fallen_count = 0
        self.base_frequency = 1.25
        self.ob_space_activate = False

        self.joint_limited_count = 0.0
        self.joint_limited = np.zeros((12,))
        self.height_scan_list = []

    def is_done(self, obs: Robot2Obs) -> Tuple[bool, Dict]:
        failed, state = self.is_failed(obs)
        info_dict = {'state': state}

        if state != CheetaState.Undefined:  # Undefined is basically not reaching any of the failure conditions
            return failed, info_dict

        info_dict['state'] = CheetaState.InProgress
        return False, info_dict

    def compute_reward(self, obs: Robot2Obs, state: CheetaState, *args) -> Tuple[float, Dict]:
        reward_total = 0.0
        reward_info = {}
        for func in self.rew_func_list:
            reward, rew_info = func(self.action, obs, state)
            reward_total = reward_total + reward
            reward_info = {**reward_info, **rew_info}
        print("REWARD " ,reward_total)

        return reward_total, reward_info

    def set_action(self, action: np.ndarray) -> Dict:
        self.robot.set_action(action)
        return {}

    def reset(self):
        self.robot.reset()

    def find_gravity_vec(self, orient):
        gZ_vector = np.array([0, 0, -9.8])
        q1 = np.quaternion(orient[3], orient[0], orient[1], orient[2])
        find_rotation = quaternion.as_rotation_matrix(q1)  # world to robot
        find_transpose = np.transpose(find_rotation)
        gravity_vector = find_transpose.dot(gZ_vector)
        return gravity_vector

    def update_commands(self):
        cmd = self.cmd_func(self.__episode_num, self.ep_step_num)

        self.ep_step_num += 1

    def get_observations(self, obs_data_struct: Robot2Obs, *args):

        np_obs = Robot2.convert_obs_to_numpy(obs_data_struct)
        orientation_quat = np_obs[0:4]

        gravity_vector = self.find_gravity_vec(orientation_quat)

        np_obs = np.append(np_obs, gravity_vector)

        np_obs = np.append(np_obs, self.base_frequency)

        np_obs = np.append(np_obs, self.fric_func(self.__episode_num, self.ep_step_num)[0])
        np_obs = np.append(np_obs, self.fric_func(self.__episode_num, self.ep_step_num)[1])

        np_obs = np.append(np_obs, self.cmd_func1.desired_direction[0])
        np_obs = np.append(np_obs, self.cmd_func1.desired_direction[1])
        np_obs = np.append(np_obs, self.cmd_func1.desired_turning_direction)
        np_obs = np.append(np_obs, obs_data_struct.foot_contact_forces)

        self.height_scan_list = self.height_scan(obs_data_struct.foot_pos_world)
        np_obs = np.append(np_obs, self.height_scan_list)

        if self.normalization:
            normalized = 2 * ((np_obs - self.get_observation_space().low) / (
                    self.get_observation_space().high - self.get_observation_space().low)) - 1
            obs = normalized
        else:
            obs = np_obs

        obs = np.append(obs, obs_data_struct.foot_contact_states)
        obs_space: Box = self.get_observation_space()


        if self.initial_Step == False:
            obs = np.append(obs, self.joint_velocity_history)
            obs = np.append(obs, self.joint_position_error_history)
            obs = np.append(obs, self.target_foot_history)

            self.initial_Step = True
        else:
            temp_swap_velcoity = self.joint_velocity_history[0:12].copy()
            self.joint_velocity_history[0:12] = self.joint_velocity_buffer
            self.joint_velocity_history[12:24] = temp_swap_velcoity


            temp_swap_joint_position  = self.joint_position_error_history[0:12].copy()
            self.joint_position_error_history[0:12] = self.des_joint_positoin_buffer - obs_data_struct.joint_positions
            self.joint_position_error_history[12:24] = temp_swap_joint_position

            temp_swap_target_foot  = self.target_foot_history[0:12].copy()
            self.target_foot_history[0:12] = np.reshape(self.target_foot_buffer ,12)
            self.target_foot_history[12:24] = temp_swap_target_foot

            obs = np.append(obs, self.joint_velocity_history)
            obs = np.append(obs, self.joint_position_error_history)
            obs = np.append(obs, self.target_foot_history)
        self.joint_velocity_buffer = obs_data_struct.joint_velocities
        self.des_joint_positoin_buffer = obs_data_struct.des_joint_pos
        self.target_foot_buffer = obs_data_struct.des_foot_pos

        obs = np.clip(obs, obs_space.low, obs_space.high)

        return obs, {'joint_friction': np.array(self.fric_func(self.__episode_num, self.ep_step_num)),
                     'joint_angles': obs_data_struct.joint_positions}

    def get_observation_space(self):
        robot_obs_space = self.robot.get_observation_space()

        low = np.append(robot_obs_space.low,
                        [0.0] * 3 + [0.0] * 1 + [0.0] * 24 + [0.0] * 2 + [-1] + [-150] * 12 + [0.0] * 36 + [
                            0.0] * 4 + [-10] * 24 + [-1] * 24 + [-0.5] * 24)
        high = np.append(robot_obs_space.high,
                         [0.0, 0.0, -9.8] + [1.25] * 1 + [3.0] * 12 + [2.0] * 12 + [3.14] * 2 + [1] + [150] * 12 + [
                             0.0] * 36 + [1.0] * 4 + [10] * 24 + [1] * 24 + [0.5] * 24)

        print("OBS SIZE", len(low))

        return Box(low, high)

    def get_action_space(self):
        pos_residual_space_max = np.ones((12,)) * 0.02
        freq_offset_space_max = np.ones(4, )

        pos_residual_space_min = pos_residual_space_max * -1
        freq_offset_space_min = freq_offset_space_max * -1
        return Box(np.concatenate((pos_residual_space_min, freq_offset_space_min), axis=0),
                   np.concatenate((pos_residual_space_max, freq_offset_space_max), axis=0), )

    def is_failed(self, obs: Robot2Obs) -> Tuple[bool, CheetaState]:
        [roll, pitch, yaw] = quaternion_to_euler(obs.orientation[3], obs.orientation[0], obs.orientation[1],
                                                 obs.orientation[2])
        info_dict = {'state': CheetaState.Undefined}
        # Check if time step exceeds limits, i.e. timed out
        # Time step starts from 1, that means if we only want to run 2 steps time_step will be 1,2
        if self.ep_step_num >= self._max_time_step:
            return True, CheetaState.Timeout

        if self.ep_step_num > 1:
            if (abs(roll) >= 0.78) or (abs(pitch) >= 0.78):
                # print('Fallen')
                self.fallen_count += 1
                return True, CheetaState.Fallen


        joint_angles = np.array(obs.joint_positions)
        upper_bound = np.array([0.5, 1.5, 2.5,
                                0.5, 1.5, 2.5,
                                0.5, 1.5, 2.5,
                                0.5, 1.5, 2.5])
        lower_bound = np.array([-0.5, -1.5, -1.5,
                                -0.5, -1.5, -1.5,
                                -0.5, -1.5, -1.5,
                                -0.5, -1.5, -1.5])
        min_dist_to_upper_bound = np.amin(upper_bound - joint_angles)
        min_dist_to_lower_bound = np.amin(joint_angles - lower_bound)

        lower_limits_reached = min_dist_to_lower_bound < 0.05
        upper_limits_reached = min_dist_to_upper_bound < 0.05
        self.joint_limited_count = sum(
            (joint_angles - lower_bound) < 0.05)  # for reward calculated per joint which hit lower bound
        self.joint_limited_count += sum(
            (upper_bound - joint_angles) < 0.05)  # for reward calculated per joint which hit higher bound
        self.joint_limited = ((joint_angles - lower_bound) < 0.05) + ((upper_bound - joint_angles) < 0.05)
        if lower_limits_reached or upper_limits_reached:
            info_dict['state'] = CheetaState.ApproachJointLimits
            if lower_limits_reached:
                self.lower_limit_count += 1
                # min_dist_lower_index = numpy.argmin(abs(joint_angles - lower_bound))
                # print(f"Joint with index {min_dist_lower_index} approached lower joint limits, current value: {joint_angles[min_dist_lower_index]}")
            else:
                self.upper_limit_count += 1

            return False, CheetaState.ApproachJointLimits

        # Didn't fail
        return False, CheetaState.Undefined

    def height_scan(self, foot_position):
        x_buffer = []
        y_buffer = []
        height_buffer = []
        for i in range(4):
            angle = 0
            x_buffer.append(foot_position[0][i])
            y_buffer.append(foot_position[1][i])

            while angle < 2 * math.pi - 0.785398:
                angle += 0.785398
                x = foot_position[0][i] + (0.1 * math.cos(angle))
                x_buffer.append(x)
                y = foot_position[1][i] + (0.1 * math.sin(angle))
                y_buffer.append(y)
        for i in range(len(x_buffer)):
            # print("come here1")
            height = get_height(x_buffer[i], y_buffer[i], self.robot.terrainconfig)
            # print("come here")
            height_buffer.append(height)
        return height_buffer
