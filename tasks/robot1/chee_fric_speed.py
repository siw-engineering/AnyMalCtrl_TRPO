from typing import Dict, Tuple
import numpy as np
from gym.spaces import Box
from UnderdogEnvs.robots import Robot1, Robot1Obs
from UnderdogEnvs.tasks import CheetaState
from UnderdogEnvs.utils import quaternion_to_euler
from DartRobots.DartRobotsPy import get_height
import math


class CheeFricSpeed:
    def __init__(self, robot: Robot1, max_time_step: int = 1000,
                 normalization: bool = False, action_range: float = 15.0, rew_func_list=None, speed_func=None,
                 yaw_rate_func=None, fric_func=None):
        if rew_func_list is None:
            rew_func_list = []
        if speed_func is None:
            def speed_func(episode_num, step_num):
                return np.array([0.0, 0.0, 0.0])
        if yaw_rate_func is None:
            def yaw_rate_func(episode_num, step_num):
                return 0.0
        if fric_func is None:
            def fric_func(episode_num, step_num):
                return (
                    np.array([0.41454885, 0.58974709, 1.10617551, 2.46297969, 0.29130383,
                              2.51383472, 0.28829522, 2.9293784, 1.4059536, 2.93028326,
                              1.81453656, 2.21779074]),
                    np.array([0.87720303, 1.97674768, 0.20408962, 0.41775351, 0.32261904,
                              1.30621665, 0.50658321, 0.93262155, 0.48885118, 0.31793917,
                              0.22075028, 1.31265918]))
        self.action = np.zeros((12,))
        self.robot = robot
        self._max_time_step = max_time_step
        self.action_range = action_range
        self.normalization = normalization
        self.rew_func_list = rew_func_list
        self.speed_func = speed_func
        self.yaw_rate_func = yaw_rate_func
        self.fric_func = fric_func
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

        self.joint_limited_count = 0.0
        self.joint_limited = np.zeros((12,))
        self.height_scan_list = []

    def is_done(self, obs: np.ndarray) -> Tuple[bool, Dict]:
        failed, state = self.is_failed(obs)
        info_dict = {'state': state}

        if state != CheetaState.Undefined:  # Undefined is basically not reaching any of the failure conditions
            return failed, info_dict

        info_dict['state'] = CheetaState.InProgress
        return False, info_dict

    def compute_reward(self, obs: Robot1Obs, state: CheetaState, *args) -> Tuple[float, Dict]:
        reward_total = 0.0
        reward_info = {}
        for func in self.rew_func_list:
            reward, rew_info = func(self.action, obs, state, self.height_scan_list)
            reward_total = reward_total + reward
            reward_info = {**reward_info, **rew_info}

        return reward_total, reward_info

    def set_action(self, action: np.ndarray) -> Dict:
        self.action = action  # Store for reward computation
        true_action = self.action * self.action_range
        self.robot.set_action(true_action)
        # Increment episode num and set desired vel only after current step finish executing
        return {}

    def reset(self):
        if self.fallen_count > 0:
            self.robot.reset()
            self.fallen_count = 0

        if self.lower_limit_count > 0 or self.upper_limit_count > 0:
            self.robot.reset()
            self.lower_limit_count = 0
            self.upper_limit_count = 0
            # print('limit reset')
        # if self.previous_z<0.1:
        #    self.robot.reset()
        # obs = self.robot.get_observations()
        # self.previous_action = numpy.zeros((12,))
        # self.previous_obs = numpy.zeros((46,))
        self.__episode_num += 1
        self.ep_step_num = 0
        self.joint_limited_count = 0.0
        self.joint_limited = np.zeros((12,))
        self.robot.set_joint_fric(*self.fric_func(self.__episode_num, self.ep_step_num))

    def update_commands(self):
        desired_vel = self.speed_func(self.__episode_num, self.ep_step_num)
        yaw_turn_rate = self.yaw_rate_func(self.__episode_num, self.ep_step_num)
        self.robot.set_user_input(desired_vel[0], desired_vel[1], yaw_turn_rate, 0.29)
        self.ep_step_num += 1

    def get_observations(self, obs_data_struct: Robot1Obs, *args):
        np_obs = Robot1.convert_obs_to_numpy(obs_data_struct)
        np_obs = np.append(np_obs, self.fric_func(self.__episode_num, self.ep_step_num)[0])
        np_obs = np.append(np_obs, self.fric_func(self.__episode_num, self.ep_step_num)[1])
        np_obs = np.append(np_obs, obs_data_struct.foot_contact_forces)
        np_obs = np.append(np_obs, obs_data_struct.foot_contact_states)
        np_obs = np.append(np_obs, obs_data_struct.foot_contact_normal)
        self.height_scan_list = self.height_scan(obs_data_struct.foot_pos_world)
        np_obs = np.append(np_obs, self.height_scan_list)
        if self.normalization == True:
            normalized = 2 * ((np_obs - self.get_observation_space().low) / (
                    self.get_observation_space().high - self.get_observation_space().low)) - 1
            obs = normalized
        else:
            obs = np_obs
        obs[81:85] = obs_data_struct.foot_contact_states
        obs_space: Box = self.get_observation_space()
        obs = np.clip(obs, obs_space.low, obs_space.high)

        return obs, {'joint_friction': np.array(self.fric_func(self.__episode_num, self.ep_step_num)), 'height_scan': np.array(self.height_scan_list)}

    def get_observation_space(self):
        robot_obs_space = self.robot.get_observation_space()
        # TODO: friction range and friction func must match
        low = np.append(robot_obs_space.low, [0.0] * 24 + [-150] * 12 + [0.0] * 4 + [-1.0] * 12 + [0.0] * 36)
        high = np.append(robot_obs_space.high, [3.0] * 12 + [2.0] * 12 + [150] * 12 + [1.0] * 4 + [1.0] * 12 + [0.3] * 36)

        print("LOW ", low, " high ", high)
        return Box(low, high)

    def get_action_space(self):
        return Box(-1, 1, (12,))

    def is_failed(self, obs: Robot1Obs) -> Tuple[bool, CheetaState]:
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

        # current_x = obs.estimated_state.world_linear_velocity
        # current_y = obs.pose.position.y
        # current_coords_2d = numpy.array([current_x, current_y])

        # Check that joint values are not approaching limits
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
        # self.accepted_dist_to_bounds is basically how close to the joint limits can the joints go,
        # i.e. limit of 1.57 with accepted dist of 0.1, then the joint can only go until 1.47
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
                # min_dist_upper_index = numpy.argmin(abs(upper_bound - joint_angles))
                # print(f"Joint with index {min_dist_upper_index} approached upper joint limits, current value: {joint_angles[min_dist_upper_index]}")
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
            # height = get_height(foot_position[0][i], foot_position[1][i], self.robot.terrainconfig)
            # self.robot.add_ball(np.array([foot_position[0][i], foot_position[1][i], height]), np.array([1.0,0.0,0.0]), 0.02, f"marker_{i}")
            while angle < 2 * math.pi - 0.785398:
                angle += 0.785398
                x = foot_position[0][i] + (0.1 * math.cos(angle))
                x_buffer.append(x)
                y = foot_position[1][i] + (0.1 * math.sin(angle))
                y_buffer.append(y)
        for i in range(len(x_buffer)):
            print("VOME")
            height = get_height(x_buffer[i], y_buffer[i], self.robot.terrainconfig)
            print("DGSAI")
            height_buffer.append(height)
        return height_buffer
