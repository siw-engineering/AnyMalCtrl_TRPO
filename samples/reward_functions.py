from scipy.spatial.transform import Rotation
from collections import deque
import numpy
import math
from UnderdogEnvs.utils import quaternion_to_euler
from UnderdogEnvs.tasks import CheetaState


class LinVelReward:
    def __init__(self):
        self.x_vel_buffer = deque(maxlen=10)
        self.y_vel_buffer = deque(maxlen=10)

    def __call__(self, action, obs, state):
        rot: Rotation = Rotation.from_quat(obs.orientation)
        world2base_rot_mat = rot.as_matrix().transpose()
        velocity = numpy.matmul(world2base_rot_mat, obs.world_lin_vel)
        self.x_vel_buffer.append(abs(obs.x_vel_cmd - velocity[0]))
        self.y_vel_buffer.append(abs(obs.y_vel_cmd - velocity[1]))
        average_x_vel = sum(self.x_vel_buffer) / len(self.x_vel_buffer)
        average_y_vel = sum(self.y_vel_buffer) / len(self.y_vel_buffer)
        z_vel_penalty = -2 * abs(velocity[2])
        reward = (3 - 3 * (abs(average_x_vel)) - 3 * (abs(average_y_vel))) + z_vel_penalty
        rew_info = {"current_x_vel": velocity[0], "current_y_vel": velocity[1], "linear_velocity_reward": reward,
                    "z_vel_penalty": z_vel_penalty}
        return reward, rew_info


def ang_vel_reward(action, obs, state):
    rot: Rotation = Rotation.from_quat(obs.orientation)
    world2base_rot_mat = rot.as_matrix().transpose()
    angular_velocity = numpy.matmul(world2base_rot_mat, obs.world_ang_vel)
    reward = 1 - (abs(angular_velocity[0]) ** 0.4 + abs(angular_velocity[1]) ** 0.4 + abs(angular_velocity[2]) ** 0.4)
    if reward < -2.0:
        reward = -2.0
    rew_info = {"angular_velocity_reward": reward}
    return reward, rew_info


def rpy_reward(action, obs, state):
    [roll, pitch, yaw] = quaternion_to_euler(obs.orientation[3], obs.orientation[0], obs.orientation[1],
                                             obs.orientation[2])

    allowable_yaw_deg = 0.1
    allowable_yaw_rad = allowable_yaw_deg * math.pi / 180
    # Note: This logic doesn't work well when yaw is beyond 90 degrees
    if abs(yaw) > allowable_yaw_rad:
        yaw_penalty_factor = math.cos(yaw)
    else:
        yaw_penalty_factor = 1.0

    allowable_roll_deg = 0.1
    allowable_roll_rad = allowable_roll_deg * math.pi / 180
    if abs(roll) > allowable_roll_rad:
        roll_penalty_factor = math.cos(roll)
    else:
        roll_penalty_factor = 1.0

    allowable_pitch_deg = 5  # default 0.1
    allowable_pitch_rad = allowable_pitch_deg * math.pi / 180
    # Note: This logic doesn't work well when yaw is beyond 90 degrees, because roll and pitch will flip sign and yaw will still be less than 90
    if abs(pitch) > allowable_pitch_rad:
        pitch_penalty_factor = math.cos(pitch)
    else:
        pitch_penalty_factor = 1.0

    yaw_penalty_factor = yaw_penalty_factor ** 10
    roll_penalty_factor = roll_penalty_factor ** 10
    pitch_penalty_factor = pitch_penalty_factor ** 10
    reward = 3 * (pitch_penalty_factor * yaw_penalty_factor * roll_penalty_factor)
    rew_info = {'roll_penalty_factor': roll_penalty_factor, 'pitch_penalty_factor': pitch_penalty_factor,
                'yaw_penalty_factor': yaw_penalty_factor, 'rpy_reward': reward}
    return reward, rew_info


def action_reward(action, obs, state):
    action_penalty_LF = 0.5 - (float(abs(action[0]) + abs(action[1]) + abs(action[2]))) ** 0.8
    action_penalty_RF = 0.5 - (float(abs(action[3]) + abs(action[4]) + abs(action[5]))) ** 0.8
    action_penalty_LH = 0.5 - (float(abs(action[6]) + abs(action[7]) + abs(action[8]))) ** 0.8
    action_penalty_RH = 0.5 - (float(abs(action[9]) + abs(action[10]) + abs(action[11]))) ** 0.8
    reward = 1.5 * (
            action_penalty_LF + action_penalty_RF + action_penalty_LH + action_penalty_RH)
    foot_pos_world = numpy.array(obs.foot_pos_world[2].tolist())
    rew_info = {"foot_pos_world": foot_pos_world, "action_penalty_LF": action_penalty_LF, "action_penalty_RF": action_penalty_RF,
                "action_penalty_LH": action_penalty_LH, "action_penalty_RH": action_penalty_RH, "action_reward": reward}
    return reward, rew_info


def fall_penalty(action, obs, state):
    penalty = 0
    if state == CheetaState.Fallen:
        penalty = -10
    rew_info = {'fallen_penalty': penalty}
    return penalty, rew_info


def joint_limit_penalty(action, obs, state):
    penalty = 0.0
    if state == CheetaState.ApproachJointLimits:  # -2
        upper_bound = numpy.array([0.5, 1.5, 2.5,
                                   0.5, 1.5, 2.5,
                                   0.5, 1.5, 2.5,
                                   0.5, 1.5, 2.5])
        lower_bound = numpy.array([-0.5, -1.5, -1.5,
                                   -0.5, -1.5, -1.5,
                                   -0.5, -1.5, -1.5,
                                   -0.5, -1.5, -1.5])
        joint_limited_count = sum(
            (obs.joint_positions - lower_bound) < 0.05)  # for reward calculated per joint which hit lower bound
        joint_limited_count += sum(
            (upper_bound - obs.joint_positions) < 0.05)  # for reward calculated per joint which hit higher bound
        penalty = ((25 / 12) * joint_limited_count) * -1
    rew_info = {'joint_limit_penalty': penalty}
    return penalty, rew_info
