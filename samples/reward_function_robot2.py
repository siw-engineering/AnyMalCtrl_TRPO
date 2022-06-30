from scipy.spatial.transform import Rotation
from collections import deque
import numpy
import math
from UnderdogEnvs.utils import quaternion_to_euler
from UnderdogEnvs.tasks.common import CheetaState


def lin_vel_reward(action, obs, state):
    rot: Rotation = Rotation.from_quat(obs.orientation)
    world2base_rot_mat = rot.as_matrix().transpose()
    velocity = numpy.matmul(world2base_rot_mat, obs.world_lin_vel)
    vel_reward_cal = 0.0

    # print("LINER VEL REWARD ", velocity)
    newVel = math.hypot(velocity[0], velocity[1])

    # print("NEW VELOCITY" , newVel)

    if newVel == 0.0 :
        vel_reward_cal = 0.0
    else:
        if newVel < 0.6:
            vel_reward_cal = math.exp(-2.0*math.pow((newVel-0.6),2))
        elif newVel >= 0.6 and newVel <= -0.6:
            vel_reward_cal = 1.0

    reward = vel_reward_cal

    rew_info = {"current_x_vel": velocity[0], "current_y_vel": velocity[1],
                "linear_velocity_reward": reward}
    return reward, rew_info


def ang_vel_reward(action, obs, state):
    rot: Rotation = Rotation.from_quat(obs.orientation)
    world2base_rot_mat = rot.as_matrix().transpose()
    angular_velocity = numpy.matmul(world2base_rot_mat, obs.world_ang_vel)
    reward = 0.0
    if angular_velocity[2]<0.6:
        reward  = math.exp(-1.5*math.pow((angular_velocity[2]-0.6),2))
    elif angular_velocity[2]>= 0.6 and angular_velocity[2]<=-0.6:
        reward = 1.0
    rew_info = {"angular_velocity_reward": reward}
    return reward, rew_info

def yaw_rate_reward(action, obs, state):
    rot: Rotation = Rotation.from_quat(obs.orientation)
    world2base_rot_mat = rot.as_matrix().transpose()
    angular_velocity = numpy.matmul(world2base_rot_mat, obs.world_ang_vel)
    reward = 0.0
    if abs(angular_velocity[2]) >= 0.8:
        reward = -4
    else:
        if obs.desired_yaw_rate == 0.0:
            reward = 1.0 - 5 * (abs(obs.desired_yaw_rate - angular_velocity[2]))
        else:
            if obs.desired_yaw_rate == 0.6:
                yawrate_diff = angular_velocity[2] - obs.desired_yaw_rate
            if obs.desired_yaw_rate == -0.6:
                yawrate_diff = obs.desired_yaw_rate - angular_velocity[2]
            if yawrate_diff >= 0:
                reward = 1.0
            else:
                reward = 1.0 - 5 * (abs(yawrate_diff))
    rew_info = {"angular_velocity": angular_velocity, "yaw_turn_rate_reward": reward}
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

    allowable_roll_deg = 5
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

    pitch_penalty_factor = pitch_penalty_factor ** 10
    roll_penalty_factor = roll_penalty_factor ** 10
    reward = 2 * (pitch_penalty_factor * roll_penalty_factor)
    rew_info = {'roll_penalty_factor': roll_penalty_factor, 'pitch_penalty_factor': pitch_penalty_factor,
                'rp_reward': reward}
    return reward, rew_info


def action_reward(action, obs, state):
    action_penalty_LF = 0.5 - (float(abs(action[0]) + abs(action[1]) + abs(action[2]))) ** 0.8
    action_penalty_RF = 0.5 - (float(abs(action[3]) + abs(action[4]) + abs(action[5]))) ** 0.8
    action_penalty_LH = 0.5 - (float(abs(action[6]) + abs(action[7]) + abs(action[8]))) ** 0.8
    action_penalty_RH = 0.5 - (float(abs(action[9]) + abs(action[10]) + abs(action[11]))) ** 0.8
    reward = 1.5 * (
            action_penalty_LF + action_penalty_RF + action_penalty_LH + action_penalty_RH)
    rew_info = {"action_penalty_LF": action_penalty_LF, "action_penalty_RF": action_penalty_RF,
                "action_penalty_LH": action_penalty_LH, "action_penalty_RH": action_penalty_RH, "action_reward": reward}
    return reward, rew_info



def fall_penalty(action, obs, state):
    penalty = 0
    if state == CheetaState.Fallen:
        penalty = -20
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
        for i in range(12):
            if (obs.joint_positions[i] - lower_bound[i]) < 0.1:
                penalty += (1 - 5*(obs.joint_positions[i] - lower_bound[i]))
            elif (upper_bound[i] - obs.joint_positions[i]) < 0.1:
                penalty += (1 - 5*(upper_bound[i] - obs.joint_positions[i]))
            else:
                penalty += 0
        penalty *= -1.5
    rew_info = {'joint_limit_penalty': penalty}
    return penalty, rew_info