from Underdog.MITCtrlPy import WalkController
from Underdog.UnderdogCommonPy import StateEstimatorData, UserInput, RobotSystem, RobotSystemOutput
from DartRobots.DartRobotsPy import MiniCheetah, World, MiniCheetahConfig, get_mini_cheetah_urdf, get_ground_urdf
from Underdog.RecoveryCtrlPy import RecoveryStandController, RecoveryStandConfig
from UnderdogEnvs.utils import terrain_generator
import numpy as np
from gym.spaces import Box
import math
from typing import List
from dataclasses import dataclass


@dataclass
class Robot1Obs:
    x_vel_cmd: float
    y_vel_cmd: float
    desired_yaw_rate: float
    foot_contact_forces: np.ndarray
    foot_contact_states: np.ndarray
    foot_contact_normal: np.ndarray
    foot_dist_from_centre: np.ndarray
    foot_pos_world: np.ndarray
    joint_coulomb_friction: np.ndarray
    joint_viscous_friction: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    orientation: np.ndarray
    world_lin_vel: np.ndarray
    world_ang_vel: np.ndarray
    gait_phases: np.ndarray
    desired_positions: np.ndarray
    desired_velocities: np.ndarray
    foot_traj_active: List[bool]


@dataclass
class StandConfig:
    render: bool = True  # Render the standing or not, mostly for debugging purposes
    fall_duration: int = 1000  # Duration in milliseconds to let the robot fall, the higher you spawn the longer this should be
    stand_duration: int = 2100  # Duration to run the stand controller, also need to account for how long it takes to stabilise
    controller_stand_time: float = 1.0  # Time in seconds for the controller to execute the stand trajectory from fold position
    controller_fold_time: float = 1.0  # Time in seconds for the controller to execute the fold trajectory from starting position
    controller_max_torque: float = 16
    controller_update_rate: int = 500  # Stand controller control frequency in Hz
    controller_joint_kp: float = 60  # kp for the joint position controller for following the stand and fold trajectories
    controller_joint_kd: float = 0.6  # kd for the joint position controller for following the stand and fold trajectories
    controller_fold_angles: np.ndarray = np.array([0.0, -1.4, 2.4, 0.0, -1.4, 2.4, 0.0, -1.4, 2.4, 0.0, -1.4, 2.4])  # radians
    controller_stand_angles: np.ndarray = np.array([0.0, -0.8, 1.6, 0.0, -0.8, 1.6, 0.0, -0.8, 1.6, 0.0, -0.8, 1.6])  # radians


class Robot1:
    def __init__(self, spawn_joint_pos=None, spawn_position=None, spawn_orientation=None, stand_config: StandConfig = None, terrain_type=None):
        config = MiniCheetahConfig()
        if spawn_joint_pos is None:
            spawn_joint_pos = np.array([0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6])
        if spawn_position is None:
            spawn_position = np.array([0, 0, 0.3])
        if spawn_orientation is None:
            spawn_orientation = np.array([0, 0, 0])
        if stand_config is None:
            stand_config = StandConfig()
        config.spawn_joint_pos = spawn_joint_pos
        config.spawn_orientation = spawn_orientation
        config.spawn_pos = spawn_position
        config.urdf_path = get_mini_cheetah_urdf()
        self.__stand_config = stand_config
        self.__world = World()
        self.terrainconfig = terrain_generator(terrain_type)
        if terrain_type is None:
            self.__world.set_terrain_urdf(get_ground_urdf())
        else:
            config.spawn_pos = np.array([0, 0, 1.0])
            self.terrainconfig = terrain_generator(terrain_type)
            self.__world.set_terrain(self.terrainconfig)
        self.__robot = MiniCheetah(config)
        self.__world.set_robot(self.__robot)
        self.__controller = WalkController()
        self.__robot_system = RobotSystem()
        self.save_stand_state()

    def add_ball(self, translation, color, radius, name):
        self.__world.add_ball(translation, color, radius, name)

    def save_stand_state(self):
        config = RecoveryStandConfig()
        # config.stand_joint_angles
        config.stand_time = self.__stand_config.controller_stand_time
        config.fold_time = self.__stand_config.controller_fold_time
        config.max_torque = self.__stand_config.controller_max_torque
        config.update_rate = self.__stand_config.controller_update_rate
        config.kp = self.__stand_config.controller_joint_kp
        config.kd = self.__stand_config.controller_joint_kd
        config.fold_joint_angles = self.__stand_config.controller_fold_angles
        config.stand_joint_angles = self.__stand_config.controller_stand_angles
        controller = RecoveryStandController(config)
        self.__world.step(self.__stand_config.fall_duration)
        for i in range(self.__stand_config.stand_duration):
            estimated_state = StateEstimatorData()
            estimated_state.joint_positions = self.__robot.get_joint_positions()
            estimated_state.joint_velocities = self.__robot.get_joint_velocities()
            estimated_state.orientation = self.__robot.get_orientation()
            output = controller.run(estimated_state, i * 1000000)
            self.__robot.set_joint_commands(output)
            self.__world.step(1)
            if self.__stand_config.render and i % 10 == 0:
                self.__world.render()
        self.__robot.set_joint_commands(np.zeros((12,)))
        self.__robot.save_state(1)
        self.__world.reset()

    def set_user_input(self, des_vel_x: float, des_vel_y: float, yaw_rate: float, height: float):
        user_input = UserInput()
        user_input.x_vel_cmd = des_vel_x
        user_input.y_vel_cmd = des_vel_y
        user_input.yaw_turn_rate = yaw_rate
        user_input.height = height
        self.__controller.set_user_input(user_input)

    def set_joint_fric(self, coulomb_fric: np.ndarray, viscous_fric: np.ndarray):
        self.__robot.set_joint_coulomb_friction(coulomb_fric)
        self.__robot.set_joint_viscous_friction(viscous_fric)

    def reset(self):
        self.__world.reset()
        self.__controller.reset()
        self.__robot.load_state(1)

    def set_action(self, compensation_torques: np.ndarray):
        for i in range(10):
            estimated_state = StateEstimatorData()
            estimated_state.joint_positions = self.__robot.get_joint_positions()
            estimated_state.joint_velocities = self.__robot.get_joint_velocities()
            estimated_state.orientation = self.__robot.get_orientation()
            estimated_state.world_lin_vel = self.__robot.get_world_lin_vel()
            estimated_state.world_ang_vel = self.__robot.get_world_ang_vel()
            ori_torques = self.__controller.run(estimated_state)
            final_torques = ori_torques + compensation_torques
            self.__robot.set_joint_commands(final_torques)
            self.__world.step(1)

    def get_observations(self) -> Robot1Obs:
        estimated_state = StateEstimatorData()
        estimated_state.joint_positions = self.__robot.get_joint_positions()
        estimated_state.joint_velocities = self.__robot.get_joint_velocities()
        estimated_state.orientation = self.__robot.get_orientation()
        robot_system_output: RobotSystemOutput = self.__robot_system.compute_state(estimated_state)
        user_input = self.__controller.get_user_input()
        controller_state = self.__controller.get_state()
        return Robot1Obs(user_input.x_vel_cmd, user_input.y_vel_cmd, user_input.yaw_turn_rate,
                         self.__robot.get_foot_contact_forces(),
                         self.__robot.get_foot_contact_states(),
                         self.__robot.get_foot_contact_normals(),
                         robot_system_output.foot_positions,
                         self.__robot.get_foot_positions(),
                         self.__robot.get_joint_coulomb_friction(),
                         self.__robot.get_joint_viscous_friction(),
                         self.__robot.get_joint_positions(),
                         self.__robot.get_joint_velocities(),
                         self.__robot.get_orientation(),
                         self.__robot.get_world_lin_vel(),
                         self.__robot.get_world_ang_vel(),
                         self.__controller.get_gait_phases(),
                         controller_state.foot_traj_output.p_des,
                         controller_state.foot_traj_output.v_des,
                         controller_state.foot_traj_output.active)

    def get_joint_positions(self):
        return self.__robot.get_joint_positions()

    def get_joint_velocities(self):
        return self.__robot.get_joint_velocities()

    def get_orientation(self):
        return self.__robot.get_orientation()

    def get_world_lin_vel(self):
        return self.__robot.get_world_lin_vel()

    def get_world_ang_vel(self):
        return self.__controller.get_world_ang_vel()

    def get_gait_phases(self):
        return self.__controller.get_gait_phases()

    def get_user_inputs(self):
        user_input = self.__controller.get_user_input()
        return user_input.x_vel_cmd, user_input.y_vel_cmd, user_input.yaw_turn_rate, user_input.height

    def get_contact_forces(self):
        return self.__robot.get_foot_contact_forces()

    def get_contact_states(self):
        return self.__robot.get_foot_contact_states()

    def get_coulomb_fric(self):
        return self.__robot.get_joint_coulomb_friction()

    def get_viscous_fric(self):
        return self.__robot.get_joint_viscous_friction()

    def render(self):
        self.__world.render()

    @staticmethod
    def get_observation_space():
        obs_min = np.array([-4, -4, -2.5,
                            -1, -1, -1, -1,  # orientation
                            -3, -3, -3,
                            -4, -4, -2.5,
                            -10, -10, -10,
                            -10, -10, -10,
                            -10, -10, -10,
                            -10, -10, -10,
                            -3.14, -3.14, -3.14,
                            -3.14, -3.14, -3.14,
                            -3.14, -3.14, -3.14,
                            -3.14, -3.14, -3.14,
                            -1, -1, -1, -1,
                            -1, -1, -1, -1])
        obs_max = np.array([4, 4, 2.5,  # user input
                            1, 1, 1, 1,  # orientation
                            3, 3, 3,  # Angular velocity
                            4, 4, 2.5,  # XYZ velocity
                            10, 10, 10,
                            10, 10, 10,
                            10, 10, 10,
                            10, 10, 10,
                            3.14, 3.14, 3.14,
                            3.14, 3.14, 3.14,
                            3.14, 3.14, 3.14,
                            3.14, 3.14, 3.14,
                            1, 1, 1, 1,
                            1, 1, 1, 1])
        return Box(obs_min, obs_max)

    @staticmethod
    def convert_obs_to_numpy(obs: Robot1Obs) -> np.ndarray:
        values = np.zeros((45,))
        values[0] = obs.x_vel_cmd
        values[1] = obs.y_vel_cmd
        values[2] = obs.desired_yaw_rate
        values[3:7] = obs.orientation
        values[7:10] = obs.world_ang_vel
        values[10:13] = obs.world_lin_vel
        values[13:25] = obs.joint_velocities
        values[25:37] = obs.joint_positions
        values[37] = math.sin(obs.gait_phases[0])
        values[38] = math.sin(obs.gait_phases[1])
        values[39] = math.sin(obs.gait_phases[2])
        values[40] = math.sin(obs.gait_phases[3])
        values[41] = math.cos(obs.gait_phases[0])
        values[42] = math.cos(obs.gait_phases[1])
        values[43] = math.cos(obs.gait_phases[2])
        values[44] = math.cos(obs.gait_phases[3])
        return values
