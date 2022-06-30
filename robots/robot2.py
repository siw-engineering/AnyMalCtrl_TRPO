from Underdog.AnymalCtrlPy import MainController
from Underdog.RecoveryCtrlPy import RecoveryStandController, RecoveryStandConfig
from Underdog.UnderdogCommonPy import StateEstimatorData, UserInput, RobotSystem, RobotSystemOutput
from DartRobots.DartRobotsPy import MiniCheetah, World, MiniCheetahConfig, \
    get_mini_cheetah_urdf, get_ground_urdf
import numpy as np

from UnderdogEnvs.utils import terrain_generator
from gym.spaces import Box
from dataclasses import dataclass
import math


# Note that foot friction coeffs represent the coulomb friction coefficient of each foot and ground
# Only friction coefficient of foot is modified at any point in time, which during contact will be the coefficient used
# Does not zero out during non-contact, should be done by user as zeroing out here will have loss of information
@dataclass
class Robot2Obs:
    # x_vel_cmd: float
    # y_vel_cmd: float
    # desired_yaw_rate: float
    foot_contact_forces: np.ndarray
    foot_contact_states: np.ndarray
    foot_friction_coeffs: np.ndarray
    foot_dist: np.ndarray
    foot_pos_world: np.ndarray
    joint_coulomb_friction: np.ndarray
    joint_viscous_friction: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    orientation: np.ndarray
    world_lin_vel: np.ndarray
    world_ang_vel: np.ndarray
    gait_phases: np.ndarray
    gait_frequencies: np.ndarray
    des_joint_pos: np.ndarray
    des_foot_pos:np.array


@dataclass
class StandConfig:
    render: bool = True  # Render the standing or not, mostly for debugging purposes
    fall_duration: int = 400  # Duration in milliseconds to let the robot fall, the higher you spawn the longer this should be
    stand_duration: int = 2100  # Duration to run the stand controller, also need to account for how long it takes to stabilise
    controller_stand_time: float = 1.0  # Time in seconds for the controller to execute the stand trajectory from fold position
    controller_fold_time: float = 1.0  # Time in seconds for the controller to execute the fold trajectory from starting position
    controller_max_torque: float = 16
    controller_update_rate: int = 500  # Stand controller control frequency in Hz
    controller_joint_kp: float = 60  # kp for the joint position controller for following the stand and fold trajectories
    controller_joint_kd: float = 0.6  # kd for the joint position controller for following the stand and fold trajectories
    controller_fold_angles: np.ndarray = np.array(
        [0.0, -1.4, 2.4, 0.0, -1.4, 2.4, 0.0, -1.4, 2.4, 0.0, -1.4, 2.4])  # radians
    controller_stand_angles: np.ndarray = np.array(
        [0.0, -0.8, 1.6, 0.0, -0.8, 1.6, 0.0, -0.8, 1.6, 0.0, -0.8, 1.6])  # radians


class Robot2:
    def __init__(self, spawn_joint_pos: np.ndarray = None, spawn_position: np.ndarray = None,
                 spawn_orientation: np.ndarray = None, stand_config: StandConfig = None, terrain_type=None):
        config = MiniCheetahConfig()
        if spawn_joint_pos is None:
            spawn_joint_pos = np.array([0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6])
        if spawn_position is None:
            spawn_position = np.array([0, 0, 0.6])
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
        self.__robot = MiniCheetah(config)

        if terrain_type is None:
            self.__world.set_terrain_urdf(get_ground_urdf())
        else:
            config.spawn_pos = np.array([0, 0, 1.0])
            self.terrainconfig = terrain_generator(terrain_type)
            self.__world.set_terrain(self.terrainconfig)

        #self.__world.set_terrain_urdf(get_ground_urdf())
        self.__world.set_robot(self.__robot)
        #self.terrianconfig = terrain_generator()
        #self.terrainconfig = terrain_generator(terrain_type)
        self.__robot_system = RobotSystem()
        # self.__robot.save_state(0)
        self.__controller = MainController()
        self.save_stand_state()
        ## Uncomment the code below for visualisation of desired foot position tracking
        self.ballName1 = self.__world.add_ball(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.02, "ball1")
        self.ballName2 = self.__world.add_ball(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.02, "ball1")

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

        '''

    def set_user_input(self, des_vel_x: float, des_vel_y: float, yaw_rate: float, height: float):
        user_input = UserInput()
        user_input.x_vel_cmd = des_vel_x
        user_input.y_vel_cmd = des_vel_y
        user_input.yaw_turn_rate = yaw_rate
        user_input.height = height
        self.__controller.set_user_input(user_input)
        
        '''

    def set_joint_fric(self, coulomb_fric: np.ndarray, viscous_fric: np.ndarray):
        self.__robot.set_joint_coulomb_friction(coulomb_fric)
        self.__robot.set_joint_viscous_friction(viscous_fric)

    def reset(self):
        self.__robot.set_joint_commands(np.zeros((12,)))
        self.__world.reset()
        self.__controller.reset()
        self.__robot.load_state(1)

    def set_action(self, action: np.ndarray):
        pos_residuals = action[0:12].reshape(3, 4)
        #print("GETTTING ACGTIONB ", pos_residuals)
        freq_offsets = action[12:16]
        #print("GETTTING freeq ", freq_offsets)
        for i in range(10):
            estimated_state = StateEstimatorData()
            estimated_state.joint_positions = self.__robot.get_joint_positions()
            estimated_state.joint_velocities = self.__robot.get_joint_velocities()
            estimated_state.orientation = self.__robot.get_orientation()
            estimated_state.world_lin_vel = self.__robot.get_world_lin_vel()
            estimated_state.world_ang_vel = self.__robot.get_world_ang_vel()
            torques = self.__controller.run(estimated_state, pos_residuals, freq_offsets, 1)
            self.__robot.set_joint_commands(torques)
            self.__world.step(1)
        ## Uncomment the following code for visualisation of foot desired position
        des_foot_pos_robot = self.__controller.get_des_foot_pos()
        body_pos = self.__robot.get_body_pos()
        foot_pos0 = des_foot_pos_robot[:, 0]
        foot_pos2 = des_foot_pos_robot[:, 1]
        pos = foot_pos0 + self.__robot.get_body_pos()
        self.__world.set_ball_translation(self.ballName1, pos)

        pos2 = foot_pos2 + self.__robot.get_body_pos()
        self.__world.set_ball_translation(self.ballName2, pos2)

    def get_observations(self) -> Robot2Obs:
        estimated_state = StateEstimatorData()
        estimated_state.joint_positions = self.__robot.get_joint_positions()
        estimated_state.joint_velocities = self.__robot.get_joint_velocities()
        estimated_state.orientation = self.__robot.get_orientation()
        robot_system_output: RobotSystemOutput = self.__robot_system.compute_state(estimated_state)
        return Robot2Obs(self.__robot.get_foot_contact_forces(),
                         self.__robot.get_foot_contact_states(),
                         self.__robot.get_foot_friction(),
                         self.__robot.get_foot_positions(),
                         robot_system_output.foot_positions,
                         self.__robot.get_joint_coulomb_friction(),
                         self.__robot.get_joint_viscous_friction(),
                         self.__robot.get_joint_positions(),
                         self.__robot.get_joint_velocities(),
                         self.__robot.get_orientation(),
                         self.__robot.get_world_lin_vel(),
                         self.__robot.get_world_ang_vel(),
                         self.__controller.get_gait_phases(),
                         self.__controller.get_gait_frequencies(),
                         self.__controller.get_des_joint_pos(),
                         self.__controller.get_des_foot_pos())

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

    def get_gait_frequencies(self):
        return self.__controller.get_gait_frequencies()

    def get_contact_forces(self):
        return self.__robot.get_foot_contact_forces()

    def get_contact_states(self):
        return self.__robot.get_foot_contact_states()

    def get_coulomb_fric(self):
        return self.__robot.get_joint_coulomb_friction()

    def get_viscous_fric(self):
        return self.__robot.get_joint_viscous_friction()

    def get_foot_fric(self):
        return self.__robot.get_foot_friction()

    def render(self):
        self.__world.render()

    @staticmethod
    def get_observation_space():
        # TODO: Finalise observations and hence observation space
        obs_min = np.array([#-4, -4, -2.5,
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
                            -1, -1, -1, -1, 0, 0, 0, 0])
        obs_max = np.array([#4, 4, 2.5,  # user input
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
                            1, 1, 1, 1, 1.25, 1.25, 1.25, 1.25
                            ])
        return Box(obs_min, obs_max)

    @staticmethod
    def convert_obs_to_numpy(obs: Robot2Obs) -> np.ndarray:
        values = np.zeros((46,))

        # values[0] = math.sin(obs.desired_yaw_rate)
        # values[1] = math.sin(obs.desired_yaw_rate)
        # values[2] = obs.desired_yaw_rate
        values[0:4] = obs.orientation
        values[4:7] = obs.world_ang_vel
        values[7:10] = obs.world_lin_vel
        values[10:22] = obs.joint_velocities
        values[22:34] = obs.joint_positions
        values[34] = math.sin(obs.gait_phases[0])
        values[35] = math.sin(obs.gait_phases[1])
        values[36] = math.sin(obs.gait_phases[2])
        values[37] = math.sin(obs.gait_phases[3])
        values[38] = math.cos(obs.gait_phases[0])
        values[39] = math.cos(obs.gait_phases[1])
        values[40] = math.cos(obs.gait_phases[2])
        values[41] = math.cos(obs.gait_phases[3])
        values[42] = obs.gait_frequencies[0]
        values[43] = obs.gait_frequencies[1]
        values[44] = obs.gait_frequencies[2]
        values[45] = obs.gait_frequencies[3]

        return values
