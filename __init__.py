from gym.envs.registration import register
from UnderdogEnvs.tasks.robot1 import CheeFricSpeed
from UnderdogEnvs.tasks.robot2 import Task1
from UnderdogEnvs.robots import Robot1, Robot2

register(
    id='cheetahAssist-v0',  # Continuous action space
    entry_point='UnderdogEnvs.envs:CheetahEnv',
    kwargs={'task_cls': CheeFricSpeed,
            'robot_cls': Robot1,
            }
)

register(
    id='cheetahPosCtrl-v0',  # Continuous action space
    entry_point='UnderdogEnvs.envs:CheetahEnv',
    kwargs={'task_cls': Task1,
            'robot_cls': Robot2,
            }
)
