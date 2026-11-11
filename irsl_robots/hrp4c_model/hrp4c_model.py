import os
exec(open('/choreonoid_ws/install/share/irsl_choreonoid/sample/irsl_import.py').read())

class HRP4C(ru.ImportedRobotModel):
    def __init__(self, robot=None, item=True, world=False, **kwargs):
        super().__init__(robot=robot, item=item, world=world, **kwargs)

    def _init_ending(self, **kwargs): ## override
        self.registerEndEffector('larm', ## end-effector
                          'L_WRIST_R', ## tip-link
                          tip_link_to_eef=coordinates(fv(0, 0, -0.08), fv(0, 0.7071067811865476, 0, 0.7071067811865476)),
                          joint_tuples = (('L_SHOULDER_P', 'shoulder-p'),
                                          ('L_SHOULDER_R', 'shoulder-r'),
                                          ('L_SHOULDER_Y', 'shoulder-y'),
                                          ('L_ELBOW_P', 'elbow-p'),
                                          ('L_WRIST_Y', 'wrist-y'),
                                          ('L_WRIST_R', 'wrist-r'),
                                          )
                          )
        self.registerEndEffector('rarm', ## end-effector
                          'R_WRIST_R', ## tip-link
                          tip_link_to_eef=coordinates(fv(0, 0, -0.08), fv(0, 0.7071067811865476, 0, 0.7071067811865476)),
                          joint_tuples = (('R_SHOULDER_P', 'shoulder-p'),
                                          ('R_SHOULDER_R', 'shoulder-r'),
                                          ('R_SHOULDER_Y', 'shoulder-y'),
                                          ('R_ELBOW_P', 'elbow-p'),
                                          ('R_WRIST_Y', 'wrist-y'),
                                          ('R_WRIST_R', 'wrist-r'),
                                          )
                          )
        self.registerEndEffector('lleg', ## end-effector
                                 'L_ANKLE_R', ## tip-link
                                 tip_link_to_eef = coordinates(fv(0, 0, -0.092849)),
                                 joint_tuples = (('L_HIP_Y', 'hip-y'),
                                                 ('L_HIP_R', 'hip-r'),
                                                 ('L_HIP_P', 'hip-p'),
                                                 ('L_KNEE_P' ,'knee-p'),
                                                 ('L_ANKLE_P','ankle-p'),
                                                 ('L_ANKLE_R','ankle-r'),
                                                 )
                                 )
        self.registerEndEffector('rleg', ## end-effector
                                 'R_ANKLE_R', ## tip-link
                                 tip_link_to_eef = coordinates(fv(0, 0, -0.092849)),
                                 joint_tuples = (('R_HIP_Y', 'hip-y'),
                                                 ('R_HIP_R', 'hip-r'),
                                                 ('R_HIP_P', 'hip-p'),
                                                 ('R_KNEE_P' ,'knee-p'),
                                                 ('R_ANKLE_P','ankle-p'),
                                                 ('R_ANKLE_R','ankle-r'),
                                                 )
                                 )
        self.registerNamedPose('default',
                               [0, 0, 0, 0, 0, 0, 0, ## LLEG
                                0, 0, 0, 0, 0, 0, 0, ## RLEG
                                0, 0, 0, ## CHEST
                                0, 0, 0, ## NECK
                                0, 0, 0, 0, 0, 0, ## LARM
                                0, 0, 0, 0, 0, 0, ## RARM
                                ])
        self.registerNamedPose('pose0',
                        angles = [0, 0, -0.5, 1.0, -.5, 0, 0, ## LLEG
                                 0, 0, -0.5, 1.0, -0.5, 0, 0, ## RLEG
                                 0, 0, 0, ## CHEST
                                 0, 0, 0, ## NECK
                                 0, 0, 0, 0, 0, 0, ## LARM
                                 0, 0, 0, 0, 0, 0, ## RARM
                                 ],
                        root_coords = coordinates(fv(0, 0, 0.68)))
        self.setDefaultPose()

### settings of model_file
HRP4C.model_file = iu.parseURL('choreonoid://share/model/HRP4C/HRP4C.body')

### robot_class:
robot_class = HRP4C

### makeRobot(robot=None, **kwargs):
def makeRobot(robot=None, item=True, world=True, **kwargs):
    return robot_class(robot, item=item, world=world, **kwargs)
