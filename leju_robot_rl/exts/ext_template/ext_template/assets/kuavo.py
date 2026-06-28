import os

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

from ext_template.actuators import DelayedPDActuatorCfg_S42
from ext_template.actuators import DelayedPDActuatorCfg_S49
from ext_template.assets import ISAAC_ASSET_DIR


def resolve_s49_urdf_path() -> str:
    """Return Isaac-ready S49 26-DoF URDF for RL training."""
    repo_root = os.path.abspath(os.path.join(ISAAC_ASSET_DIR, "../../../../.."))
    candidates = [
        os.environ.get("KUAVO_S49_URDF", ""),
        os.path.join(ISAAC_ASSET_DIR, "Robots/Kuavo/s49_train/urdf/biped_s49_26dof_lite.urdf"),
        os.path.join(ISAAC_ASSET_DIR, "Robots/Kuavo/s49_train/urdf/biped_s49_26dof.urdf"),
        os.path.join(ISAAC_ASSET_DIR, "Robots/Kuavo/biped_s49/urdf/biped_s49_26dof.urdf"),
        os.path.join(
            repo_root,
            "leju_robot_rl/exts/ext_template/ext_template/assets/Robots/Kuavo/s49_train/urdf/biped_s49_26dof.urdf",
        ),
        os.path.join(ISAAC_ASSET_DIR, "Robots/Kuavo/s49_train/urdf/biped_s49_rl.urdf"),
        os.path.join(ISAAC_ASSET_DIR, "Robots/Kuavo/biped_s49/urdf/biped_s49_rl.urdf"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "S49 URDF not found. Run: bash scripts/tools/setup_s49_training_assets.sh"
    )


Kuavos46_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # 修改 kuavo.py 第 12 行
        usd_path="/home/lwy/leju_robot_rl/exts/ext_template/ext_template/assets/Robots/Kuavo/biped_s46.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        joint_pos={
            "leg_[l,r]1_joint": 0.0,
            "leg_[l,r]2_joint": 0.0,
            "leg_[l,r]3_joint": -0.27,
            "leg_[l,r]4_joint": 0.52,
            "leg_[l,r]5_joint": -0.3,
            "leg_[l,r]6_joint": 0.0,
            "zarm_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "motor": DelayedPDActuatorCfg_S42(
            joint_names_expr=[
                "leg_.*",
                "zarm_.*_joint"
            ],
            effort_limit={
                "leg_[l,r]1_joint": 180.0,
                "leg_[l,r]2_joint": 100.0,
                "leg_[l,r]3_joint": 100.0,
                "leg_[l,r]4_joint": 180.0,
                "leg_[l,r]5_joint": 72.0,
                "leg_[l,r]6_joint": 36.0,
                "zarm_[l,r]1_joint": 100.0,
                "zarm_[l,r]2_joint": 50.0,
                "zarm_[l,r]3_joint": 36.0,
                "zarm_[l,r]4_joint": 50.0,
                "zarm_[l,r]5_joint": 12.0,
                "zarm_[l,r]6_joint": 12.0,
                "zarm_[l,r]7_joint": 12.0,
            },
            velocity_limit={
                "leg_[l,r]1_joint": 14.0,
                "leg_[l,r]2_joint": 23.0,
                "leg_[l,r]3_joint": 23.0,
                "leg_[l,r]4_joint": 14.0,
                "leg_[l,r]5_joint": 10.0,
                "leg_[l,r]6_joint": 10.0,

                "zarm_[l,r]1_joint": 23.0,
                "zarm_[l,r]2_joint": 23.0,
                "zarm_[l,r]3_joint": 23.0,
                "zarm_[l,r]4_joint": 23.0,
                "zarm_[l,r]5_joint": 23.0,
                "zarm_[l,r]6_joint": 23.0,
                "zarm_[l,r]7_joint": 23.0,
            },
            stiffness={
                "leg_[l,r]1_joint": 100.0,
                "leg_[l,r]2_joint": 100.0,
                "leg_[l,r]3_joint": 100.0,
                "leg_[l,r]4_joint": 150.0,
                "leg_[l,r]5_joint": 40.0,
                "leg_[l,r]6_joint": 40.0,
                "zarm_[l,r]1_joint": 30.0,
                "zarm_[l,r]2_joint": 30.0,
                "zarm_[l,r]3_joint": 30.0,
                "zarm_[l,r]4_joint": 20.0,
                "zarm_[l,r]5_joint": 10.0,
                "zarm_[l,r]6_joint": 10.0,
                "zarm_[l,r]7_joint": 10.0,
            },
            damping={
                "leg_[l,r]1_joint": 4.0,
                "leg_[l,r]2_joint": 4.0,
                "leg_[l,r]3_joint": 4.0,
                "leg_[l,r]4_joint": 8.0,
                "leg_[l,r]5_joint": 4.0,
                "leg_[l,r]6_joint": 4.0,
                "zarm_[l,r]1_joint": 3.0,
                "zarm_[l,r]2_joint": 3.0,
                "zarm_[l,r]3_joint": 3.0,
                "zarm_[l,r]4_joint": 3.0,
                "zarm_[l,r]5_joint": 3.0,
                "zarm_[l,r]6_joint": 3.0,
                "zarm_[l,r]7_joint": 3.0,
            },
            armature={
                "leg_[l,r]1_joint": 0.05,
                "leg_[l,r]2_joint": 0.025,
                "leg_[l,r]3_joint": 0.025,
                "leg_[l,r]4_joint": 0.05,
                "leg_[l,r]5_joint": 0.05,
                "leg_[l,r]6_joint": 0.05,
                "zarm_[l,r]1_joint": 0.025,
                "zarm_[l,r]2_joint": 0.02,
                "zarm_[l,r]3_joint": 0.02,
                "zarm_[l,r]4_joint": 0.02,
                "zarm_[l,r]5_joint": 0.01,
                "zarm_[l,r]6_joint": 0.01,
                "zarm_[l,r]7_joint": 0.01,
            },
            friction=0,
            min_delay=0,
            max_delay=4,
            friction_static={
                "leg_[l,r]1_joint": 1.0,
                "leg_[l,r]2_joint": 0.5,
                "leg_[l,r]3_joint": 0.5,
                "leg_[l,r]4_joint": 1.0,
                "leg_[l,r]5_joint": 0.2,
                "leg_[l,r]6_joint": 0.2,
                "zarm_[l,r]1_joint": 0.5,
                "zarm_[l,r]2_joint": 0.3,
                "zarm_[l,r]3_joint": 0.2,
                "zarm_[l,r]4_joint": 0.3,
                "zarm_[l,r]5_joint": 0.1,
                "zarm_[l,r]6_joint": 0.1,
                "zarm_[l,r]7_joint": 0.1,
            },
            activation_vel=0.1,
            friction_dynamic=0,
        ),
    },
)


Kuavos49_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        asset_path=resolve_s49_urdf_path(),
        activate_contact_sensors=True,
        # Isaac Lab 1.4.1: use default_drive_type, not JointDriveCfg (newer API only)
        default_drive_type="none",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            "leg_[l,r]1_joint": 0.0,
            "leg_[l,r]2_joint": 0.0,
            "leg_[l,r]3_joint": -0.27,
            "leg_[l,r]4_joint": 0.52,
            "leg_[l,r]5_joint": -0.3,
            "leg_[l,r]6_joint": 0.0,
            "zarm_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Joint-space PD (no S42 ankle solver). Effort limits match S46 training USD for sim stability.
        "motor": DelayedPDActuatorCfg_S49(
            joint_names_expr=[
                "leg_.*",
                "zarm_.*_joint",
            ],
            effort_limit={
                "leg_[l,r]1_joint": 180.0,
                "leg_[l,r]2_joint": 100.0,
                "leg_[l,r]3_joint": 100.0,
                "leg_[l,r]4_joint": 180.0,
                "leg_[l,r]5_joint": 72.0,
                "leg_[l,r]6_joint": 36.0,
                "zarm_[l,r]1_joint": 100.0,
                "zarm_[l,r]2_joint": 50.0,
                "zarm_[l,r]3_joint": 36.0,
                "zarm_[l,r]4_joint": 50.0,
                "zarm_[l,r]5_joint": 12.0,
                "zarm_[l,r]6_joint": 12.0,
                "zarm_[l,r]7_joint": 12.0,
            },
            velocity_limit={
                "leg_[l,r]1_joint": 14.0,
                "leg_[l,r]2_joint": 23.0,
                "leg_[l,r]3_joint": 23.0,
                "leg_[l,r]4_joint": 14.0,
                "leg_[l,r]5_joint": 10.0,
                "leg_[l,r]6_joint": 10.0,
                "zarm_[l,r]1_joint": 23.0,
                "zarm_[l,r]2_joint": 23.0,
                "zarm_[l,r]3_joint": 23.0,
                "zarm_[l,r]4_joint": 23.0,
                "zarm_[l,r]5_joint": 23.0,
                "zarm_[l,r]6_joint": 23.0,
                "zarm_[l,r]7_joint": 23.0,
            },
            stiffness={
                "leg_[l,r]1_joint": 100.0,
                "leg_[l,r]2_joint": 100.0,
                "leg_[l,r]3_joint": 100.0,
                "leg_[l,r]4_joint": 150.0,
                "leg_[l,r]5_joint": 40.0,
                "leg_[l,r]6_joint": 40.0,
                "zarm_[l,r]1_joint": 30.0,
                "zarm_[l,r]2_joint": 30.0,
                "zarm_[l,r]3_joint": 30.0,
                "zarm_[l,r]4_joint": 20.0,
                "zarm_[l,r]5_joint": 10.0,
                "zarm_[l,r]6_joint": 10.0,
                "zarm_[l,r]7_joint": 10.0,
            },
            damping={
                "leg_[l,r]1_joint": 4.0,
                "leg_[l,r]2_joint": 4.0,
                "leg_[l,r]3_joint": 4.0,
                "leg_[l,r]4_joint": 8.0,
                "leg_[l,r]5_joint": 4.0,
                "leg_[l,r]6_joint": 4.0,
                "zarm_[l,r]1_joint": 3.0,
                "zarm_[l,r]2_joint": 3.0,
                "zarm_[l,r]3_joint": 3.0,
                "zarm_[l,r]4_joint": 3.0,
                "zarm_[l,r]5_joint": 3.0,
                "zarm_[l,r]6_joint": 3.0,
                "zarm_[l,r]7_joint": 3.0,
            },
            armature={
                "leg_[l,r]1_joint": 0.05,
                "leg_[l,r]2_joint": 0.025,
                "leg_[l,r]3_joint": 0.025,
                "leg_[l,r]4_joint": 0.05,
                "leg_[l,r]5_joint": 0.05,
                "leg_[l,r]6_joint": 0.05,
                "zarm_[l,r]1_joint": 0.025,
                "zarm_[l,r]2_joint": 0.02,
                "zarm_[l,r]3_joint": 0.02,
                "zarm_[l,r]4_joint": 0.02,
                "zarm_[l,r]5_joint": 0.01,
                "zarm_[l,r]6_joint": 0.01,
                "zarm_[l,r]7_joint": 0.01,
            },
            friction=0,
            min_delay=0,
            max_delay=4,
            friction_static={
                "leg_[l,r]1_joint": 1.0,
                "leg_[l,r]2_joint": 0.5,
                "leg_[l,r]3_joint": 0.5,
                "leg_[l,r]4_joint": 1.0,
                "leg_[l,r]5_joint": 0.2,
                "leg_[l,r]6_joint": 0.2,
                "zarm_[l,r]1_joint": 0.5,
                "zarm_[l,r]2_joint": 0.3,
                "zarm_[l,r]3_joint": 0.2,
                "zarm_[l,r]4_joint": 0.3,
                "zarm_[l,r]5_joint": 0.1,
                "zarm_[l,r]6_joint": 0.1,
                "zarm_[l,r]7_joint": 0.1,
            },
            activation_vel=0.1,
            friction_dynamic=0,
        ),
    },
)
