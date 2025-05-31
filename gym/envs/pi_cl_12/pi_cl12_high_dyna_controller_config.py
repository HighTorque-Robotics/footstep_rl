"""
Configuration file for "fixed arm" (FA) humanoid environment
with potential-based rewards implemented
"""

import torch
from gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotRunnerCfg


class Picl12HighDynaControllerCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actuators = 12
        episode_length_s = 17  # 100
        frame_stack = 1
        num_single_obs = 35
        num_observations = frame_stack * num_single_obs

    class terrain(LeggedRobotCfg.terrain):
        curriculum = False
        mesh_type = "plane"  # 'plane' 'heightfield' 'trimesh'
        measure_heights = False  # True, False
        measured_points_x_range = [-0.8, 0.8]
        measured_points_x_num_sample = 33
        measured_points_y_range = [-0.8, 0.8]
        measured_points_y_num_sample = 33
        selected = True  # True, False
        terrain_kwargs = {"type": "stepping_stones"}
        # terrain_kwargs = {'type': 'random_uniform'}
        # terrain_kwargs = {'type': 'gap'}
        # difficulty = 0.35 # For gap terrain
        # platform_size = 5.5 # For gap terrain
        difficulty = 5.0  # For rough terrain
        terrain_length = 18.0  # For rough terrain
        terrain_width = 18.0  # For rough terrain
        # terrain types: [pyramid_sloped, random_uniform, stairs down, stairs up, discrete obstacles, stepping_stones, gap, pit]
        terrain_proportions = [0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0]

    class init_state(LeggedRobotCfg.init_state):
        # reset_mode = 'reset_to_range' # 'reset_to_basic'
        reset_mode = "reset_to_basic"  # 'reset_to_basic'
        pos = [0.0, 0.0, 0.3453]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # ranges for [x, y, z, roll, pitch, yaw]
        root_pos_range = [
            [0.0, 0.0],  # x
            [0.0, 0.0],  # y
            [0.3453, 0.36],  # z
            [-torch.pi / 10, torch.pi / 10],  # roll
            [-torch.pi / 10, torch.pi / 10],  # pitch
            [-torch.pi / 10, torch.pi / 10],  # yaw
        ]

        # ranges for [v_x, v_y, v_z, w_x, w_y, w_z]
        root_vel_range = [
            [-0.5, 0.5],  # x
            [-0.5, 0.5],  # y
            [-0.5, 0.5],  # z
            [-0.5, 0.5],  # roll
            [-0.5, 0.5],  # pitch
            [-0.5, 0.5],  # yaw
        ]

        default_joint_angles = {
            "r_hip_pitch_joint": 0.0,
            "r_hip_roll_joint": 0.0,
            "r_thigh_joint": 0,
            "r_calf_joint": 0.0,
            "r_ankle_pitch_joint": 0.0,
            "r_ankle_roll_joint": 0.0,
            "l_hip_pitch_joint": 0.0,
            "l_hip_roll_joint": 0.0,
            "l_thigh_joint": 0,
            "l_calf_joint": 0.0,
            "l_ankle_pitch_joint": 0.0,
            "l_ankle_roll_joint": 0.0,
        }

        dof_pos_range = {
            "r_hip_pitch_joint": [-1.25, 1.75],
            "r_hip_roll_joint": [-0.5, 0.12],
            "r_thigh_joint": [-0.6, 0.3],
            "r_calf_joint": [-0.65, 1.65],
            "r_ankle_pitch_joint": [-0.5, 1.3],
            "r_ankle_roll_joint": [-0.15, 0.15],
            "l_hip_pitch_joint": [-1.25, 1.75],
            "l_hip_roll_joint": [-0.12, 0.5],
            "l_thigh_joint": [-0.3, 0.6],
            "l_calf_joint": [-0.65, 1.65],
            "l_ankle_pitch_joint": [-0.5, 1.3],
            "l_ankle_roll_joint": [-0.15, 0.15],
        }

        dof_vel_range = {
            "r_hip_pitch_joint": [-0.1, 0.1],
            "r_hip_roll_joint": [-0.1, 0.1],
            "r_thigh_joint": [-0.1, 0.1],
            "r_calf_joint": [-0.1, 0.1],
            "r_ankle_pitch_joint": [-0.1, 0.1],
            "r_ankle_roll_joint": [-0.1, 0.1],
            "l_hip_pitch_joint": [-0.1, 0.1],
            "l_hip_roll_joint": [-0.1, 0.1],
            "l_thigh_joint": [-0.1, 0.1],
            "l_calf_joint": [-0.1, 0.1],
            "l_ankle_pitch_joint": [-0.1, 0.1],
            "l_ankle_roll_joint": [-0.1, 0.1],
        }

    class control(LeggedRobotCfg.control):
        # stiffness and damping for joints
        
        stiffness = {
            "r_hip_pitch_joint": 25.0,
            "r_hip_roll_joint": 25.0,
            "r_thigh_joint": 25.0,
            "r_calf_joint": 25.0,
            "r_ankle_pitch_joint": 25.0,
            "r_ankle_roll_joint": 25.0,
            "l_hip_pitch_joint": 25.0,
            "l_hip_roll_joint": 25.0,
            "l_thigh_joint": 25.0,
            "l_calf_joint": 25.0,
            "l_ankle_pitch_joint": 25.0,
            "l_ankle_roll_joint": 25.0,
        }
        damping = {
            "r_hip_pitch_joint": 0.2,
            "r_hip_roll_joint": 0.2,
            "r_thigh_joint": 0.2,
            "r_calf_joint": 0.2,
            "r_ankle_pitch_joint": 0.2,
            "r_ankle_roll_joint": 0.2,
            "l_hip_pitch_joint": 0.2,
            "l_hip_roll_joint": 0.2,
            "l_thigh_joint": 0.2,
            "l_calf_joint": 0.2,
            "l_ankle_pitch_joint": 0.2,
            "l_ankle_roll_joint": 0.2,
        }

        # stiffness = {
        #     "r_hip_pitch_joint": 60.0,
        #     "r_hip_roll_joint": 40.0,
        #     "r_thigh_joint": 20.0,
        #     "r_calf_joint": 60.0,
        #     "r_ankle_pitch_joint": 30,
        #     "r_ankle_roll_joint": 10,
        #     "l_hip_pitch_joint": 60.0,
        #     "l_hip_roll_joint": 40.0,
        #     "l_thigh_joint": 20.0,
        #     "l_calf_joint": 60.0,
        #     "l_ankle_pitch_joint": 30,
        #     "l_ankle_roll_joint": 10,
        # }
        # damping = {
        #     "r_hip_pitch_joint": 2.4,
        #     "r_hip_roll_joint": 0.8,
        #     "r_thigh_joint": 0.4,
        #     "r_calf_joint": 2.8,
        #     "r_ankle_pitch_joint": 1.6,
        #     "r_ankle_roll_joint": 0.3,
        #     "l_hip_pitch_joint": 2.4,
        #     "l_hip_roll_joint": 0.8,
        #     "l_thigh_joint": 0.4,
        #     "l_calf_joint": 2.8,
        #     "l_ankle_pitch_joint": 1.6,
        #     "l_ankle_roll_joint": 0.3,
        # }

        actuation_scale = 1.0
        exp_avg_decay = None
        decimation = 10

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 5.0
        num_commands = 3
        resampling_time = 5.0  # 5.

        succeed_step_radius = 0.03
        succeed_step_angle = 10
        apex_height_percentage = 0.8

        sample_angle_offset = 20 # no use
        sample_radius_offset = 0.05# no use

        dstep_length = 0.1
        dstep_width = 0.16

        class ranges(LeggedRobotCfg.commands.ranges):
            # TRAINING STEP COMMAND RANGES #
            sample_period = [20, 21]  # [20, 21] # equal to gait frequency
            dstep_width = [0.13, 0.2]  # [0.2, 0.4] # min max [m]

            lin_vel_x = [-1.5, 1.5]  # [-3.0, 3.0] # min max [m/s]
            lin_vel_y = 0.75  # 1.5   # min max [m/s]
            yaw_vel = 0.0  # min max [rad/s]

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True  # True, False
        friction_range = [0.1, 1.25]

        randomize_base_mass = True  # True, False
        added_mass_range = [-1.2, 1.2]

        push_robots = True
        push_interval_s = 3
        max_push_vel_xy = 2.7

        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]

        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]

        randomize_Kp_factor = True
        Kp_factor_range = [0.9, 1.1]

        randomize_Kd_factor = True
        Kd_factor_range = [0.9, 1.1]

        # Add DR for rotor inertia and angular damping

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/pi_12dof_release_v1/urdf/pi_12dof_release_v1_rl.urdf"
        keypoints = ["base_link"]
        foot_name = "ankle_roll"
        end_effectors = ["r_" + foot_name, "l_" + foot_name]
        terminate_after_contacts_on = [
            "base_link",
            "r_thigh_link",
            "r_calf_link",
            "r_ankle_pitch_link",
            # "r_ankle_roll_link",
            "l_thigh_link",
            "l_calf_link",
            "l_ankle_pitch_link",
            # "l_ankle_roll_link",
        ]

        disable_gravity = False
        disable_actuations = False
        disable_motors = False

        # (1: disable, 0: enable...bitwise filter)
        self_collisions = 0
        collapse_fixed_joints = True
        flip_visual_attachments = False

        # Check GymDofDriveModeFlags
        # (0: none, 1: pos tgt, 2: vel target, 3: effort)
        default_dof_drive_mode = 3

        angular_damping = 0.1
        rotor_inertia = [
            0.0001188,  # RIGHT LEG
            0.0001188,
            0.0001188,
            0.0001188,
            0.0001188,
            0.0001188,
            0.0001188,  # LEFT LEG
            0.0001188,
            0.0001188,
            0.0001188,
            0.0001188,
            0.0001188,
        ]
        apply_humanoid_jacobian = False  # True, False

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.3453
        base_height_range = 0.01
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.8
        max_contact_force = 100.0

        curriculum = False
        only_positive_rewards = False
        tracking_sigma = 0.25
        min_dist_feet = 0.1602 - 0.03  # 0.240257 0.1601802 0.0956606
        max_dist_feet = 0.1602 + 0.04
        class weights(LeggedRobotCfg.rewards.weights):
            # * Regularization rewards * #
            actuation_rate = 2e-4
            actuation_rate2 = 1e-3
            torques = 1e-4
            dof_vel = 1e-3
            lin_vel_z = 1
            ang_vel_xy = 1e-2
            dof_pos_limits = 10
            torque_limits = 1

            # * Floating base rewards * #
            base_height = 1.0
            base_heading = 4.0
            base_z_orientation = 2.0
            tracking_lin_vel_world = 4.0

            # * Stepping rewards * #
            joint_regularization = 2.0
            contact_schedule = 3.0

            # * Other * #
            feet_distance = 1.0
            ankle_roll_posture_roll = 5.0
            ankle_roll_posture_pitch = 0.5


            #base over com
            # base_com = 1.0
        class termination_weights(LeggedRobotCfg.rewards.termination_weights):
            termination = 1.0

    class scaling(LeggedRobotCfg.scaling):
        base_height = 1.0
        base_lin_vel = 1.0  # .5
        base_ang_vel = 1.0  # 2.
        projected_gravity = 1.0
        foot_states_right = 1.0
        foot_states_left = 1.0
        dof_pos = 1.0
        dof_vel = 1.0  # .1
        dof_pos_target = dof_pos  # scale by range of motion

        # Action scales
        commands = 1.0
        clip_actions = 10.0


class Picl12HighDynaControllerRunnerCfg(LeggedRobotRunnerCfg):
    do_wandb = True
    seed = -1

    class policy(LeggedRobotRunnerCfg.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [256, 256, 256]
        critic_hidden_dims = [256, 256, 256]
        # (elu, relu, selu, crelu, lrelu, tanh, sigmoid)
        activation = "elu"
        normalize_obs = True  # True, False
        
        single_actor_obs = [
            "phase_sin",
            "phase_cos",
            "commands",
            "dof_pos",
            "dof_vel",
            "base_ang_vel",
            "base_euler_xyz",
        ]

        actor_obs = [
            "phase_sin",
            "phase_cos",
            "commands",
            "dof_pos",
            "dof_vel",
            "base_ang_vel",
            "base_euler_xyz",
        ]

        critic_obs = [
            "base_height",
            "base_lin_vel_world",  # "base_lin_vel",
            "base_euler_xyz",
            "base_ang_vel",
            "projected_gravity",
            "foot_states_right",
            "foot_states_left",
            "step_commands_right",
            "step_commands_left",
            "commands",
            "phase_sin",
            "phase_cos",
            "dof_pos",
            "dof_vel",
        ]

        actions = ["dof_pos_target"]

        class noise:
            # base_height = 0.05
            base_lin_vel = 0.05
            base_lin_vel_world = 0.05
            base_heading = 0.01
            # base_ang_vel = 0.05
            base_euler_xyz = 0.1
            projected_gravity = 0.1
            foot_states_right = 0.01
            foot_states_left = 0.01
            step_commands_right = 0.01
            step_commands_left = 0.01
            commands = 0.1
            dof_pos = 0.05
            # dof_vel = 0.5
            foot_contact = 0.1

            #dreamwaq noise
            dof_vel = 0.5 * 2
            base_ang_vel = 0.1
            quat = 0.03
            gravity = 0.5
            base_height = 0.1

    class algorithm(LeggedRobotRunnerCfg.algorithm):
        class PPO:
            # algorithm training hyperparameters
            value_loss_coef = 1.0
            use_clipped_value_loss = True
            clip_param = 0.2
            entropy_coef = 0.01
            num_learning_epochs = 5
            num_mini_batches = 4  # minibatch size = num_envs*nsteps/nminibatches
            learning_rate = 2.0e-5
            schedule = "adaptive"  # could be adaptive, fixed
            gamma = 0.99
            lam = 0.95
            desired_kl = 0.01
            max_grad_norm = 1.0

    class runner(LeggedRobotRunnerCfg.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 20000
        run_name = "sf"
        experiment_name = "Picl12_high_dyna_Controller"
        # experiment_name = "Picl12_Controller"
        save_interval = 100
        plot_input_gradients = False
        plot_parameter_gradients = False
