"""
Hierarchical structure for Deep Stepper for Humanoid
1) Low-level policy: Step controller trained by PPO
    - It is divided into two section. (1) Only one-step controller (2) Continuous-step controller
2) High-level policy: Step planner trained by SAC

Purpose: Given a base velocity command (linear x,y velocity, angular velocity), 
         robot determines stepping locations to follow the commanded velocity

This script serves as a Low-level policy which actuate the robot to take a step

* All variables are calculated w.r.t world frame
* However, when the variables are put into observation, it is converted w.r.t base frame
"""

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from gym.envs.pi_cl_12.pi_cl12_controller_config import Picl12ControllerCfg
from gym.utils.math import *
from gym.envs import LeggedRobot
from isaacgym import gymapi, gymutil
import numpy as np
from typing import Tuple, Dict
from .pi_cl12_utils import (
    FootStepGeometry,
    SimpleLineGeometry,
    VelCommandGeometry,
    smart_sort,
)
from gym.utils import XCoMKeyboardInterface
from .jacobian import apply_coupling
from scipy.signal import correlate
import torch.nn.functional as F


def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz


class Picl12HighDynaController(LeggedRobot):
    cfg: Picl12ControllerCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _setup_keyboard_interface(self):
        self.keyboard_interface = XCoMKeyboardInterface(self)

    def _init_buffers(self):
        super()._init_buffers()
        # * Robot states
        self.base_height = self.root_states[:, 2:3]
        self.right_hip_pos = self.rigid_body_state[
            :, self.rigid_body_idx["r_hip_pitch_link"], :3
        ]
        self.left_hip_pos = self.rigid_body_state[
            :, self.rigid_body_idx["l_hip_pitch_link"], :3
        ]
        self.CoM = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.foot_states = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            7,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # num_envs x (right & left foot) x (x, y, z, quat)
        self.foot_states_right = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )  # num_envs x (x, y, z, heading, projected_gravity)
        self.foot_states_left = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )  # num_envs x (x, y, z, heading, projected_gravity)
        self.foot_heading = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # num_envs x (right & left foot heading)
        self.foot_projected_gravity = torch.stack(
            (self.gravity_vec, self.gravity_vec), dim=1
        )  # (num_envs x 2 x 3), [0., 0., -1.]
        self.foot_contact = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )  # contacts on right & left sole
        self.ankle_vel_history = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            2 * 3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.base_heading = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.base_lin_vel_world = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )

        # * Step commands
        self.step_commands = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # (right & left foot) x (x, y, heading) wrt base x,y-coordinate
        self.step_commands_right = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )  # (right foot) x (x, y, heading) wrt base x,y-coordinate
        self.step_commands_left = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )  # (left foot) x (x, y, heading) wrt base x,y-coordinate
        self.foot_on_motion = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )  # True foot is on command
        self.step_period = torch.zeros(
            self.num_envs, 1, dtype=torch.long, device=self.device, requires_grad=False
        )
        self.full_step_period = torch.zeros(
            self.num_envs, 1, dtype=torch.long, device=self.device, requires_grad=False
        )  # full_step_period = 2 * step_period
        self.ref_foot_trajectories = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # (right & left foot) x (x, y, heading) wrt base x,y-coordinate

        # * Step states
        self.current_step = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # (right & left foot) x (x, y, heading) wrt base x,y-coordinate
        self.prev_step_commands = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # (right & left foot) x (x, y, heading) wrt base x,y-coordinate
        self.step_location_offset = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # num_envs x (right & left foot)
        self.step_heading_offset = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # num_envs x (right & left foot)
        self.succeed_step_radius = torch.tensor(
            self.cfg.commands.succeed_step_radius,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.succeed_step_angle = torch.tensor(
            np.deg2rad(self.cfg.commands.succeed_step_angle),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.semi_succeed_step = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )  # whether foot is close to step_command
        self.succeed_step = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )  # whether steps are successful
        self.already_succeed_step = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )  # check if robot succeed given step command
        self.had_wrong_contact = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )  # check if it has had wrong contact
        self.step_stance = torch.zeros(
            self.num_envs, 1, dtype=torch.long, device=self.device, requires_grad=False
        )  # step_stance = previous step_period

        # * Others
        self.update_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
        )  # Number of transition since the beginning of the episode
        self.update_commands_ids = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )  # envs whose step commands are updated
        self.phase_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
        )  # Number of phase progress
        self.update_phase_ids = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )  # envs whose phases are updated
        self.phase = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )  # phase of current step in a whole gait cycle
        self.ICP = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )  # Instantaneous Capture Point (ICP) for the robot
        self.raibert_heuristic = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # step_location & angle by raibert heuristic
        self.w = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )  # eigenfrequency of the inverted pendulum
        self.step_length = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )  # step length
        self.step_width = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )  # step width
        self.dstep_length = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )  # desired step length
        self.dstep_width = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )  # desired step width
        self.support_foot_pos = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )  # position of the support foot
        self.prev_support_foot_pos = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )  # position of the support foot
        self.LIPM_CoM = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )  # base position of the Linear Inverted Pendulum model

        # * Observation variables
        self.phase_sin = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.phase_cos = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.contact_schedule = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )

    def _compute_torques(self):
        self.desired_pos_target = self.dof_pos_target + self.default_dof_pos
        q = self.dof_pos.clone()
        qd = self.dof_vel.clone()
        q_des = self.desired_pos_target.clone()
        qd_des = torch.zeros_like(self.dof_pos_target)
        tau_ff = torch.zeros_like(self.dof_pos_target)
        kp = self.p_gains.clone()
        kd = self.d_gains.clone()

        if self.cfg.asset.apply_humanoid_jacobian:
            torques = apply_coupling(q, qd, q_des, qd_des, kp, kd, tau_ff)
        else:
            torques = kp * (q_des - q) + kd * (qd_des - qd) + tau_ff

        torques = torch.clip(torques, -self.torque_limits, self.torque_limits)

        return torques.view(self.torques.shape)

    def _resample_commands(self, env_ids):
        """Randomly select foot step commands one/two steps ahead"""
        super()._resample_commands(env_ids)

        self.step_period[env_ids] = torch.randint(
            low=self.command_ranges["sample_period"][0],
            high=self.command_ranges["sample_period"][1],
            size=(len(env_ids), 1),
            device=self.device,
        )
        self.full_step_period = 2 * self.step_period

        self.step_stance[env_ids] = torch.clone(self.step_period[env_ids])

        # * Randomly select the desired step width
        self.dstep_width[env_ids] = torch_rand_float(
            self.command_ranges["dstep_width"][0],
            self.command_ranges["dstep_width"][1],
            (len(env_ids), 1),
            self.device,
        )

    def _reset_system(self, env_ids):
        super()._reset_system(env_ids)
        # * Robot states
        self.foot_states[env_ids] = self._calculate_foot_states(
            self.rigid_body_state[:, self.feet_ids, :7]
        )[env_ids]
        self.foot_projected_gravity[env_ids, 0] = self.gravity_vec[env_ids]
        self.foot_projected_gravity[env_ids, 1] = self.gravity_vec[env_ids]

        # * Step commands
        self.step_commands[env_ids, 0] = self.env_origins[env_ids] + torch.tensor(
            [0.0, -0.085, 0.0], device=self.device
        )  # right foot initializatoin
        self.step_commands[env_ids, 1] = self.env_origins[env_ids] + torch.tensor(
            [0.0, 0.085, 0.0], device=self.device
        )  # left foot initializatoin
        self.foot_on_motion[env_ids, 0] = False
        self.foot_on_motion[env_ids, 1] = True  # When resample, left feet is swing foot

        # * Step states
        self.current_step[env_ids] = torch.clone(
            self.step_commands[env_ids]
        )  # current_step initializatoin
        self.prev_step_commands[env_ids] = torch.clone(self.step_commands[env_ids])
        self.semi_succeed_step[env_ids] = False
        self.succeed_step[env_ids] = False
        self.already_succeed_step[env_ids] = False
        self.had_wrong_contact[env_ids] = False

        # * Others
        self.update_count[env_ids] = 0
        self.update_commands_ids[env_ids] = False
        self.phase_count[env_ids] = 0
        self.update_phase_ids[env_ids] = False
        self.phase[env_ids] = 0
        self.ICP[env_ids] = 0.0
        self.raibert_heuristic[env_ids] = 0.0
        self.w[env_ids] = 0.0
        self.dstep_length[env_ids] = self.cfg.commands.dstep_length
        self.dstep_width[env_ids] = self.cfg.commands.dstep_width

    def _post_physics_step_callback(self):
        # print("_post_physics_step_callback")
        super()._post_physics_step_callback()

        self._update_robot_states()
        self._calculate_CoM()
        self._calculate_raibert_heuristic()
        self._calculate_ICP()
        self._measure_success_rate()
        self._update_commands()
        # self._log_info()

    def _update_robot_states(self):
        # print("_update_robot_states")
        """Update robot state variables"""
        self.base_height[:] = self.root_states[:, 2:3]
        forward = quat_apply(self.base_quat, self.forward_vec)
        self.base_heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
        self.right_hip_pos = self.rigid_body_state[
            :, self.rigid_body_idx["r_hip_pitch_link"], :3
        ]
        self.left_hip_pos = self.rigid_body_state[
            :, self.rigid_body_idx["l_hip_pitch_link"], :3
        ]
        self.foot_states = self._calculate_foot_states(
            self.rigid_body_state[:, self.feet_ids, :7]
        )

        right_foot_forward = quat_apply(self.foot_states[:, 0, 3:7], self.forward_vec)
        left_foot_forward = quat_apply(self.foot_states[:, 1, 3:7], self.forward_vec)
        right_foot_heading = wrap_to_pi(
            torch.atan2(right_foot_forward[:, 1], right_foot_forward[:, 0])
        )
        left_foot_heading = wrap_to_pi(
            torch.atan2(left_foot_forward[:, 1], left_foot_forward[:, 0])
        )
        self.foot_heading[:, 0] = right_foot_heading
        self.foot_heading[:, 1] = left_foot_heading

        self.foot_projected_gravity[:, 0] = quat_rotate_inverse(
            self.foot_states[:, 0, 3:7], self.gravity_vec
        )
        self.foot_projected_gravity[:, 1] = quat_rotate_inverse(
            self.foot_states[:, 1, 3:7], self.gravity_vec
        )

        self.update_count += 1
        self.phase_count += 1
        self.phase += 1 / self.full_step_period

        # * Ground truth foot contact
        self.foot_contact = torch.gt(self.contact_forces[:, self.feet_ids, 2], 0)

        # * Phase-based foot contact
        self.contact_schedule = self.smooth_sqr_wave(self.phase)

        # * Update current step
        current_step_masked = self.current_step[self.foot_contact]
        current_step_masked[:, :2] = self.foot_states[self.foot_contact][:, :2]
        current_step_masked[:, 2] = self.foot_heading[self.foot_contact]
        self.current_step[self.foot_contact] = current_step_masked

        naxis = 3
        self.ankle_vel_history[:, 0, naxis:] = self.ankle_vel_history[:, 0, :naxis]
        self.ankle_vel_history[:, 0, :naxis] = self.rigid_body_state[
            :, self.rigid_body_idx["r_ankle_roll_link"], 7:10
        ]
        self.ankle_vel_history[:, 1, naxis:] = self.ankle_vel_history[:, 1, :naxis]
        self.ankle_vel_history[:, 1, :naxis] = self.rigid_body_state[
            :, self.rigid_body_idx["l_ankle_roll_link"], 7:10
        ]

    def _calculate_foot_states(self, foot_states):
        # print("_calculate_foot_states")
        foot_height_vec = (
            torch.tensor([-0.06, 0.0, -0.05063]).repeat(self.num_envs, 1).to(self.device)
        )
        # print("foot_states[:, 0, 3:7]:",foot_states[:, 0, :3])
        rfoot_height_vec_in_world = quat_apply(foot_states[:, 0, 3:7], foot_height_vec)
        lfoot_height_vec_in_world = quat_apply(foot_states[:, 1, 3:7], foot_height_vec)
        foot_states[:, 0, :3] += rfoot_height_vec_in_world
        foot_states[:, 1, :3] += lfoot_height_vec_in_world

        return foot_states

    def _calculate_CoM(self):
        # print("_calculate_CoM")
        """Calculates the Center of Mass of the robot"""
        self.CoM = (
            self.rigid_body_state[:, :, :3] * self.rigid_body_mass.unsqueeze(1)
        ).sum(dim=1)  / self.mass_total 

        self.CoM[:,0]-=0.015

    def _calculate_ICP(self):
        # print("_calculate_ICP")
        """Calculates the Instantaneous Capture Point (ICP) of the robot
        x_ic = x + x'/w where w = sqrt(g/z)
        y_ic = y + y'/w where w = sqrt(g/z)
        """
        g = -self.sim_params.gravity.z
        self.w = torch.sqrt(g / self.CoM[:, 2:3])
        self.ICP[:, :2] = self.CoM[:, :2] + self.root_states[:, 7:9] / self.w

    def _calculate_raibert_heuristic(self):
        # print("_calculate_raibert_heuristic")
        """<step location>
        r = p_hip + p_symmetry + p_centrifugal
        where p_hip is the position of the hip
              p_symmetry = (0.5 * t_stance) * v + k * (v - v_cmd)
              p_centrifugal = 0.5 * sqrt(h/g) * (v x w_cmd)
        <step angle>
        theta = previous_step_angle + 2 * w_cmd * step_period
        """
        g = -self.sim_params.gravity.z
        k = torch.sqrt(self.CoM[:, 2:3] / g)
        p_symmetry = 0.5 * self.step_stance * self.dt * self.base_lin_vel_world[
            :, :2
        ] + k * (self.base_lin_vel_world[:, :2] - self.commands[:, :2])

        self.raibert_heuristic[:, 0, :2] = self.right_hip_pos[:, :2] + p_symmetry
        self.raibert_heuristic[:, 1, :2] = self.left_hip_pos[:, :2] + p_symmetry

    def _measure_success_rate(self):
        # print("_measure_success_rate")
        # * Measure success rate of step commands
        # print("_measure_success_rate len(self.feet_ids):",len(self.feet_ids))
        self.step_location_offset = torch.norm(
            self.foot_states[:, :, :3]
            - torch.cat(
                (
                    self.step_commands[:, :, :2],
                    torch.zeros(
                        (self.num_envs, len(self.feet_ids), 1), device=self.device
                    ),
                ),
                dim=2,
            ),
            dim=2,
        )
        # print("_measure_success_rate self.step_location_offset:",self.step_location_offset)
        self.step_heading_offset = torch.abs(
            wrap_to_pi(self.foot_heading - self.step_commands[:, :, 2])
        )
        self.semi_succeed_step = (
            self.step_location_offset < self.succeed_step_radius
        ) & (self.step_heading_offset < self.succeed_step_angle)

        self.prev_step_location_offset = torch.norm(
            self.foot_states[:, :, :3]
            - torch.cat(
                (
                    self.prev_step_commands[:, :, :2],
                    torch.zeros(
                        (self.num_envs, len(self.feet_ids), 1), device=self.device
                    ),
                ),
                dim=2,
            ),
            dim=2,
        )
        self.prev_step_heading_offset = torch.abs(
            wrap_to_pi(self.foot_heading - self.prev_step_commands[:, :, 2])
        )
        self.prev_semi_succeed_step = (
            self.prev_step_location_offset < self.succeed_step_radius
        ) & (self.prev_step_heading_offset < self.succeed_step_angle)

        self.had_wrong_contact |= (
            self.foot_contact * ~self.semi_succeed_step * ~self.prev_semi_succeed_step
        )

        self.succeed_step = self.semi_succeed_step & ~self.had_wrong_contact

        self.succeed_step_ids = self.succeed_step.sum(dim=1) == 2
        # print('succeed_step:', self.succeed_step_ids)
        # self.succeed_step_ids = self._check_succeed_step_ids(self.succeed_step, self.foot_on_motion)
        self.already_succeed_step[self.succeed_step_ids] = True
        # contact_rewards = (
        #     self.foot_contact[:, 0].int() - self.foot_contact[:, 1].int()
        # ) * self.contact_schedule.squeeze(1)
        # print("_measure_success_rate contact_rewards: ",contact_rewards)

    def _update_commands(self):
        # print("_update_commands")
        """Update step commands"""
        # * Check env ids to update phase(freq*2)
        self.update_phase_ids = self.phase_count >= self.full_step_period.squeeze(1)
        self.phase_count[self.update_phase_ids] = 0
        self.phase[self.update_phase_ids] = 0

        # * Check env ids to update commands
        self.update_commands_ids = self.update_count >= self.step_period.squeeze(1)
        self.already_succeed_step[self.update_commands_ids] = False
        self.had_wrong_contact[self.update_commands_ids] = False
        self.update_count[self.update_commands_ids] = 0
        self.step_stance[self.update_commands_ids] = torch.clone(
            self.step_period[self.update_commands_ids]
        )

        # * Update foot_on_motion (At least one foot should be on motion, otherwise the robot cannot update step command)
        self.foot_on_motion[self.update_commands_ids] = ~self.foot_on_motion[
            self.update_commands_ids
        ]

        # * Update step_commands
        # self.update_commands_ids[:] = True
        update_step_commands_mask = self.step_commands[self.update_commands_ids]
        self.prev_step_commands[self.update_commands_ids] = torch.clone(
            self.step_commands[self.update_commands_ids]
        )
        # update_step_commands_mask[self.foot_on_motion[self.update_commands_ids]] = self._generate_step_command_by_raibert_heuristic(self.update_commands_ids) # * Raibert heuristic
        # update_step_commands_mask[self.foot_on_motion[self.update_commands_ids]] = self._generate_dynamic_step_command_by_raibert_heuristic(self.update_commands_ids) # * Raibert heuristic with dynamically changing step command
        update_step_commands_mask[self.foot_on_motion[self.update_commands_ids]] = (
            self._generate_step_command_by_3DLIPM_XCoM(self.update_commands_ids)
        )  # * XCoM paper
        # update_step_commands_mask[self.foot_on_motion[self.update_commands_ids]] = self._generate_dynamic_step_command_by_3DLIPM_XCoM(self.update_commands_ids) # * XCoM paper with dynamically changing step command
        self._update_LIPM_CoM(self.update_commands_ids)

        foot_collision_ids = (
            update_step_commands_mask[:, 0, :2] - update_step_commands_mask[:, 1, :2]
        ).norm(dim=1) < 0.2
        update_step_commands_mask[foot_collision_ids, :, :2] = (
            self._adjust_foot_collision(
                update_step_commands_mask[foot_collision_ids, :, :2],
                self.foot_on_motion[self.update_commands_ids][foot_collision_ids],
            )
        )

        if self.cfg.terrain.measure_heights:
            # update_step_commands_mask[self.foot_on_motion[self.update_commands_ids]] = self._adjust_step_command_in_gap_terrain(self.update_commands_ids,
            # update_step_commands_mask)
            update_step_commands_mask[self.foot_on_motion[self.update_commands_ids]] = (
                self._adjust_step_command_in_rough_terrain(
                    self.update_commands_ids, update_step_commands_mask
                )
            )

        self.step_commands[self.update_commands_ids] = update_step_commands_mask

    def _generate_step_command_by_3DLIPM_XCoM(self, update_commands_ids):
        # print("_generate_step_command_by_3DLIPM_XCoM")
        """Generate random step command by step command based on the XCoM paper
        x_0, y_0 : CoM position w.r.t support foot pos
        vx_0, vy_0 : CoM velocity
        We are assuming that the robot is always facing forward i.e. base_lin_vel is always (vx, 0, 0)
        """
        foot_on_motion = self.foot_on_motion[update_commands_ids]
        step_period = self.step_period[update_commands_ids]
        commands = self.commands[update_commands_ids]
        current_step = self.current_step[update_commands_ids]
        CoM = self.CoM[update_commands_ids]
        T = step_period * self.dt
        w = self.w[update_commands_ids]
        dstep_length = torch.norm(commands[:, :2], dim=1, keepdim=True) * T
        dstep_width = self.dstep_width[update_commands_ids]
        theta = torch.atan2(commands[:, 1:2], commands[:, 0:1])

        right_step_ids = torch.where(torch.where(foot_on_motion)[1] == 0)[0]
        left_step_ids = torch.where(torch.where(foot_on_motion)[1] == 1)[0]

        root_states = self.root_states[update_commands_ids]
        support_foot_pos = self.support_foot_pos[update_commands_ids]
        support_foot_pos[right_step_ids] = current_step[
            right_step_ids, 1, :3
        ]  # Left foot(=1) is support foot
        support_foot_pos[left_step_ids] = current_step[
            left_step_ids, 0, :3
        ]  # Right foot(=0) is support foot

        # * For logging purpose * #
        rright_foot_pos_x = (
            torch.cos(theta) * current_step[:, 0, 0:1]
            + torch.sin(theta) * current_step[:, 0, 1:2]
        )
        rright_foot_pos_y = (
            -torch.sin(theta) * current_step[:, 0, 0:1]
            + torch.cos(theta) * current_step[:, 0, 1:2]
        )
        rleft_foot_pos_x = (
            torch.cos(theta) * current_step[:, 1, 0:1]
            + torch.sin(theta) * current_step[:, 1, 1:2]
        )
        rleft_foot_pos_y = (
            -torch.sin(theta) * current_step[:, 1, 0:1]
            + torch.cos(theta) * current_step[:, 1, 1:2]
        )

        self.step_length[update_commands_ids] = torch.abs(
            rright_foot_pos_x - rleft_foot_pos_x
        )
        self.step_width[update_commands_ids] = torch.abs(
            rright_foot_pos_y - rleft_foot_pos_y
        )

        self.dstep_length[update_commands_ids] = dstep_length
        self.dstep_width[update_commands_ids] = dstep_width
        # * #################### * #

        random_step_command = torch.zeros(
            foot_on_motion.sum(),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        x_0 = CoM[:, 0:1] - support_foot_pos[:, 0:1]
        y_0 = CoM[:, 1:2] - support_foot_pos[:, 1:2]
        vx_0 = root_states[:, 7:8]
        vy_0 = root_states[:, 8:9]

        x_f = x_0 * torch.cosh(T * w) + vx_0 * torch.sinh(T * w) / w
        vx_f = x_0 * w * torch.sinh(T * w) + vx_0 * torch.cosh(T * w)
        y_f = y_0 * torch.cosh(T * w) + vy_0 * torch.sinh(T * w) / w
        vy_f = y_0 * w * torch.sinh(T * w) + vy_0 * torch.cosh(T * w)

        x_f_world = x_f + support_foot_pos[:, 0:1]
        y_f_world = y_f + support_foot_pos[:, 1:2]
        eICP_x = x_f_world + vx_f / w
        eICP_y = y_f_world + vy_f / w
        b_x = dstep_length / (torch.exp(T * w) - 1)
        b_y = dstep_width / (torch.exp(T * w) + 1)

        original_offset_x = -b_x
        original_offset_y = -b_y
        original_offset_y[left_step_ids] = b_y[left_step_ids]

        offset_x = (
            torch.cos(theta) * original_offset_x - torch.sin(theta) * original_offset_y
        )
        offset_y = (
            torch.sin(theta) * original_offset_x + torch.cos(theta) * original_offset_y
        )

        u_x = eICP_x + offset_x
        u_y = eICP_y + offset_y

        random_step_command[:, 0] = u_x.squeeze(1)
        random_step_command[:, 1] = u_y.squeeze(1)
        random_step_command[:, 2] = theta.squeeze(1)

        return random_step_command

    def _update_LIPM_CoM(self, update_commands_ids):
        # print("_update_LIPM_CoM")
        self.LIPM_CoM[update_commands_ids] = self.CoM[update_commands_ids].clone()

        T = self.dt
        g = -self.sim_params.gravity.z
        w = torch.sqrt(g / self.LIPM_CoM[:, 2:3])

        right_step_ids = torch.where(torch.where(self.foot_on_motion)[1] == 0)[0]
        left_step_ids = torch.where(torch.where(self.foot_on_motion)[1] == 1)[0]

        support_foot_pos = self.support_foot_pos.clone()
        support_foot_pos[right_step_ids] = self.current_step[
            right_step_ids, 1, :3
        ]  # Left foot(=1) is support foot
        support_foot_pos[left_step_ids] = self.current_step[
            left_step_ids, 0, :3
        ]  # Right foot(=0) is support foot

        x_0 = self.LIPM_CoM[:, 0:1] - support_foot_pos[:, 0:1]
        y_0 = self.LIPM_CoM[:, 1:2] - support_foot_pos[:, 1:2]
        vx_0 = self.root_states[:, 7:8]
        vy_0 = self.root_states[:, 8:9]

        x_f = x_0 * torch.cosh(T * w) + vx_0 * torch.sinh(T * w) / w
        vx_f = x_0 * w * torch.sinh(T * w) + vx_0 * torch.cosh(T * w)
        y_f = y_0 * torch.cosh(T * w) + vy_0 * torch.sinh(T * w) / w
        vy_f = y_0 * w * torch.sinh(T * w) + vy_0 * torch.cosh(T * w)

        x_f_world = x_f + support_foot_pos[:, 0:1]
        y_f_world = y_f + support_foot_pos[:, 1:2]

        self.LIPM_CoM[:, 0:1] = x_f_world
        self.LIPM_CoM[:, 1:2] = y_f_world
        # self.LIPM_CoM[:,2] = self.CoM[:,2]

    def _adjust_foot_collision(self, collision_step_commands, collision_foot_on_motion):
        # print("_adjust_foot_collision")
        """Adjust foot collision by moving the foot to the nearest point on the boundary"""
        collision_distance = (
            collision_step_commands[:, 0] - collision_step_commands[:, 1]
        ).norm(dim=1, keepdim=True)
        adjust_step_commands = torch.clone(collision_step_commands)
        adjust_step_commands[collision_foot_on_motion] = (
            collision_step_commands[~collision_foot_on_motion]
            + 0.1
            * (
                collision_step_commands[collision_foot_on_motion]
                - collision_step_commands[~collision_foot_on_motion]
            )
            / collision_distance
        )
        return adjust_step_commands

    def _update_command_curriculum(self, env_ids):
        # print("_update_command_curriculum")
        """Implements a curriculum of increasing commands
        Args: env_ids (List[int]): ids of environments being reset
        """
        pass

    def _update_reward_curriculum(self, env_ids):
        # print("_update_reward_curriculum")
        """Implements a curriculum of rewards
        Args: env_ids (List[int]): ids of environments being reset
        """
        pass

    def _set_obs_variables(self):
        # print("_set_obs_variables")
        self.foot_states_right[:, :3] = quat_rotate_inverse(
            self.base_quat, self.foot_states[:, 0, :3] - self.base_pos
        )
        self.foot_states_left[:, :3] = quat_rotate_inverse(
            self.base_quat, self.foot_states[:, 1, :3] - self.base_pos
        )
        self.foot_states_right[:, 3] = wrap_to_pi(
            self.foot_heading[:, 0] - self.base_heading.squeeze(1)
        )
        self.foot_states_left[:, 3] = wrap_to_pi(
            self.foot_heading[:, 1] - self.base_heading.squeeze(1)
        )

        self.step_commands_right[:, :3] = quat_rotate_inverse(
            self.base_quat,
            torch.cat(
                (
                    self.step_commands[:, 0, :2],
                    torch.zeros((self.num_envs, 1), device=self.device),
                ),
                dim=1,
            )
            - self.base_pos,
        )
        self.step_commands_left[:, :3] = quat_rotate_inverse(
            self.base_quat,
            torch.cat(
                (
                    self.step_commands[:, 1, :2],
                    torch.zeros((self.num_envs, 1), device=self.device),
                ),
                dim=1,
            )
            - self.base_pos,
        )
        self.step_commands_right[:, 3] = wrap_to_pi(
            self.step_commands[:, 0, 2] - self.base_heading.squeeze(1)
        )
        self.step_commands_left[:, 3] = wrap_to_pi(
            self.step_commands[:, 1, 2] - self.base_heading.squeeze(1)
        )

        self.phase_sin = torch.sin(2 * torch.pi * self.phase)
        self.phase_cos = torch.cos(2 * torch.pi * self.phase)

        self.base_lin_vel_world = self.root_states[:, 7:10].clone()

    def check_termination(self):
        # print("check_termination")
        """Check if environments need to be reset"""
        # * Termination for contact
        term_contact = torch.norm(
            self.contact_forces[:, self.termination_contact_indices, :], dim=-1
        )
        self.terminated = torch.any((term_contact > 1.0), dim=1)
        # print(
        #     "term_contact: ",
        #     torch.sum(
        #         torch.any((term_contact > 1.0), dim=1)
        #     ).item(),
        # )
        # * Termination for velocities, orientation, and low height
        self.terminated |= torch.any(
            torch.norm(self.base_lin_vel, dim=-1, keepdim=True) > 15.0, dim=1
        )
        # print(
        #     "self.base_lin_vel: ",
        #     torch.sum(
        #         torch.any(
        #             torch.norm(self.base_lin_vel, dim=-1, keepdim=True) > 15.0, dim=1
        #         )
        #     ).item(),
        # )
        self.terminated |= torch.any(
            torch.norm(self.base_ang_vel, dim=-1, keepdim=True) > 15.0, dim=1
        )
        # print(
        #     "self.base_ang_vel: ",
        #     torch.sum(
        #         torch.any(
        #             torch.norm(self.base_ang_vel, dim=-1, keepdim=True) > 15.0, dim=1
        #         )
        #     ).item(),
        # )
        # print("self.base_ang_vel: ",torch.norm(self.base_ang_vel, dim=-1, keepdim=True))
        # self.terminated |= torch.any(
        #     torch.abs(self.projected_gravity[:, 0:1]) > 0.9, dim=1
        # )
        # print(
        #     "self.projected_gravity: ",
        #     torch.sum(
        #         torch.any(torch.abs(self.projected_gravity[:, 0:1]) > 0.9, dim=1)
        #     ).item(),
        # )

        # self.terminated |= torch.any(
        #     torch.abs(self.projected_gravity[:, 1:2]) > 0.9, dim=1
        # )
        # print(
        #     "self.projected_gravity: ",
        #     torch.sum(
        #         torch.any(torch.abs(self.projected_gravity[:, 1:2]) > 0.9, dim=1)
        #     ).item(),
        # )

        self.terminated |= torch.any(self.base_pos[:, 2:3] < 0.3, dim=1)
        # print(
        #     "self.base_pos: ",
        #     torch.sum(torch.any(self.base_pos[:, 2:3] < 0.3, dim=1)).item(),
        # )
        # print(
        #     "self.terminated: ",
        #     torch.sum(self.terminated).item(),
        # )
        # * No terminal reward for time-outs
        self.timed_out = self.episode_length_buf > self.max_episode_length
        # print(
        #     "self.timed_out: ",
        #     torch.sum(self.timed_out).item(),
        # )
        self.reset_buf = self.terminated | self.timed_out

    def post_physics_step(self):
        # print("post_physics_step")
        super().post_physics_step()
        # self._log_info()

    # * ########################## REWARDS ######################## * #

    # * Floating base Rewards * #

    def _reward_base_height(self):
        """Reward tracking specified base height"""
        error = (
            torch.abs(self.cfg.rewards.base_height_target - self.base_height)
            - self.cfg.rewards.base_height_range
        ).flatten()
        return self._negsqrd_exp(torch.clamp(error, 0))

    def _reward_base_heading(self):
        # Reward tracking desired base heading
        command_heading = torch.atan2(self.commands[:, 1], self.commands[:, 0])
        
        base_heading_error = torch.abs(
            wrap_to_pi(command_heading - self.base_heading.squeeze(1))
        )

        return self._neg_exp(base_heading_error, a=torch.pi / 2)

    def _reward_base_z_orientation(self):
        """Reward tracking upright orientation"""
        error = torch.norm(self.projected_gravity[:, :2], dim=1)
        return self._negsqrd_exp(error, a=0.2)

    def _reward_tracking_lin_vel_world(self):
        # Reward tracking linear velocity command in world frame
        error = self.commands[:, :2] - self.root_states[:, 7:9]
        error *= 1.0 / (1.0 + torch.abs(self.commands[:, :2]))
        return self._negsqrd_exp(error, a=1.0).sum(dim=1)

    # * Stepping Rewards * #

    def _reward_joint_regularization(self):
        # Reward joint poses and symmetry
        error = 0.0

        # Yaw joints regularization around 0
        error += self._negsqrd_exp((self.dof_pos[:, 1]) / self.scales["dof_pos"],0.1) # hip roll
        error += self._negsqrd_exp((self.dof_pos[:, 7]) / self.scales["dof_pos"],0.1) 

        error += self._negsqrd_exp((self.dof_pos[:, 2]) / self.scales["dof_pos"],0.1) # thigh yaw
        error += self._negsqrd_exp((self.dof_pos[:, 8]) / self.scales["dof_pos"],0.1) 

        error += self._negsqrd_exp((self.dof_pos[:, 5]) / self.scales["dof_pos"],0.1) # ankle roll
        error += self._negsqrd_exp((self.dof_pos[:, 11]) / self.scales["dof_pos"],0.1)

        return error / 6

    def _reward_contact_schedule(self):
        """Alternate right and left foot contacts
        First, right foot contacts (left foot swing), then left foot contacts (right foot swing)
        """
        contact_rewards = (
            self.foot_contact[:, 0].int() - self.foot_contact[:, 1].int()
        ) * self.contact_schedule.squeeze(1)
        k = 3.0
        a = 1.0
        tracking_rewards = k * self._neg_exp(
            self.step_location_offset[~self.foot_on_motion], a=a
        )
        return contact_rewards * tracking_rewards

    # * Other * #
    def _reward_feet_distance(self):
        """
        计算指定link间的距离（x和y方向的距离）作为奖励。这样做可以让腿尽可能的打开
        Calculates the reward based on the distance between the feet. Penilize feet get close to each other or too far away.
        """
        foot_pos = (
            self.rigid_body_state[:, self.feet_ids, :2]
            + self.rigid_body_state[:, self.feet_ids-1, :2]
        ) / 2
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist_feet
        max_df = self.cfg.rewards.max_dist_feet
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (
            torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)
        ) / 2

    def _reward_ankle_roll_posture_roll(self):
        feet_eular_0 = get_euler_xyz_tensor(
            self.rigid_body_state[:, self.feet_ids[0], 3:7]
        )[:, 0:1]
        feet_eular_1 = get_euler_xyz_tensor(
            self.rigid_body_state[:, self.feet_ids[1], 3:7]
        )[:, 0:1]
        # print(feet_eular_1.size())
        rew = torch.exp(
            -(torch.norm(feet_eular_0, dim=1) + torch.norm(feet_eular_1, dim=1)) * 40
        )
        # print(nn.size())
        return rew

    def _reward_ankle_roll_posture_pitch(self):
        feet_eular_0 = get_euler_xyz_tensor(
            self.rigid_body_state[:, self.feet_ids[0], 3:7]
        )[:, 1:2]
        feet_eular_1 = get_euler_xyz_tensor(
            self.rigid_body_state[:, self.feet_ids[1], 3:7]
        )[:, 1:2]
        # print(feet_eular_1.size())
        rew = torch.exp(
            -(torch.norm(feet_eular_0, dim=1) + torch.norm(feet_eular_1, dim=1)) * 5
        )
        # print(nn.size())
        return rew

    # def _reward_base_com(self):
    #     return torch.exp(-torch.norm(self.base_pos[:, :2] - self.CoM[:, :2],dim=1))
    

    
    # ##################### HELPER FUNCTIONS ################################## #

    def smooth_sqr_wave(self, phase):
        p = 2.0 * torch.pi * phase
        eps = 0.2
        return torch.sin(p) / torch.sqrt(torch.sin(p) ** 2.0 + eps**2.0)

    def _log_info(self):
        # print("_log_info")
        """Log any information for debugging"""
        self.extras["dof_vel"] = self.dof_vel
        self.extras["step_commands"] = self.step_commands
        self.extras["update_count"] = self.update_count

    def _visualization(self):
        # print("_visualization")
        self.gym.clear_lines(self.viewer)
        # self._draw_heightmap_vis()
        # self._draw_debug_vis()
        # self._draw_velocity_arrow_vis()
        self._draw_world_velocity_arrow_vis()
        # self._draw_base_pos_vis()
        self._draw_CoM_vis()
        self._draw_raibert_vis()
        self._draw_step_vis()
        self._draw_step_command_vis()

    def _draw_debug_vis(self):
        """Draws anything for debugging for humanoid"""
        sphere_origin = gymutil.WireframeSphereGeometry(
            0.02, 4, 4, None, color=(1, 1, 1)
        )
        origins = self.base_pos + quat_apply(
            self.base_quat,
            torch.tensor([0.0, 0.0, 0.5]).repeat(self.num_envs, 1).to(self.device),
        )

        for i in range(self.num_envs):
            env_origin = gymapi.Transform(gymapi.Vec3(*self.env_origins[i]), r=None)
            gymutil.draw_lines(
                sphere_origin, self.gym, self.viewer, self.envs[i], env_origin
            )

    def _draw_velocity_arrow_vis(self):
        """Draws linear / angular velocity arrow for humanoid
        Angular velocity is described by axis-angle representation"""
        origins = self.base_pos + quat_apply(
            self.base_quat,
            torch.tensor([0.0, 0.0, 0.5]).repeat(self.num_envs, 1).to(self.device),
        )
        lin_vel_command = quat_apply(
            self.base_quat,
            torch.cat(
                (
                    self.commands[:, :2],
                    torch.zeros((self.num_envs, 1), device=self.device),
                ),
                dim=1,
            )
            / 5,
        )
        ang_vel_command = quat_apply(
            self.base_quat,
            torch.cat(
                (
                    torch.zeros((self.num_envs, 2), device=self.device),
                    self.commands[:, 2:3],
                ),
                dim=1,
            )
            / 5,
        )
        for i in range(self.num_envs):
            lin_vel_arrow = VelCommandGeometry(
                origins[i], lin_vel_command[i], color=(0, 1, 0)
            )
            ang_vel_arrow = VelCommandGeometry(
                origins[i], ang_vel_command[i], color=(0, 1, 0)
            )
            gymutil.draw_lines(
                lin_vel_arrow, self.gym, self.viewer, self.envs[i], pose=None
            )
            gymutil.draw_lines(
                ang_vel_arrow, self.gym, self.viewer, self.envs[i], pose=None
            )

    def _draw_world_velocity_arrow_vis(self):
        """Draws linear / angular velocity arrow for humanoid
        Angular velocity is described by axis-angle representation"""
        origins = self.base_pos + quat_apply(
            self.base_quat,
            torch.tensor([0.0, 0.0, 0.5]).repeat(self.num_envs, 1).to(self.device),
        )
        lin_vel_command = (
            torch.cat(
                (
                    self.commands[:, :2],
                    torch.zeros((self.num_envs, 1), device=self.device),
                ),
                dim=1,
            )
            / 5
        )
        # ang_vel_command = quat_apply(self.base_quat, torch.cat((torch.zeros((self.num_envs,2), device=self.device), self.commands[:, 2:3]), dim=1)/5)
        for i in range(self.num_envs):
            lin_vel_arrow = VelCommandGeometry(
                origins[i], lin_vel_command[i], color=(0, 1, 0)
            )
            # ang_vel_arrow = VelCommandGeometry(origins[i], ang_vel_command[i], color=(0,1,0))
            gymutil.draw_lines(
                lin_vel_arrow, self.gym, self.viewer, self.envs[i], pose=None
            )
            # gymutil.draw_lines(ang_vel_arrow, self.gym, self.viewer, self.envs[i], pose=None)

    def _draw_base_pos_vis(self):
        sphere_base = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 1))
        sphere_left_hip = gymutil.WireframeSphereGeometry(
            0.02, 4, 4, None, color=(0, 0, 1)
        )
        sphere_right_hip = gymutil.WireframeSphereGeometry(
            0.02, 4, 4, None, color=(1, 0, 0)
        )

        base_projection = torch.cat(
            (self.base_pos[:, :2], torch.zeros((self.num_envs, 1), device=self.device)),
            dim=1,
        )
        right_hip_projection = torch.cat(
            (
                self.right_hip_pos[:, :2],
                torch.zeros((self.num_envs, 1), device=self.device),
            ),
            dim=1,
        )
        left_hip_projection = torch.cat(
            (
                self.left_hip_pos[:, :2],
                torch.zeros((self.num_envs, 1), device=self.device),
            ),
            dim=1,
        )
        for i in range(self.num_envs):
            base_loc = gymapi.Transform(gymapi.Vec3(*base_projection[i]), r=None)
            gymutil.draw_lines(
                sphere_base, self.gym, self.viewer, self.envs[i], base_loc
            )
            right_hip_loc = gymapi.Transform(
                gymapi.Vec3(*right_hip_projection[i]), r=None
            )
            gymutil.draw_lines(
                sphere_right_hip, self.gym, self.viewer, self.envs[i], right_hip_loc
            )
            left_hip_loc = gymapi.Transform(
                gymapi.Vec3(*left_hip_projection[i]), r=None
            )
            gymutil.draw_lines(
                sphere_left_hip, self.gym, self.viewer, self.envs[i], left_hip_loc
            )

    def _draw_CoM_vis(self):
        sphere_CoM = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 1))
        CoM_projection = torch.cat(
            (self.CoM[:, :2], torch.zeros((self.num_envs, 1), device=self.device)),
            dim=1,
        )
        for i in range(self.num_envs):
            CoM_loc = gymapi.Transform(gymapi.Vec3(*self.CoM[i, :3]), r=None)
            gymutil.draw_lines(sphere_CoM, self.gym, self.viewer, self.envs[i], CoM_loc)

    def _draw_raibert_vis(self):
        sphere_right_raibert = gymutil.WireframeSphereGeometry(
            0.02, 4, 4, None, color=(1, 0, 0)
        )
        sphere_left_raibert = gymutil.WireframeSphereGeometry(
            0.02, 4, 4, None, color=(0, 0, 1)
        )

        for i in range(self.num_envs):
            right_raibert_loc = gymapi.Transform(
                gymapi.Vec3(*self.raibert_heuristic[i, 0]), r=None
            )
            gymutil.draw_lines(
                sphere_right_raibert,
                self.gym,
                self.viewer,
                self.envs[i],
                right_raibert_loc,
            )

            left_raibert_loc = gymapi.Transform(
                gymapi.Vec3(*self.raibert_heuristic[i, 1]), r=None
            )
            gymutil.draw_lines(
                sphere_left_raibert,
                self.gym,
                self.viewer,
                self.envs[i],
                left_raibert_loc,
            )

    def _draw_step_vis(self):
        """Draws current foot steps for humanoid"""
        for i in range(self.num_envs):
            right_foot_step = FootStepGeometry(
                self.current_step[i, 0, :2], self.current_step[i, 0, 2], color=(1, 0, 1)
            )  # Right foot: Pink
            left_foot_step = FootStepGeometry(
                self.current_step[i, 1, :2], self.current_step[i, 1, 2], color=(0, 1, 1)
            )  # Left foot: Cyan
            gymutil.draw_lines(
                left_foot_step, self.gym, self.viewer, self.envs[i], pose=None
            )
            gymutil.draw_lines(
                right_foot_step, self.gym, self.viewer, self.envs[i], pose=None
            )

    def _draw_step_command_vis(self):
        """Draws step command for humanoid"""
        for i in range(self.num_envs):
            right_step_command = FootStepGeometry(
                self.step_commands[i, 0, :2],
                self.step_commands[i, 0, 2],
                color=(1, 0, 0),
            )  # Right foot: Red
            left_step_command = FootStepGeometry(
                self.step_commands[i, 1, :2],
                self.step_commands[i, 1, 2],
                color=(0, 0, 1),
            )  # Left foot: Blue
            gymutil.draw_lines(
                left_step_command, self.gym, self.viewer, self.envs[i], pose=None
            )
            gymutil.draw_lines(
                right_step_command, self.gym, self.viewer, self.envs[i], pose=None
            )


""" Code Explanation
0.
[Axis] X-axis: Red, Y-axis: Green, Z-axis: Blue

1.
self.base_pos = self.root_states[:, 0:3] : position of the base
self.base_quat = self.root_states[:, 3:7] : quaternion of the base
self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10]) : base linear velocity wrt base frame
self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13]) : base angular velocity wrt base frame

2.                                                    
quat_rotate_inverse() : World frame -> Base frame
quat_rotate(), quat_apply() : Base frame -> World frame

3.
self.rigid_body_state : [num_envs, num_bodies, 13] = [num_envs, 21, 13] 
[position | orientation (Quat) | linear velocity | angular velocity]

self._rigid_body_pos = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
self._rigid_body_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 7:10]

4.
21 bodies: base / right_hip_yaw / right_hip_abad / right_upper_leg / right_lower_leg / right_foot / left_hip_yaw / left_hip_abad / left_upper_leg / left_lower_leg / left_foot
                / right_shoulder / right_shoulder_2 / right_upper_arm / right_lower_arm / right_hand / left_shoulder / left_shoulder_2 / left_upper_arm / left_lower_arm / left_hand

right_foot[5] / left_foot[10] are end-effector

5.
self.dof_pos : joint position [num_envs, 10]       
self.dof_vel : joint velocity [num_envs, 10]                     
10 dof: 01_right_hip_yaw / 02_right_hip_abad / 03_right_hip_pitch / 04_right_knee / 05_right_ankle
        06_left_hip_yaw / 07_left_hip_abad / 08_left_hip_pitch / 09_left_knee / 10_left_ankle

6.
self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec) : gravity wrt base frame

7.
self.contact_forces : contact forces of each body parts [num_envs, num_bodies, 3] = [num_envs, 21, 3]
Contact forces are only measured when the collision body is defined. 

self.foot_contact : which foot (right & left foot) are in contact with ground [num_envs, 2]

8.
self.feet_ids: right_foot[5], left_foot[10]
self.end_eff_ids: right_foot[5], left_foot[10]

9.
Maximum reward we can get is "Max value of reward function * reward weight".
Since how it records the reward is "value * weight * dt  * (max_episode_length_s / dt) / max_episode_length_s = value * weight"
"""

""" TODO: 
1) Fix foot_reference_trajectory reward. It forces not to do sprint. 
Because the trajectory always start from the previous step command. Gradually increase the rewards.
2) Systematic training curriculum is necessary
"""
