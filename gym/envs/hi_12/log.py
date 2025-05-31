import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from gym.envs.hi_12.hi_controller_config import HiControllerCfg
from gym.utils.math import *
from gym.envs import LeggedRobot
from isaacgym import gymapi, gymutil
import numpy as np
from typing import Tuple, Dict
from .hi_utils import (
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


class HiControllerIndependent(LeggedRobot):
    cfg: HiControllerCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _setup_keyboard_interface(self):
        self.keyboard_interface = XCoMKeyboardInterface(self)

    def _init_buffers(self):
        super()._init_buffers()
        
        # [NEW] Independent control buffers
        self.linear_velocity = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )  # Linear velocity command [vx, vy]
        self.angular_velocity = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )  # Angular velocity command [yaw_rate]
        self.desired_heading = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )  # Desired robot heading
        
        # Original buffers
        self.base_height = self.root_states[:, 2:3]
        # ... [其他原始buffer初始化代码保持不变]

    def _generate_step_command_by_3DLIPM_XCoM_independent(self, update_commands_ids):
        """Modified step command generation for independent control of translation and rotation
        Separates the handling of linear and angular motion
        """
        # Get basic parameters
        foot_on_motion = self.foot_on_motion[update_commands_ids]
        step_period = self.step_period[update_commands_ids]
        linear_vel = self.linear_velocity[update_commands_ids]  # [NEW] Use linear velocity directly
        angular_vel = self.angular_velocity[update_commands_ids]  # [NEW] Use angular velocity directly
        current_step = self.current_step[update_commands_ids]
        CoM = self.CoM[update_commands_ids]
        
        # Calculate time and frequency parameters
        T = step_period * self.dt
        w = self.w[update_commands_ids]
        
        # [NEW] Calculate step parameters separately for translation and rotation
        # Translation
        dstep_length = torch.norm(linear_vel, dim=1, keepdim=True) * T
        translation_theta = torch.atan2(linear_vel[:, 1:2], linear_vel[:, 0:1])
        
        # [NEW] Rotation
        rotation_step = angular_vel * T  # Simple proportional mapping
        
        # Rest of the function remains similar but uses separated parameters
        # ... [原有的3D-LIPM计算代码]
        
        # [NEW] Combine translation and rotation effects
        random_step_command = torch.zeros(
            foot_on_motion.sum(),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        
        # [NEW] Set translation component
        random_step_command[:, :2] = torch.cat((u_x, u_y), dim=1)
        
        # [NEW] Add rotation component
        random_step_command[:, 2] = translation_theta.squeeze(1) + rotation_step.squeeze(1)
        
        return random_step_command

    def _update_command_curriculum(self, env_ids):
        """Updates the command ranges for curriculum learning
        Modified to handle independent control parameters
        """
        # [NEW] Separate updates for linear and angular velocities
        linear_vel_indices = torch.nonzero(torch.rand(len(env_ids)) < 0.5).squeeze(-1)
        angular_vel_indices = torch.nonzero(torch.rand(len(env_ids)) < 0.3).squeeze(-1)
        
        if len(linear_vel_indices) > 0:
            self.linear_velocity[env_ids[linear_vel_indices]] = torch_rand_float(
                self.command_ranges["lin_vel_x"][0],
                self.command_ranges["lin_vel_x"][1],
                (len(linear_vel_indices), 2),
                device=self.device
            )
            
        if len(angular_vel_indices) > 0:
            self.angular_velocity[env_ids[angular_vel_indices]] = torch_rand_float(
                self.command_ranges["ang_vel"][0],
                self.command_ranges["ang_vel"][1],
                (len(angular_vel_indices), 1),
                device=self.device
            )

    # [NEW] Helper function for command conversion
    def _convert_independent_to_combined_commands(self):
        """Converts independent linear and angular velocities to combined commands format"""
        self.commands[:, 0] = self.linear_velocity[:, 0]  # vx
        self.commands[:, 1] = self.linear_velocity[:, 1]  # vy
        self.commands[:, 2] = self.angular_velocity[:, 0]  # yaw_rate

    def _update_commands(self):
        """Update step commands
        Modified to handle independent control
        """
        # Convert independent commands to combined format for compatibility
        self._convert_independent_to_combined_commands()
        
        # Update phase
        self.update_phase_ids = self.phase_count >= self.full_step_period.squeeze(1)
        self.phase_count[self.update_phase_ids] = 0
        self.phase[self.update_phase_ids] = 0
        
        # Update commands
        self.update_commands_ids = self.update_count >= self.step_period.squeeze(1)
        self.already_succeed_step[self.update_commands_ids] = False
        self.had_wrong_contact[self.update_commands_ids] = False
        self.update_count[self.update_commands_ids] = 0
        
        # Update foot motion
        self.foot_on_motion[self.update_commands_ids] = ~self.foot_on_motion[self.update_commands_ids]
        
        # Generate and update step commands using independent control
        update_step_commands_mask = self.step_commands[self.update_commands_ids]
        self.prev_step_commands[self.update_commands_ids] = torch.clone(self.step_commands[self.update_commands_ids])
        
        # Use the new independent step command generation
        update_step_commands_mask[self.foot_on_motion[self.update_commands_ids]] = (
            self._generate_step_command_by_3DLIPM_XCoM_independent(self.update_commands_ids)
        )
        
        # Handle foot collision
        foot_collision_ids = (
            update_step_commands_mask[:, 0, :2] - update_step_commands_mask[:, 1, :2]
        ).norm(dim=1) < 0.1
        
        if foot_collision_ids.any():
            update_step_commands_mask[foot_collision_ids, :, :2] = (
                self._adjust_foot_collision(
                    update_step_commands_mask[foot_collision_ids, :, :2],
                    self.foot_on_motion[self.update_commands_ids][foot_collision_ids],
                )
            )
        
        self.step_commands[self.update_commands_ids] = update_step_commands_mask