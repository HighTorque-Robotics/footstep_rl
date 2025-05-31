# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R

from legged_gym import LEGGED_GYM_ROOT_DIR

# from legged_gym.envs.pai.pai_config import PaiRoughCfg
from legged_gym.envs.pai_none_phase.pai_config_none_phase import PaiNonePhaseCfg as cfg
from isaacgym.torch_utils import *

import torch

import csv
import pandas as pd
import threading
import queue
import pygame
import time


class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0


class env:
    obs = 48
    num_single_obs = obs
    frame_stack = 31
    obs_his = obs * 30


def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def quat_rotate_inverse_ori(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = (
        q_vec
        * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1)
        * 2.0
    )
    return a - b + c


def get_obs(data):
    """Extracts an observation from the mujoco data structure"""
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    # print("p:", (target_q - q) * kp )
    # print("d", (target_dq - dq) * kd)
    return (target_q - q) * kp + (target_dq - dq) * kd


def process_value(value, threshold=0.01):
    """Process the input value, setting it to zero if below a threshold."""
    return 0 if abs(value) < threshold else value


def run_mujoco(control_queue: queue.Queue, policy, cfg: cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    obs_history = torch.zeros(1, env.obs_his, dtype=torch.float)

    hist_obs = deque()
    for _ in range(env.frame_stack):
        hist_obs.append(np.zeros([1, env.num_single_obs], dtype=np.double))

    count_lowlevel = 0
    count_csv = 0
    clip = []
    for joint, vals in cfg.init_state.dof_pos_range_virtual.items():
        print(f"joint: {joint} vals: {vals}")
        clip.append(vals)
    cl = np.array(clip, dtype=np.float32)
    phase = 0
    step_period = cfg.commands.ranges.sample_period[0]
    command = [0, 0, 0]  # vx, vy, dyaw
    for _ in tqdm(
        range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)),
        desc="Simulating...",
    ):
        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions :]
        dq = dq[-cfg.env.num_actions :]

        for i in range(6):
            tmpq = q[i]
            q[i] = q[i + 6]
            q[i + 6] = tmpq

            tmpdq = dq[i]
            dq[i] = dq[i + 6]
            dq[i + 6] = tmpdq

        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:
            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            """
            obs[0, 0] = math.sin(
                2 * math.pi * count_lowlevel * cfg.sim_config.dt / 0.64
            )
            obs[0, 1] = math.cos(
                2 * math.pi * count_lowlevel * cfg.sim_config.dt / 0.64
            )
            obs[0, 2] = cmd.vx * cfg.normalization.obs_scales.lin_vel
            obs[0, 3] = cmd.vy * cfg.normalization.obs_scales.lin_vel
            obs[0, 4] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
            obs[0, 5:17] = q * cfg.normalization.obs_scales.dof_pos
            obs[0, 17:29] = dq * cfg.normalization.obs_scales.dof_vel
            obs[0, 29:41] = action
            obs[0, 41:44] = omega
            obs[0, 44:47] = eu_ang
            """
            
            full_step_period = step_period * 2
            phase += 1/full_step_period
            obs[0, 0] = math.sin(
                2 * math.pi * phase
            )
            obs[0, 1] = math.cos(
                2 * math.pi * phase
            )
            obs[0, 2] = cmd.vx * cfg.scaling.commands
            obs[0, 3] = cmd.vy * cfg.scaling.commands
            obs[0, 4] = cmd.dyaw * cfg.scaling.commands
            obs[0, 5:17] = q * cfg.scaling.dof_pos
            obs[0, 17:29] = dq * cfg.scaling.dof_vel
            # obs[0, 29:41] = action
            obs[0, 29:32] = omega * cfg.scaling.base_ang_vel
            obs[0, 32:35] = eu_ang
            
            obs = np.clip(
                obs,
                -20,
                20,
            )
            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                policy_input[
                    0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs
                ] = hist_obs[i][0, :]
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(
                action,
                -cfg.scaling.clip_actions,
                cfg.scaling.clip_actions,
            )



            target_q = action \
                    * 0.5

            policy_input = np.zeros(
                [1, int(env.frame_stack * env.num_single_obs)], dtype=np.float32
            )
            print("policy_input.shape",policy_input.shape)
            for i in range(env.frame_stack):
                policy_input[
                    0, i * env.num_single_obs : (i + 1) * env.num_single_obs
                ] = hist_obs[i][0, :]
            _action = policy(torch.tensor(policy_input))
            # _action,mean_vel = policy(torch.tensor(policy_input))
            # print("action:\n",_action)
            action[:] = _action[0].detach().numpy()
            # obs_history长度为47*5 ，在给入网络之后再更新
            # action[:] = load_policy(logdir,obs,obs_history)[0].detach().numpy()
            # obs_history = torch.cat((obs_history[:,env.obs:], obs[:,:]), dim=-1)

            action = np.clip(
                action,
                -cfg.normalization.clip_actions,
                cfg.normalization.clip_actions,
            )
            target_q = action * cfg.control.action_scale

        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        # print("==============================")
        # print(target_q)
        target_q = np.clip(target_q, cl[:, 0], cl[:, 1])
        # print(target_q)
        # Generate PD control
        tau = pd_control(
            target_q, q, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds
        )  # Calc torques
        tau = np.clip(
            tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit
        )  # Clamp torques
        for i in range(6):
            tmptau = tau[i]
            tau[i] = tau[i + 6]
            tau[i + 6] = tmptau
        data.ctrl = tau
        # print(tau)

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()


class GamepadHandler:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        self.joystick_count = pygame.joystick.get_count()
        self.joysticks = []

        for i in range(self.joystick_count):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            self.joysticks.append(joystick)
            print(f"Initialized Joystick {i}: {joystick.get_name()}")

    def process_events(self, command_queue):
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                axis_values = [
                    self.joysticks[0].get_axis(1),
                    self.joysticks[0].get_axis(0),
                    self.joysticks[0].get_axis(3),
                ]
                command_queue.put(axis_values)
            elif event.type == pygame.QUIT:
                return False
        return True

    def quit(self):
        pygame.quit()


def gamepad_input(control_queue):
    handler = GamepadHandler()
    running = True
    while running:
        running = handler.process_events(control_queue)
        time.sleep(0.00001)  # Adjust polling rate if necessary
    handler.quit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument(
        "--logdir",
        type=str,
        required=False,
        default=f"{LEGGED_GYM_ROOT_DIR}/logs/pai_none_phase/exported/policies",
        help="Run to load from.",
    )
    parser.add_argument("--terrain", action="store_true", help="terrain or plane")
    args = parser.parse_args()

    class Sim2simCfg(cfg):
        class sim_config:
            # mujoco_model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/pi_12dof_release_v1/mjcf/pi_12dof_release_v2_fixedbase.xml"  # 平地
            mujoco_model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/pi_12dof_release_v1/mjcf/pi_12dof_release_v2.xml"  # 平地
            # mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/pi_12dof_release_v1/mjcf/pi_12dof_release_v1_hfield_l1.xml' #hfield
            # mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/pi_12dof_release_v1/mjcf/pi_12dof_release_v1_hfield.xml' #hfield

            # mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/pi_12dof_release_v1/mjcf/pi_12dof_release_v1_slope.xml' #hfield

            sim_duration = 60.0
            dt = 0.001
            decimation = 10

        class robot_config:
            kps_l = []
            kds_l = []
            for joint, vals in cfg.control.stiffness.items():
                print(f"joint: {joint} vals: {vals}")
                kps_l.append(vals)

            for joint, vals in cfg.control.damping.items():
                print(f"joint: {joint} vals: {vals}")
                kds_l.append(vals)
            kps = np.array(kps_l * (2), dtype=np.float32)
            kds = np.array(kds_l * (2), dtype=np.float32)
            print("kps: ", kps)
            print("kds: ", kds)
            # kps = np.array(
            #     [40, 30, 10, 40, 20, 10, 20, 10, 10, 20, 20, 10], dtype=np.double
            # )  # v7

            # kds = np.array(
            #     [2, 1.6, 1, 2, 2, 1, 2, 1.6, 1, 2, 2, 1],
            #     dtype=np.double,
            # )

            tau_limit = 10.0 * np.ones(12, dtype=np.double)

    a = args.logdir + "/combined_model_dwaq.pt"
    policy = torch.jit.load(a)
    # run_mujoco(policy, Sim2simCfg())

    control_queue = queue.Queue()

    # Thread for MuJoCo simulation
    thread_a = threading.Thread(
        target=run_mujoco, args=(control_queue, policy, Sim2simCfg())
    )
    thread_a.start()

    # Thread for gamepad input
    thread_b = threading.Thread(target=gamepad_input, args=(control_queue,))
    thread_b.start()

    # Wait for threads to complete
    thread_a.join()
    thread_b.join()
