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
from gym.envs import LEGGED_GYM_ROOT_DIR
from gym.envs import Hicl12ControllerCfg
import torch
import time
import csv
import pandas as pd

import os


class cmd:
    vx = -0.5
    vy = 0.
    dyaw = 0


def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])



def get_obs(data):
    """Extracts an observation from the mujoco data structure"""
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    state_tau = data.qfrc_actuator.astype(np.double)-data.qfrc_bias.astype(np.double)
    return (q, dq, quat, v, omega, gvec,state_tau)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    # print("p:", (target_q - q) * kp )
    # print("d", (target_dq - dq) * kd)
    return (target_q - q) * kp + (target_dq - dq) * kd


def run_mujoco(policy, cfg:Hicl12ControllerCfg):
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

    nq = model.nq  # 关节位置的自由度
    nv = model.nv  # 关节速度的自由度

    # 打印自由度信息
    print(f"Model has {nq} position DOFs and {nv} velocity DOFs.")
    # aa = [-0.6, 0.0, 0.0, 1.2, -0.6, 0.0]
    # a = aa + aa
    # # data.qpos[:nq] = np.array(
    # #     [0, 0, 0.0] + [1, 0, 0, 0] + [0.0] * (12)
    # # )  # 根据你的模型调整
    # data.qpos[:nq] = np.array([0 , 0, 0.]+[1,0,0,0]+a)  # 根据你的模型调整
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    cont = 0 * 1000
    while cont:
        viewer.render()
        print(cont)
        cont -= 1
    target_q = np.zeros((cfg.env.num_actuators), dtype=np.double)
    action = np.zeros((cfg.env.num_actuators), dtype=np.double)

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 0

    count_csv = 0
    with open("sim2sim_robot_states.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        # csvwriter.writerow([f'q_{i}' for i in range(19)])
        csvwriter.writerow(
            [
                "sim2sim_base_euler_roll",
                "sim2sim_base_euler_pitch",
                "sim2sim_base_euler_yaw",
                # "sim2sim_base_quat_x", "sim2sim_base_quat_y", "sim2sim_base_quat_z", "sim2sim_base_quat_w",
                "sim2sim_dof_pos_0",
                "sim2sim_dof_pos_1",
                "sim2sim_dof_pos_2",
                "sim2sim_dof_pos_3",
                "sim2sim_dof_pos_4",
                "sim2sim_dof_pos_5",
                "sim2sim_dof_pos_6",
                "sim2sim_dof_pos_7",
                "sim2sim_dof_pos_8",
                "sim2sim_dof_pos_9",
                "sim2sim_dof_pos_10",
                "sim2sim_dof_pos_11",
                "sim2sim_target_dof_pos_0",
                "sim2sim_target_dof_pos_1",
                "sim2sim_target_dof_pos_2",
                "sim2sim_target_dof_pos_3",
                "sim2sim_target_dof_pos_4",
                "sim2sim_target_dof_pos_5",
                "sim2sim_target_dof_pos_6",
                "sim2sim_target_dof_pos_7",
                "sim2sim_target_dof_pos_8",
                "sim2sim_target_dof_pos_9",
                "sim2sim_target_dof_pos_10",
                "sim2sim_target_dof_pos_11",
                "sim2sim_target_dof_pos_10",
                "sim2sim_target_dof_pos_11",
                "sim2sim_state_dof_tau_0",
                "sim2sim_state_dof_tau_1",
                "sim2sim_state_dof_tau_2",
                "sim2sim_state_dof_tau_3",
                "sim2sim_state_dof_tau_4",
                "sim2sim_state_dof_tau_5",
                "sim2sim_state_dof_tau_6",
                "sim2sim_state_dof_tau_7",
                "sim2sim_state_dof_tau_8",
                "sim2sim_state_dof_tau_9",
                "sim2sim_state_dof_tau_10",
                "sim2sim_state_dof_tau_11",
                "sim2sim_target_dof_tau_0",
                "sim2sim_target_dof_tau_1",
                "sim2sim_target_dof_tau_2",
                "sim2sim_target_dof_tau_3",
                "sim2sim_target_dof_tau_4",
                "sim2sim_target_dof_tau_5",
                "sim2sim_target_dof_tau_6",
                "sim2sim_target_dof_tau_7",
                "sim2sim_target_dof_tau_8",
                "sim2sim_target_dof_tau_9",
                "sim2sim_target_dof_tau_10",
                "sim2sim_target_dof_tau_11",
            ]
        )
        
        phase = 0
        step_period = 25#cfg.commands.ranges.sample_period[0]
        
        for _ in tqdm(
            range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)),
            desc="Simulating...",
        ):
            # Obtain an observation
            q, dq, quat, v, omega, gvec,state_tau = get_obs(data)

            q = q[-cfg.env.num_actuators :]
            dq = dq[-cfg.env.num_actuators :]
            state_tau = state_tau[-cfg.env.num_actuators :]

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
                     * 1.0
                    #  * cfg.control.action_scale

                if count_csv < 500:
                    csv_q = np.zeros(27)
                    csv_euler_ang = quaternion_to_euler_array(quat)
                    # csv_euler_ang = quaternion_to_euler_array(q[3:7])
                    csv_q[0:3] = csv_euler_ang
                    csv_q[3:15] = q[:]
                    csv_q[15:] = target_q[:]
                    csvwriter.writerow(csv_q.tolist())
                count_csv += 1

            target_dq = np.zeros((cfg.env.num_actuators), dtype=np.double)

            # Generate PD control
            # new_q = [x + y for x, y in zip(q, a)]
            # array2 = np.array(a)
            # new_q = q-array2
            # print(q.shape,new_q)
            tau = pd_control(
                target_q, q, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds
            )  # Calc torques
            tau = np.clip(
                tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit
            )  # Clamp torques

            if count_csv < 5000:
                csv_q = np.zeros(51)
                csv_euler_ang = quaternion_to_euler_array(quat)
                # csv_euler_ang = quaternion_to_euler_array(q[3:7])
                csv_q[0:3] = csv_euler_ang
                csv_q[3:15] = q[:]
                csv_q[15:27] = target_q[:]
                csv_q[27:39] = state_tau[:]
                csv_q[39:51] = tau[:]
                csvwriter.writerow(csv_q.tolist())
                count_csv += 1
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument(
        "--load_model", type=str, required=False, help="Run to load from.",\
        default="/home/hpx/HPXLoco/ModelBasedFootstepPlanning-IROS2024/logs/Hicl12_Controller/exported/policy.pt"
    )
    parser.add_argument("--terrain", action="store_true", help="terrain or plane")
    args = parser.parse_args()

    class Sim2simCfg(Hicl12ControllerCfg):

        class sim_config:
            if args.terrain:
                mujoco_model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L-terrain.xml"
            else:
                # mujoco_model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/clpai_12dof_0905/mjcf/pai_12dof.xml"
                mujoco_model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/hi_cl_23_240925/mjcf/hi_12dof.xml"
                
            sim_duration = 60.0
            dt = 0.001
            decimation = 10

        class robot_config:
            kps = np.ones(12, dtype=np.double) * 30
            kds = np.ones(12, dtype=np.double) * 1
            print(kds)
            tau_limit = 36.0 * np.ones(12, dtype=np.double)

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
