# -*-coding:UTF-8 -*-
import numpy as np

# Deinition of 3D Linear Inverted Pendulum
class LIPM3D:
    def __init__(self,
                 dt=0.001,
                 T=1.0,
                 T_d = 0.4,
                 s_d = 0.5,
                 w_d = 0.4,
                 support_leg='left_leg'):
        self.dt = dt
        self.t = 0
        self.T = T # step time 步态周期
        self.T_d = T # desired step duration 期望步态周期时间
        self.s_d = s_d # desired step length 期望步长
        self.w_d = w_d # desired step width 期望步宽

        self.eICP_x = 0 # ICP location x ICP位置 x
        self.eICP_y = 0 # ICP location y ICP位置 y
        self.u_x = 0 # step location x 步态位置 x
        self.u_y = 0 # step location y 步态位置 y

        # COM initial state COM初始位置
        self.x_0 = 0
        self.vx_0 = 0
        self.y_0 = 0
        self.vy_0 = 0

        # COM real-time state 实时状态
        self.x_t = 0
        self.vx_t = 0
        self.y_t = 0
        self.vy_t = 0

        self.support_leg = support_leg
        self.support_foot_pos = [0.0, 0.0, 0.0]
        self.left_foot_pos = [0.0, 0.0, 0.0]
        self.right_foot_pos = [0.0, 0.0, 0.0]
        self.COM_pos = [0.0, 0.0, 0.0]
    
    def initializeModel(self, COM_pos, left_foot_pos, right_foot_pos):
        # 初始化了模型的 CoM 位置和支撑脚位置，并计算了摆的自然频率 w_0
        self.COM_pos = COM_pos

        if self.support_leg == 'left_leg':
            self.left_foot_pos = left_foot_pos
            self.right_foot_pos = left_foot_pos
            self.support_foot_pos = left_foot_pos
        elif self.support_leg == 'right_leg':
            self.left_foot_pos = right_foot_pos
            self.right_foot_pos = right_foot_pos
            self.support_foot_pos = right_foot_pos

        self.zc = self.COM_pos[2]
        self.w_0 = np.sqrt(9.81 / self.zc)
    
    def step(self):
        # 计算了当前时间步长下 CoM 的位置和速度。它使用了 LIPM 的解析解，其中 cosh 和 sinh 函数用于计算 CoM 的位置和速度
        self.t += self.dt
        t = self.t

        self.x_t = self.x_0*np.cosh(t*self.w_0) + self.vx_0*np.sinh(t*self.w_0)/self.w_0
        self.vx_t = self.x_0*self.w_0*np.sinh(t*self.w_0) + self.vx_0*np.cosh(t*self.w_0)

        self.y_t = self.y_0*np.cosh(t*self.w_0) + self.vy_0*np.sinh(t*self.w_0)/self.w_0
        self.vy_t = self.y_0*self.w_0*np.sinh(t*self.w_0) + self.vy_0*np.cosh(t*self.w_0)

    def calculateXfVf(self):
        # 方法计算了步态周期结束时 CoM 的位置和速度。
        x_f = self.x_0*np.cosh(self.T*self.w_0) + self.vx_0*np.sinh(self.T*self.w_0)/self.w_0
        vx_f = self.x_0*self.w_0*np.sinh(self.T*self.w_0) + self.vx_0*np.cosh(self.T*self.w_0)

        y_f = self.y_0*np.cosh(self.T*self.w_0) + self.vy_0*np.sinh(self.T*self.w_0)/self.w_0
        vy_f = self.y_0*self.w_0*np.sinh(self.T*self.w_0) + self.vy_0*np.cosh(self.T*self.w_0)

        return x_f, vx_f, y_f, vy_f

    def calculateFootLocationForNextStepXcoMWorld(self, theta=0.):
        # 计算了下一步的位置。它首先计算 CoM 在步态周期结束时的位置和速度，然后计算期望的捕获点（eICP）。
        # 接着，它计算步态位置的偏移量，并将其应用于 eICP 以确定下一步的位置
        x_f, vx_f, y_f, vy_f = self.calculateXfVf()
        x_f_world = x_f + self.support_foot_pos[0]
        y_f_world = y_f + self.support_foot_pos[1]
        self.eICP_x = x_f_world + vx_f/self.w_0
        self.eICP_y = y_f_world + vy_f/self.w_0
        b_x = self.s_d / (np.exp(self.w_0*self.T_d) - 1)
        b_y = self.w_d / (np.exp(self.w_0*self.T_d) + 1)

        original_offset_x = -b_x
        original_offset_y = -b_y if self.support_leg == "left_leg" else b_y 
        offset_x = np.cos(theta) * original_offset_x - np.sin(theta) * original_offset_y
        offset_y = np.sin(theta) * original_offset_x + np.cos(theta) * original_offset_y

        self.u_x = self.eICP_x + offset_x
        self.u_y = self.eICP_y + offset_y

    def calculateFootLocationForNextStepXcoMBase(self, theta=0.):
        x_f, vx_f, y_f, vy_f = self.calculateXfVf()
        x_f_world = x_f + self.support_foot_pos[0]
        y_f_world = y_f + self.support_foot_pos[1]
        self.eICP_x = x_f_world + vx_f/self.w_0
        self.eICP_y = y_f_world + vy_f/self.w_0
        b_x = self.s_d / (np.exp(self.w_0*self.T_d) - 1)
        b_y = self.w_d / (np.exp(self.w_0*self.T_d) + 1)

        original_offset_x = -b_x
        original_offset_y = -b_y if self.support_leg == "left_leg" else b_y 
        offset_x = np.cos(theta) * original_offset_x - np.sin(theta) * original_offset_y
        offset_y = np.sin(theta) * original_offset_x - np.cos(theta) * original_offset_y

        self.u_x = self.eICP_x + offset_x
        self.u_y = self.eICP_y + offset_y

    def switchSupportLeg(self):
        # 在步态周期结束时切换支撑脚，并更新 CoM 的初始状态。
        if self.support_leg == 'left_leg':
            print('\n---- switch the support leg to the right leg')
            self.support_leg = 'right_leg'
            COM_pos_x = self.x_t + self.left_foot_pos[0]
            COM_pos_y = self.y_t + self.left_foot_pos[1]
            self.x_0 = COM_pos_x - self.right_foot_pos[0]
            self.y_0 = COM_pos_y - self.right_foot_pos[1]
            self.support_foot_pos = self.right_foot_pos
        elif self.support_leg == 'right_leg':
            print('\n---- switch the support leg to the left leg')
            self.support_leg = 'left_leg'
            COM_pos_x = self.x_t + self.right_foot_pos[0]
            COM_pos_y = self.y_t + self.right_foot_pos[1]
            self.x_0 = COM_pos_x - self.left_foot_pos[0]
            self.y_0 = COM_pos_y - self.left_foot_pos[1]
            self.support_foot_pos = self.left_foot_pos

        self.t = 0
        self.vx_0 = self.vx_t
        self.vy_0 = self.vy_t
