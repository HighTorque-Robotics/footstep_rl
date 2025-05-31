
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Deployment script.")
parser.add_argument(
    "--show_what", type=str, required=True, help="what kind of data would be plot.",\
    default="pos"
)
args = parser.parse_args()



# 读取 CSV 文件
df = pd.read_csv('sim2sim_robot_states.csv')

# 提取目标关节位置的列名
target_dof_columns = [f'sim2sim_target_dof_pos_{i}' for i in range(12)]
state_dof_columns = [f'sim2sim_dof_pos_{i}' for i in range(12)]
state_tau_columns = [f'sim2sim_state_dof_tau_{i}' for i in range(12)]
target_tau_columns = [f'sim2sim_target_dof_tau_{i}' for i in range(12)]

# 提取数据
target_dof_data = df[target_dof_columns]
# 设置绘图风格
plt.style.use('seaborn-darkgrid')

# 创建 6 行 2 列的子图
fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(15, 20))
axes = axes.flatten()

# 为每个关节绘制曲线图
for i, cols in enumerate(zip(target_dof_columns,state_dof_columns,state_tau_columns,target_tau_columns)):
    if args.show_what == "pos":
        axes[i].plot(df.index, df[cols[0]], label=f'Target DOF {i}')
        axes[i].plot(df.index, df[cols[1]], label=f'state DOF {i}')
        axes[i].set_title(f'Target & state DOF pos {i}')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('position')
        axes[i].legend()
    if args.show_what == "tau":
        axes[i].plot(df.index, df[cols[2]], label=f'state DOF tau {i}', color='r')
        axes[i].plot(df.index, df[cols[3]], label=f'target DOF tau {i}', color='b')
        axes[i].set_title(f'Target & state DOF tau {i}')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('tau')


plt.tight_layout()
plt.show()
