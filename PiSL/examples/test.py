import numpy as np
import math
from scipy import io

# 加载数据
sim_measurement = io.loadmat('../systems/Double pendulum/double_pendulum_dynamics_X05.mat')['x'][:727, :]
num_frames = sim_measurement.shape[0]

# 定义物理参数
m1 = 35
m2 = 10
L1 = 9.1
L2 = 7
g = 981

# 使用最后的数据
last_state = sim_measurement[-1, :]  # 最后一个状态
x1, x2, w1, w2 = last_state  # 解包最后的状态值

# 创建一个包含当前变量值的字典
local_vars = {
    'm1': m1, 'm2': m2, 'L1': L1, 'L2': L2, 'g': g,
    'x1': x1, 'x2': x2, 'w1': w1, 'w2': w2
}

# 创建一个包含 math 模块的字典
global_vars = {
    'math': math
}

# 计算最后一步的角加速度
true_value_w1 = '(m2*L1*w1**2*math.sin(2*x1-2*x2) + 2*m2*L2*w2**2*math.sin(x1-x2) + 2*g*m2*math.cos(x2)*math.sin(x1-x2) + 2*g*m1*math.sin(x1))' + \
               '/ (-2*L1*(m2*math.sin(x1-x2)**2 + m1))'
true_value_w2 = '(m2*L2*w2**2*math.sin(2*x1-2*x2) + 2*(m1+m2)*L1*w1**2*math.sin(x1-x2) + 2*g*(m1+m2)*math.cos(x1)*math.sin(x1-x2))' + \
               '/ (2*L2*(m2*math.sin(x1-x2)**2 + m1))'

dw1 = eval(true_value_w1, global_vars, local_vars)
dw2 = eval(true_value_w2, global_vars, local_vars)

# 定义时间步长
dt = 0.01

# 定义预测下一步的函数
def predict_next_step(theta1, theta2, omega1, omega2, dw1, dw2, dt):
    new_omega1 = omega1 + dw1 * dt
    new_omega2 = omega2 + dw2 * dt
    new_theta1 = theta1 + new_omega1 * dt
    new_theta2 = theta2 + new_omega2 * dt
    return new_theta1, new_theta2, new_omega1, new_omega2

# 计算下一时刻的数据
new_theta1, new_theta2, new_omega1, new_omega2 = predict_next_step(x1, x2, w1, w2, dw1, dw2, dt)

# 打印下一时刻的数据
print("Next moment results:")
print(f"theta1: {new_theta1}, theta2: {new_theta2}, omega1: {new_omega1}, omega2: {new_omega2}")