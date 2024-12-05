# import gym
# import os
# os.environ["MUJOCO_GL"] = "offscreen"

# env = gym.make("Pendulum-v0")
# obs = env.reset()
# done = False
# while (not done):
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     env.render()  # 渲染当前环境状态

from pyvirtualdisplay import Display
import gym

# 启动虚拟显示
display = Display(visible=0, size=(1400, 900))
display.start()

# 创建环境
env = gym.make('Pendulum-v1')

# 渲染环境
env.render()
