from mujoco_py import load_model_from_path, MjSim

model = load_model_from_path("/home/wuchenxi/project/mujoco_files/ant_reacher.xml")  # 加载 Ant Reacher 模型
sim = MjSim(model)

# 获取 action 维度
action_dim = len(sim.model.actuator_ctrlrange)
print(f"Action Dimension: {action_dim}")  # 输出：8
