import pybullet as p
import time

# 启动PyBullet仿真（可选择GUI显示）
p.connect(p.GUI)  # 使用图形界面（p.DIRECT 不显示图形界面）

# 设置物理环境（例如重力）
p.setGravity(0, 0, -9.81)

# 加载URDF文件（替换成你自己的URDF文件路径）
robot_id = p.loadURDF("DOGlove_meshes/DOGlove_Ver20241223.urdf", basePosition=[0, 0, 0.1])

# 在仿真中保持一段时间，以便查看
for _ in range(1000):
    p.stepSimulation()  # 进行一步仿真
    time.sleep(1./240.)  # 控制仿真步伐的时间（通常为240Hz）

# 断开PyBullet连接
p.disconnect()
