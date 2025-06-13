import numpy as np
from kinematics import SixAxisArmKinematics
from visualization import ArmVisualizer
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 假設的DH參數（需根據實際機器手臂填寫）
dh_params = [
    # (theta, d, a, alpha)
    (0, 0.3, 0, np.pi/2),
    (0, 0, 0.6, 0),
    (0, 0, 0.5, 0),
    (0, 0.3, 0, np.pi/2),
    (0, 0, 0, -np.pi/2),
    (0, 0.15, 0, 0)
]

# 可自訂每個連桿的半徑（單位：米）
link_radii = [0.1, 0.08, 0.06, 0.04, 0.04, 0.03]
# 可自訂每個節點黑色圓柱體的半徑
joint_axis_radii = [0.025, 0.06, 0.06, 0.025, 0.025, 0.025]

kin = SixAxisArmKinematics(dh_params)
vis = ArmVisualizer(dh_params, link_radii=link_radii, joint_axis_radii=joint_axis_radii)

# 初始化六個關節角度
init_angles = np.zeros(6)
init_angles[1] = np.pi / 2  # 第二節點初始值設為 pi/2
joint_angles = init_angles.copy()

# 建立滑條介面
fig = vis.fig
axcolor = 'lightgoldenrodyellow'
slider_axes = []
sliders = []
for i in range(6):
    if i == 1:
        ax_slider = plt.axes([0.15, 0.02 + i*0.04, 0.65, 0.03], facecolor=axcolor)
        slider = Slider(ax_slider, f'Joint {i+1}', 0, np.pi, valinit=init_angles[i])
    else:
        ax_slider = plt.axes([0.15, 0.02 + i*0.04, 0.65, 0.03], facecolor=axcolor)
        slider = Slider(ax_slider, f'Joint {i+1}', -np.pi, np.pi, valinit=init_angles[i])
    sliders.append(slider)
    slider_axes.append(ax_slider)

def update(val):
    angles = np.array([slider.val for slider in sliders])
    vis.draw_arm(angles)

for slider in sliders:
    slider.on_changed(update)

# 初始繪製
vis.draw_arm(joint_angles)
plt.show()
