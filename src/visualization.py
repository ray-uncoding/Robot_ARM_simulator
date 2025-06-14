import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from kinematics import SixAxisArmKinematics

class ArmVisualizer:
    def __init__(self, dh_params, link_radii=None, joint_axis_radii=None):
        self.dh_params = dh_params
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.kin = SixAxisArmKinematics(dh_params)
        # 新增：每個連桿的半徑
        if link_radii is None:
            self.link_radii = [0.05] * (len(dh_params))
        else:
            self.link_radii = link_radii
        # 新增：每個關節黑色圓柱體半徑
        if joint_axis_radii is None:
            self.joint_axis_radii = [0.025] * (len(dh_params))
        else:
            self.joint_axis_radii = joint_axis_radii

    def draw_cylinder(self, p0, p1, radius=0.04, color='b', resolution=16):
        # p0, p1: 3D endpoints of the cylinder
        v = p1 - p0
        mag = np.linalg.norm(v)
        if mag == 0:
            return
        v = v / mag
        # Create a vector not parallel to v
        not_v = np.array([1, 0, 0]) if abs(v[0]) < 0.99 else np.array([0, 1, 0])
        n1 = np.cross(v, not_v)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(v, n1)
        t = np.linspace(0, mag, 2)
        theta = np.linspace(0, 2 * np.pi, resolution)
        t, theta = np.meshgrid(t, theta)
        X, Y, Z = [p0[i] + v[i] * t + radius * np.sin(theta) * n1[i] + radius * np.cos(theta) * n2[i] for i in range(3)]
        self.ax.plot_surface(X, Y, Z, color=color, alpha=0.8, linewidth=0, shade=True)

    def draw_gripper(self, p, direction, up=None, length=0.15, width=0.04, color='k'):
        """在末端加一個簡單的爪子（兩指），可指定up方向"""
        # direction: 爪子朝向（單位向量，通常為末端Z軸）
        # up: 爪子張開的法向（單位向量，通常為末端Y軸）
        if up is None or np.linalg.norm(up) < 1e-6:
            # 若未指定up，則自動找一個與direction不平行的向量
            if np.allclose(direction, [0, 0, 1]):
                ortho1 = np.array([1, 0, 0])
            else:
                ortho1 = np.cross(direction, [0, 0, 1])
                ortho1 /= np.linalg.norm(ortho1)
        else:
            ortho1 = up / np.linalg.norm(up)
        # 爪子張開方向
        ortho2 = np.cross(direction, ortho1)
        ortho2 /= np.linalg.norm(ortho2)
        base1 = p + width * ortho1
        base2 = p - width * ortho1
        tip1 = base1 + length * direction
        tip2 = base2 + length * direction
        self.ax.plot([base1[0], tip1[0]], [base1[1], tip1[1]], [base1[2], tip1[2]], color=color, linewidth=4)
        self.ax.plot([base2[0], tip2[0]], [base2[1], tip2[1]], [base2[2], tip2[2]], color=color, linewidth=4)

    def draw_joint_axis(self, p, axis, length=0.18, radius=0.025, color='k'):
        """在節點畫出沿旋轉軸向的圓柱體，表示可旋轉節點"""
        p0 = p - 0.5 * length * axis
        p1 = p + 0.5 * length * axis
        self.draw_cylinder(p0, p1, radius=radius, color=color)

    def draw_arm(self, joint_angles):
        self.ax.clear()
        positions = self.kin.get_joint_positions(joint_angles)
        xs, ys, zs = positions[:,0], positions[:,1], positions[:,2]
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        transforms = self.kin.get_joint_transforms(joint_angles)
        axes = [T[:3, 2] for T in transforms[:-1]]
        # 畫連桿
        for i in range(len(xs)-1):
            p0 = positions[i]
            p1 = positions[i+1]
            radius = self.link_radii[i] if i < len(self.link_radii) else 0.05
            self.draw_cylinder(p0, p1, radius=radius, color=colors[i%len(colors)])
            self.ax.scatter(*p0, color=colors[i%len(colors)], s=40)
        self.ax.scatter(*positions[-1], color=colors[-1], s=60)
        # 畫每個可旋轉節點的旋轉軸圓柱體，半徑可獨立調整
        for i in range(len(axes)):
            axis_radius = self.joint_axis_radii[i] if i < len(self.joint_axis_radii) else 0.025
            self.draw_joint_axis(positions[i], axes[i], radius=axis_radius, color='k')
        # 在末端加上爪子，方向需考慮第六軸旋轉
        if len(positions) >= 2:
            gripper_dir = transforms[-1][:3, 2]  # 末端Z軸
            gripper_up = transforms[-1][:3, 1]   # 末端Y軸
            norm_dir = np.linalg.norm(gripper_dir)
            norm_up = np.linalg.norm(gripper_up)
            if norm_dir > 1e-6 and norm_up > 1e-6:
                gripper_dir = gripper_dir / norm_dir
                gripper_up = gripper_up / norm_up
                self.draw_gripper(positions[-1], gripper_dir, up=gripper_up)
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(0, 2)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.fig.canvas.draw_idle()
