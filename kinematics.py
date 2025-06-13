import numpy as np

class SixAxisArmKinematics:
    def __init__(self, dh_params):
        """
        dh_params: list of Denavit-Hartenberg parameters for each joint
        """
        self.dh_params = dh_params

    def forward_kinematics(self, joint_angles):
        """
        joint_angles: list or np.array of 6 joint angles (radians)
        return: 4x4 np.array, end-effector pose
        """
        # TODO: implement FK using DH parameters
        pass

    def inverse_kinematics(self, target_pose):
        """
        target_pose: 4x4 np.array, desired end-effector pose
        return: list of 6 joint angles (radians)
        """
        # TODO: implement IK (can use numerical or analytical method)
        pass

    def get_joint_positions(self, joint_angles):
        """
        joint_angles: list or np.array of 6 joint angles (radians)
        return: list of 3D positions (x, y, z) for each joint (including base and end-effector)
        """
        n = len(self.dh_params)
        T = np.eye(4)
        positions = [T[:3, 3].copy()]
        for i in range(n):
            theta, d, a, alpha = self.dh_params[i]
            th = joint_angles[i] + theta
            ct, st = np.cos(th), np.sin(th)
            ca, sa = np.cos(alpha), np.sin(alpha)
            A = np.array([
                [ct, -st*ca,  st*sa, a*ct],
                [st,  ct*ca, -ct*sa, a*st],
                [0,      sa,     ca,    d],
                [0,       0,      0,    1]
            ])
            T = T @ A
            positions.append(T[:3, 3].copy())
        return np.array(positions)

    def get_joint_transforms(self, joint_angles):
        """
        joint_angles: list or np.array of 6 joint angles (radians)
        return: list of 4x4 np.array, each is the transform from base to that joint
        """
        n = len(self.dh_params)
        T = np.eye(4)
        transforms = [T.copy()]
        for i in range(n):
            theta, d, a, alpha = self.dh_params[i]
            th = joint_angles[i] + theta
            ct, st = np.cos(th), np.sin(th)
            ca, sa = np.cos(alpha), np.sin(alpha)
            A = np.array([
                [ct, -st*ca,  st*sa, a*ct],
                [st,  ct*ca, -ct*sa, a*st],
                [0,      sa,     ca,    d],
                [0,       0,      0,    1]
            ])
            T = T @ A
            transforms.append(T.copy())
        return transforms
