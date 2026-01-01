import numpy as np
from math import pi
from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
import time

def FK(theta, L):
    """
    3R 平面机械臂正运动学
    theta: (t1, t2, t3)  [rad]
    L: (l1, l2, l3)
    return: x, y, phi
    """
    t1, t2, t3 = theta
    l1, l2, l3 = L

    a1 = t1
    a2 = t1 + t2
    a3 = t1 + t2 + t3

    x = l1*np.cos(a1) + l2*np.cos(a2) + l3*np.cos(a3)
    y = l1*np.sin(a1) + l2*np.sin(a2) + l3*np.sin(a3)
    phi = a3
    return x, y, phi

def IK(x, y, L):
    """
    3R 平面机械臂 IK（只给定末端位置 x,y），通过数值优化求一组角度，
    同时让 theta1, theta2, theta3 尽量相等（均匀分摊弯曲）。

    输入:
      x, y: 目标末端位置
      L: [l1, l2, l3] 连杆长度

    输出:
      theta1, theta2, theta3  (rad)
    """

    l1, l2, l3 = map(float, L)
    target = np.array([x, y], dtype=float)

    # ===== 可调权重（一般不用你传参，直接在这里调） =====
    w_pos = 1.0          # 位置误差权重
    w_equal = 0.3        # “角度相等”权重（越大越趋向 θ1≈θ2≈θ3，但会牺牲一点到达精度）
    w_mag = 1e-3         # 角度幅值正则（防止角度漂很大；很小即可）

    # 数值参数
    max_iters = 200
    tol_pos = 1e-6
    damping0 = 1e-3
    step_limit = 0.6

    # 多初值，减少落入差的局部解（覆盖不同构型）
    seeds = [
        np.array([0.0,  0.0,  0.0]),
        np.array([-0.6, -0.6, -0.6]),
        np.array([ 0.6,  0.6,  0.6]),
        np.array([ 0.2, -0.8,  0.2]),
        np.array([-0.8,  0.2,  0.2]),
        np.array([ 1.0, -1.0,  1.0]),
        np.array([-1.0,  1.0, -1.0]),
    ]

    def wrap_to_pi(a):
        return (a + np.pi) % (2*np.pi) - np.pi

    def fk_pos(theta):
        t1, t2, t3 = theta
        a1 = t1
        a2 = t1 + t2
        a3 = t1 + t2 + t3
        px = l1*np.cos(a1) + l2*np.cos(a2) + l3*np.cos(a3)
        py = l1*np.sin(a1) + l2*np.sin(a2) + l3*np.sin(a3)
        return np.array([px, py], dtype=float)

    def jacobian_pos(theta):
        t1, t2, t3 = theta
        a1 = t1
        a2 = t1 + t2
        a3 = t1 + t2 + t3

        dx1 = -l1*np.sin(a1) - l2*np.sin(a2) - l3*np.sin(a3)
        dx2 =                - l2*np.sin(a2) - l3*np.sin(a3)
        dx3 =                                  - l3*np.sin(a3)

        dy1 =  l1*np.cos(a1) + l2*np.cos(a2) + l3*np.cos(a3)
        dy2 =                 l2*np.cos(a2) + l3*np.cos(a3)
        dy3 =                                   l3*np.cos(a3)

        return np.array([[dx1, dx2, dx3],
                         [dy1, dy2, dy3]], dtype=float)

    def score(theta):
        # 用于挑多初值里最好的解：位置误差 + 相等惩罚 + 幅值惩罚
        p = fk_pos(theta)
        epos = target - p
        eq = np.array([theta[0]-theta[1], theta[1]-theta[2]], dtype=float)
        return (w_pos*np.dot(epos, epos)
                + w_equal*np.dot(eq, eq)
                + w_mag*np.dot(theta, theta))

    def solve(theta0):
        theta = theta0.astype(float).copy()
        damping = damping0

        for _ in range(max_iters):
            p = fk_pos(theta)
            epos = target - p
            pos_err = float(np.linalg.norm(epos))
            if pos_err < tol_pos:
                break

            Jp = jacobian_pos(theta)  # 2x3

            # ===== 构造增广残差 r 和增广雅可比 J =====
            # r = [ sqrt(w_pos)*epos,
            #       sqrt(w_equal)*(t1-t2),
            #       sqrt(w_equal)*(t2-t3),
            #       sqrt(w_mag)*t1, sqrt(w_mag)*t2, sqrt(w_mag)*t3 ]
            swp = np.sqrt(w_pos)
            swe = np.sqrt(w_equal)
            swm = np.sqrt(w_mag)

            r = np.concatenate([
                swp * epos,
                swe * np.array([theta[0]-theta[1], theta[1]-theta[2]], dtype=float),
                swm * theta
            ])

            # 对应雅可比
            Jeq = np.array([
                [ 1.0, -1.0,  0.0],   # d(t1-t2)/dtheta
                [ 0.0,  1.0, -1.0],   # d(t2-t3)/dtheta
            ], dtype=float)

            J = np.vstack([
                swp * Jp,     # 2x3
                swe * Jeq,    # 2x3
                swm * np.eye(3)  # 3x3
            ])  # 总共 7x3

            # ===== LM / Damped Least Squares：min ||J*delta - r||^2 =====
            # (J^T J + λI) delta = J^T r
            A = J.T @ J + damping * np.eye(3)
            b = J.T @ r

            try:
                delta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(A) @ b

            # 限制步长防发散
            dnorm = float(np.linalg.norm(delta))
            if dnorm > step_limit:
                delta *= step_limit / (dnorm + 1e-12)

            theta_new = theta + delta
            theta_new = np.array([wrap_to_pi(a) for a in theta_new], dtype=float)

            # 接受准则：score 下降就接受，否则增大阻尼
            if score(theta_new) <= score(theta):
                theta = theta_new
                damping = max(damping * 0.7, 1e-8)
            else:
                damping = min(damping * 2.0, 1e2)

        return theta

    # 多初值求解，选 score 最小
    best_theta = None
    best_score = None
    for s in seeds:
        th = solve(s)
        sc = score(th)
        if best_score is None or sc < best_score:
            best_score = sc
            best_theta = th

    return float(best_theta[0]), float(best_theta[1]), float(best_theta[2])


# theta 是弧度值，标准坐标系值
def Calc(L_glove, L_leap, theta, k):
    
    gamma = 180 / pi
    # Servo[1]:   277.471 deg   Δ=   +4.482 deg
    # Joint 04:   227.243 deg   Δ=  -46.972 deg
    # Joint 05:   145.847 deg   Δ=  +66.336 deg
    # Joint 06:   196.859 deg   Δ=  +11.434 deg

    # L_glove = [6.95, 5.95, 2.7]
    # L_leap = [5.5, 3.5, 4.6]
    # theta = [-4.482 / gamma, -66.336 / gamma, -46.972 / gamma]
    
    x, y, phi = FK(theta, L_glove)

    x += 0.8
    y += 2.8

    # print(f"映射前  x:{x:3f}, y:{y:3f}")
    # k = 3
    x *= k 
    y *= k
    # print(f"映射后  x:{x:3f}, y:{y:3f}")

    th1, th2, th3 = IK(x, y, L_leap)
    # print("theta(rad):", th1, th2, th3)
    # print("theta(deg):", th1 * gamma, th2 * gamma, th3 * gamma)
    return [th1, th2, th3]

# if __name__ == "__main__":
#     L_glove = [6.95, 5.95, 2.7]
#     L_leap = [5.5, 3.5, 4.6]
#     theta = []
#     print(Calc(L_glove, L_leap, theta))

def IK2R(x, y, L):
    """
    2R 平面机械臂 IK（只给定末端位置 x,y），通过数值优化求一组角度，
    同时让 theta1, theta2 尽量相等（均匀分摊弯曲风格）。

    输入:
      x, y: 目标末端位置
      L: [l1, l2] 连杆长度

    输出:
      theta1, theta2  (rad)
    """
    l1, l2 = map(float, L)
    target = np.array([x, y], dtype=float)

    # ===== 可调权重 =====
    w_pos = 1.0          # 位置误差权重
    w_equal = 0.1        # “角度相等”权重（越大越趋向 θ1≈θ2，但可能影响到达精度）
    w_mag = 1e-3         # 角度幅值正则（防止角度漂很大；很小即可）

    # 数值参数
    max_iters = 200
    tol_pos = 1e-6
    damping0 = 1e-3
    step_limit = 0.6

    # 多初值：覆盖“肘上/肘下”等构型
    seeds = [
        np.array([0.0,  0.0]),
        np.array([0.6,  0.6]),
        np.array([-0.6, -0.6]),
        np.array([ 0.8, -0.8]),
        np.array([-0.8,  0.8]),
        np.array([ 1.2, -0.2]),
        np.array([-1.2,  0.2]),
    ]

    def wrap_to_pi(a):
        return (a + np.pi) % (2*np.pi) - np.pi

    def fk_pos(theta):
        t1, t2 = theta
        a1 = t1
        a2 = t1 + t2
        px = l1*np.cos(a1) + l2*np.cos(a2)
        py = l1*np.sin(a1) + l2*np.sin(a2)
        return np.array([px, py], dtype=float)

    def jacobian_pos(theta):
        t1, t2 = theta
        a1 = t1
        a2 = t1 + t2

        dx1 = -l1*np.sin(a1) - l2*np.sin(a2)
        dx2 =                - l2*np.sin(a2)

        dy1 =  l1*np.cos(a1) + l2*np.cos(a2)
        dy2 =                 l2*np.cos(a2)

        return np.array([[dx1, dx2],
                         [dy1, dy2]], dtype=float)

    def score(theta):
        # 用于挑多初值里最好的解：位置误差 + 相等惩罚 + 幅值惩罚
        p = fk_pos(theta)
        epos = target - p
        eq = np.array([theta[0] - theta[1]], dtype=float)  # θ1≈θ2
        return (w_pos*np.dot(epos, epos)
                + w_equal*np.dot(eq, eq)
                + w_mag*np.dot(theta, theta))

    def solve(theta0):
        theta = theta0.astype(float).copy()
        damping = damping0

        for _ in range(max_iters):
            p = fk_pos(theta)
            epos = target - p
            pos_err = float(np.linalg.norm(epos))
            if pos_err < tol_pos:
                break

            Jp = jacobian_pos(theta)  # 2x2

            # ===== 构造增广残差 r 和增广雅可比 J =====
            # r = [ sqrt(w_pos)*epos,
            #       sqrt(w_equal)*(t1-t2),
            #       sqrt(w_mag)*t1, sqrt(w_mag)*t2 ]
            swp = np.sqrt(w_pos)
            swe = np.sqrt(w_equal)
            swm = np.sqrt(w_mag)

            r = np.concatenate([
                swp * epos,
                swe * np.array([theta[0] - theta[1]], dtype=float),
                swm * theta
            ])  # 2 + 1 + 2 = 5 维

            Jeq = np.array([[1.0, -1.0]], dtype=float)  # d(t1-t2)/dtheta

            J = np.vstack([
                swp * Jp,         # 2x2
                swe * Jeq,        # 1x2
                swm * np.eye(2)   # 2x2
            ])  # 总共 5x2

            # ===== LM / Damped Least Squares：min ||J*delta - r||^2 =====
            A = J.T @ J + damping * np.eye(2)
            b = J.T @ r

            try:
                delta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(A) @ b

            # 限制步长防发散
            dnorm = float(np.linalg.norm(delta))
            if dnorm > step_limit:
                delta *= step_limit / (dnorm + 1e-12)

            theta_new = theta + delta
            theta_new = np.array([wrap_to_pi(a) for a in theta_new], dtype=float)

            # 接受准则：score 下降就接受，否则增大阻尼
            if score(theta_new) <= score(theta):
                theta = theta_new
                damping = max(damping * 0.7, 1e-8)
            else:
                damping = min(damping * 2.0, 1e2)

        return theta

    # 多初值求解，选 score 最小
    best_theta = None
    best_score = None
    for s in seeds:
        th = solve(s)
        sc = score(th)
        if best_score is None or sc < best_score:
            best_score = sc
            best_theta = th

    return float(best_theta[0]), float(best_theta[1])


# 接受弧度值，返回弧度值
def Calc_thumb(L_glove, L_leap, theta, k):
    
    # L_glove.append(0)
    # theta.append(0)
    x, y, phi = FK(theta, L_glove)

    # x += 0.8
    y -= 2.0

    print(f"映射前  x:{x:3f}, y:{y:3f}")
    # k = 3
    x *= k 
    y *= k
    print(f"映射后  x:{x:3f}, y:{y:3f}")

    th1, th2 = IK2R(x, y, L_leap)
    # print("theta(rad):", th1, th2, th3)
    # print("theta(deg):", th1 * gamma, th2 * gamma, th3 * gamma)
    return [th1, th2]
