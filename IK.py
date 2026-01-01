import numpy as np
import xml.etree.ElementTree as ET

# ============================================================
# Part 1) 手套 FK（来自你上传的脚本结构，保持关节顺序不变）
#   - 关节顺序 JOINT_ORDER 决定你的传感器数组如何对齐（避免错位）
#   - TIP_LINKS 决定取哪个 link 的位置当作“指尖”
# ============================================================

# -------------------------
# 1) 手套关节名称 -> 手指 + 从指根开始第几个关节（注释用）
# -------------------------
JOINT_META = {
    # Thumb (拇指) 5关节
    "thumb_mcp":    ("thumb", 1),
    "thumb_split":  ("thumb", 2),
    "thumb_bend_3": ("thumb", 3),
    "thumb_bend_2": ("thumb", 4),
    "thumb_bend_1": ("thumb", 5),

    # Index (食指) 4关节
    "index_split":  ("index", 1),
    "index_bend_3": ("index", 2),
    "index_bend_2": ("index", 3),
    "index_bend_1": ("index", 4),

    # Middle (中指) 4关节
    "middle_split":  ("middle", 1),
    "middle_bend_3": ("middle", 2),
    "middle_bend_2": ("middle", 3),
    "middle_bend_1": ("middle", 4),

    # Ring (无名指) 4关节
    "ring_split":  ("ring", 1),
    "ring_bend_3": ("ring", 2),
    "ring_bend_2": ("ring", 3),
    "ring_bend_1": ("ring", 4),

    # Pinky (小指) 4关节（LEAP不需要，但手套FK依然算得出来）
    "pinky_split":  ("pinky", 1),
    "pinky_bend_3": ("pinky", 2),
    "pinky_bend_2": ("pinky", 3),
    "pinky_bend_1": ("pinky", 4),
}

# 手套指尖末端 link（终端 link）
TIP_LINKS = {
    "thumb":  "thumb_bend_1",
    "index":  "index_bend_1",
    "middle": "middle_bend_1",
    "ring":   "ring_bend_1",
    "pinky":  "pinky_bend_1",
}

# ✅ 手套传感器数组输入必须严格按这个顺序（避免错位）
JOINT_ORDER = [
    # thumb
    "thumb_mcp", "thumb_split", "thumb_bend_3", "thumb_bend_2", "thumb_bend_1",
    # index
    "index_split", "index_bend_3", "index_bend_2", "index_bend_1",
    # middle
    "middle_split", "middle_bend_3", "middle_bend_2", "middle_bend_1",
    # ring
    "ring_split", "ring_bend_3", "ring_bend_2", "ring_bend_1",
    # pinky
    "pinky_split", "pinky_bend_3", "pinky_bend_2", "pinky_bend_1",
]

def rpy_to_R(roll, pitch, yaw):
    """URDF rpy: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]])
    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0],
                   [sy, cy, 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

def axis_angle_to_R(axis, theta):
    """Rodrigues"""
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    axis = axis / n
    x, y, z = axis
    c, s = np.cos(theta), np.sin(theta)
    C = 1 - c
    return np.array([
        [c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
    ])

def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def parse_vec(s, default):
    if s is None:
        return np.array(default, dtype=float)
    return np.array([float(x) for x in s.strip().split()], dtype=float)

def load_urdf_joints(urdf_path):
    root = ET.parse(urdf_path).getroot()
    joints = {}
    for j in root.findall("joint"):
        name = j.attrib["name"]
        jtype = j.attrib.get("type", "revolute")
        parent = j.find("parent").attrib["link"]
        child = j.find("child").attrib["link"]
        origin = j.find("origin")
        xyz = parse_vec(origin.attrib.get("xyz") if origin is not None else None, [0, 0, 0])
        rpy = parse_vec(origin.attrib.get("rpy") if origin is not None else None, [0, 0, 0])
        axis_el = j.find("axis")
        axis = parse_vec(axis_el.attrib.get("xyz") if axis_el is not None else None, [0, 0, 1])
        joints[name] = {"name": name, "type": jtype, "parent": parent, "child": child, "xyz": xyz, "rpy": rpy, "axis": axis}
    return joints

def build_tree(joints):
    tree = {}
    for j in joints.values():
        tree.setdefault(j["parent"], []).append(j["name"])
    return tree

def forward_kinematics_all_links(joints, base_link, joint_angles_rad):
    tree = build_tree(joints)
    link_T = {base_link: np.eye(4)}

    def dfs(parent_link):
        parent_T = link_T[parent_link]
        for jname in tree.get(parent_link, []):
            j = joints[jname]
            q = float(joint_angles_rad.get(jname, 0.0))
            R0 = rpy_to_R(j["rpy"][0], j["rpy"][1], j["rpy"][2])
            T_origin = make_T(R0, j["xyz"])
            Rq = axis_angle_to_R(j["axis"], q)
            T_rot = make_T(Rq, np.zeros(3))
            child_T = parent_T @ (T_origin @ T_rot)
            link_T[j["child"]] = child_T
            dfs(j["child"])

    dfs(base_link)
    return link_T

def fk_fingertips_from_sensor(glove_urdf_path, sensor_joint_angles, sensor_unit="rad", base_link="base_link"):
    """
    sensor_joint_angles:
      - dict: {joint_name: angle}（最稳，不会错位）
      - 或 ndarray/list: 按 JOINT_ORDER 顺序提供（长度=21）
    输出:
      tip_pos: dict[finger]->(3,)  五指指尖在手套 base_link 下坐标
    """
    joints = load_urdf_joints(glove_urdf_path)

    if isinstance(sensor_joint_angles, dict):
        q = dict(sensor_joint_angles)
    else:
        arr = np.asarray(sensor_joint_angles, dtype=float).reshape(-1)
        if arr.size != len(JOINT_ORDER):
            raise ValueError(f"手套传感器数组长度应为 {len(JOINT_ORDER)}，但你给了 {arr.size}")
        q = {name: float(arr[i]) for i, name in enumerate(JOINT_ORDER)}

    if sensor_unit.lower() == "deg":
        for k in list(q.keys()):
            q[k] = np.deg2rad(q[k])

    # 防呆：关节名必须存在
    for jname in q.keys():
        if jname not in joints:
            raise KeyError(f"手套URDF里找不到关节名: {jname}")

    link_T = forward_kinematics_all_links(joints, base_link=base_link, joint_angles_rad=q)

    tip_pos = {}
    for finger, tip_link in TIP_LINKS.items():
        if tip_link not in link_T:
            raise KeyError(f"手套FK结果里找不到指尖 link: {tip_link}")
        tip_pos[finger] = link_T[tip_link][:3, 3].copy()

    return tip_pos


# ============================================================
# Part 2) LEAP URDF FK + 多指指尖位置 IK（输出16维关节角）
#   你的LEAP URDF关节名是 "0"..."15"
#   ✅ 输出 cmd16 顺序固定为 LEAP_CMD_ORDER（防错位）
# ============================================================

class URDFKinematics:
    def __init__(self, urdf_path: str):
        self.urdf_path = urdf_path
        self.joints = self._load_urdf_joints(urdf_path)
        self.tree = self._build_tree(self.joints)
        self.limits = self._load_limits(urdf_path)

    def _load_urdf_joints(self, urdf_path):
        root = ET.parse(urdf_path).getroot()
        joints = {}
        for j in root.findall("joint"):
            name = j.attrib["name"]
            jtype = j.attrib.get("type", "revolute")
            parent = j.find("parent").attrib["link"]
            child = j.find("child").attrib["link"]
            origin = j.find("origin")
            xyz = parse_vec(origin.attrib.get("xyz") if origin is not None else None, [0, 0, 0])
            rpy = parse_vec(origin.attrib.get("rpy") if origin is not None else None, [0, 0, 0])
            axis_el = j.find("axis")
            axis = parse_vec(axis_el.attrib.get("xyz") if axis_el is not None else None, [0, 0, 1])
            joints[name] = {"name": name, "type": jtype, "parent": parent, "child": child, "xyz": xyz, "rpy": rpy, "axis": axis}
        return joints

    def _load_limits(self, urdf_path):
        root = ET.parse(urdf_path).getroot()
        limits = {}
        for j in root.findall("joint"):
            name = j.attrib["name"]
            lim = j.find("limit")
            if lim is None:
                continue
            lo = lim.attrib.get("lower", None)
            hi = lim.attrib.get("upper", None)
            if lo is None or hi is None:
                continue
            try:
                limits[name] = (float(lo), float(hi))
            except ValueError:
                pass
        return limits

    def _build_tree(self, joints):
        tree = {}
        for j in joints.values():
            tree.setdefault(j["parent"], []).append(j["name"])
        return tree

    def fk_all_links(self, base_link: str, q_rad: dict):
        link_T = {base_link: np.eye(4)}

        def dfs(parent_link):
            parent_T = link_T[parent_link]
            for jname in self.tree.get(parent_link, []):
                j = self.joints[jname]
                ang = float(q_rad.get(jname, 0.0))
                R0 = rpy_to_R(j["rpy"][0], j["rpy"][1], j["rpy"][2])
                T_origin = make_T(R0, j["xyz"])
                Rq = axis_angle_to_R(j["axis"], ang)
                T_rot = make_T(Rq, np.zeros(3))
                child_T = parent_T @ (T_origin @ T_rot)
                link_T[j["child"]] = child_T
                dfs(j["child"])

        dfs(base_link)
        return link_T

    def clip_to_limits(self, q_rad: dict):
        if not self.limits:
            return q_rad
        out = dict(q_rad)
        for jname, (lo, hi) in self.limits.items():
            if jname in out:
                out[jname] = float(np.clip(out[jname], lo, hi))
        return out


# ---- LEAP URDF 固定信息（按你上传的LEAP URDF解析确认过）----
LEAP_BASE_LINK = "palm_lower"

LEAP_TIP_LINKS = {
    "index":  "fingertip",
    "middle": "fingertip_2",
    "ring":   "fingertip_3",
    "thumb":  "thumb_fingertip",
}

# ✅ 16维命令向量固定输出顺序（非常重要：不要改，除非你的驱动要求不同）
# index:  1 -> 0 -> 2 -> 3
# middle: 5 -> 4 -> 6 -> 7
# ring:   9 -> 8 ->10 ->11
# thumb: 12 ->13 ->14 ->15
LEAP_CMD_ORDER = ["1","0","2","3",  "5","4","6","7",  "9","8","10","11",  "12","13","14","15"]

def solve_leap_ik_dls_numeric(
    kin: URDFKinematics,
    p_target_base: dict,   # dict finger->(3,) 目标指尖位置（在 palm_lower 坐标系）
    q_init_rad: dict,      # dict joint->rad  初值（上一帧解）
    iters: int = 8,
    step: float = 0.7,
    damping: float = 1e-2,
    eps: float = 1e-5,
):
    """
    多端点位置 IK：用数值Jacobian + DLS。
    输出: q_sol_rad dict[joint]->rad（joint名是 "0"..."15"）
    """
    fingers = ["thumb", "index", "middle", "ring"]

    def dict_to_vec(qd):
        return np.array([float(qd.get(j, 0.0)) for j in LEAP_CMD_ORDER], dtype=float)

    def vec_to_dict(qv):
        return {LEAP_CMD_ORDER[i]: float(qv[i]) for i in range(len(LEAP_CMD_ORDER))}

    def fingertip_positions(qd):
        link_T = kin.fk_all_links(base_link=LEAP_BASE_LINK, q_rad=qd)
        pts = {}
        for f in fingers:
            pts[f] = link_T[LEAP_TIP_LINKS[f]][:3, 3].copy()
        return pts

    qv = dict_to_vec(q_init_rad)

    for _ in range(iters):
        qd = kin.clip_to_limits(vec_to_dict(qv))
        p_cur = fingertip_positions(qd)

        e = np.concatenate([(p_target_base[f] - p_cur[f]) for f in fingers], axis=0)  # (12,)

        J = np.zeros((12, 16), dtype=float)
        for j_idx, jname in enumerate(LEAP_CMD_ORDER):
            qv_pert = qv.copy()
            qv_pert[j_idx] += eps
            qd_pert = kin.clip_to_limits(vec_to_dict(qv_pert))
            p_pert = fingertip_positions(qd_pert)
            de = np.concatenate([(p_pert[f] - p_cur[f]) for f in fingers], axis=0)
            J[:, j_idx] = de / eps

        A = J.T @ J + (damping**2) * np.eye(16)
        b = J.T @ e
        dq = np.linalg.solve(A, b)

        qv = qv + step * dq

    return kin.clip_to_limits(vec_to_dict(qv))

def qdict_to_cmd16(q_sol_rad: dict):
    """把关节字典按 LEAP_CMD_ORDER 导出为 16 维命令（弧度）"""
    return np.array([float(q_sol_rad.get(j, 0.0)) for j in LEAP_CMD_ORDER], dtype=float)


# ============================================================
# Part 3) 手套指尖 -> LEAP 指尖目标（坐标对齐/尺度）
#   你把标定得到的 R_align / scale 填进去即可
# ============================================================

def map_glove_tips_to_leap_targets(
    glove_tip_pos_base: dict,   # glove base_link 下的 tip坐标
    glove_base_pos: np.ndarray, # glove base_link原点（若就是base_link，则填0；若你想用wrist点，则填wrist坐标）
    R_align: np.ndarray,        # 3x3: glove_base -> leap_palm_lower 的旋转
    scale: float,               # glove -> leap 尺度
):
    out = {}
    for f in ["thumb", "index", "middle", "ring"]:
        pg = glove_tip_pos_base[f] - glove_base_pos
        out[f] = R_align @ (scale * pg)
    return out


# ============================================================
# Part 4) 一条龙接口：手套关节角 -> LEAP 16维关节角
#   ✅ 你只需要喂“标定后的手套关节角”（按JOINT_ORDER），返回cmd16
# ============================================================

class GloveToLeapSolver:
    def __init__(
        self,
        glove_urdf_path: str,
        leap_urdf_path: str,
        glove_base_link: str = "base_link",
        glove_sensor_unit: str = "rad",
        # 标定参数（你填自己的）
        R_align: np.ndarray = None,
        scale: float = 1.0,
        glove_base_pos: np.ndarray = None,
    ):
        self.glove_urdf_path = glove_urdf_path
        self.leap_urdf_path = leap_urdf_path
        self.glove_base_link = glove_base_link
        self.glove_sensor_unit = glove_sensor_unit

        self.kin_leap = URDFKinematics(leap_urdf_path)

        self.R_align = np.eye(3) if R_align is None else R_align
        self.scale = float(scale)
        self.glove_base_pos = np.zeros(3) if glove_base_pos is None else glove_base_pos.astype(float)

        # warm start（上一帧解），第一帧全0
        self.q_prev = {j: 0.0 for j in LEAP_CMD_ORDER}

    def step(self, glove_sensor_angles):
        """
        glove_sensor_angles:
          - dict: {joint_name: angle}  （强烈推荐）
          - 或 ndarray/list: 按 JOINT_ORDER 顺序（长度=21）
        返回:
          cmd16: np.ndarray shape(16,)  （弧度，顺序=LEAP_CMD_ORDER）
        """

        # 1) 手套 FK -> 五指 tip
        glove_tip_pos = fk_fingertips_from_sensor(
            glove_urdf_path=self.glove_urdf_path,
            sensor_joint_angles=glove_sensor_angles,
            sensor_unit=self.glove_sensor_unit,
            base_link=self.glove_base_link,
        )

        # 2) 映射到 LEAP palm_lower 坐标系的目标 tip（忽略pinky）
        p_target_base = map_glove_tips_to_leap_targets(
            glove_tip_pos_base=glove_tip_pos,
            glove_base_pos=self.glove_base_pos,
            R_align=self.R_align,
            scale=self.scale,
        )

        # 3) IK 解 LEAP 16关节
        q_sol = solve_leap_ik_dls_numeric(
            kin=self.kin_leap,
            p_target_base=p_target_base,
            q_init_rad=self.q_prev,
            iters=8,
            step=0.7,
            damping=1e-2,
            eps=1e-5,
        )

        # 4) 导出16维命令（固定顺序）
        cmd16 = qdict_to_cmd16(q_sol)

        # 5) warm start 更新
        self.q_prev = q_sol
        return cmd16


# ============================================================
# Demo：把这里替换成你的实时传感器角度（标定后）
# ============================================================
if __name__ == "__main__":
    # 你手套URDF、LEAP URDF路径（按本对话环境的文件路径给默认）

    GLOVE_URDF = "DOGlove_meshes/DOGlove_Ver20241223.urdf"
    LEAP_URDF  = "LEAP_Hand_Sim/assets/leap_hand/robot.urdf"

    # 你标定得到的对齐参数（先填单位阵/1.0也能跑，但效果通常需要标定）



    # R_align = np.eye(3)
    # scale = 1.0



    R_align = np.array([
        [0, 0, -1],
        [0, -1,  0],
        [-1, 0,  0],
    ], dtype=float)   # x_leap = -z_glove, y_leap = y_glove, z_leap = x_glove

    scale = 0.37      # 人手长度 -> LEAP 长度（大概 0.35~0.45 之间）


    glove_base_pos = np.zeros(3)

    solver = GloveToLeapSolver(
        glove_urdf_path=GLOVE_URDF,
        leap_urdf_path=LEAP_URDF,
        glove_base_link="base_link",
        glove_sensor_unit="rad",  # 你的“标定后角度”如果是度，改成 "deg"
        R_align=R_align,
        scale=scale,
        glove_base_pos=glove_base_pos,
    )

    # ✅ 这里喂“标定后的手套关节角”
    # 推荐用 dict（完全避免顺序错位）：
    glove_angles_dict = {name: 0.0 for name in JOINT_ORDER}

    glove_angles_dict["index_bend_3"] = 0
    glove_angles_dict["index_bend_2"] = 0
    glove_angles_dict["index_bend_1"] = 0

    cmd16 = solver.step(glove_angles_dict)


    print(repr(list(cmd16)))

    print("LEAP_CMD_ORDER =", LEAP_CMD_ORDER)

    exit()















