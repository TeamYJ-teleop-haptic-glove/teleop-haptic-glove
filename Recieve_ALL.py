import socket
import struct
import time
import sys
import numpy as np
from math import pi
from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
import time
from Calc_theta import Calc
from Calc_theta import Calc_thumb

def abs(x):
    if (x < 0):
        return -x
    else:
        return x

class LeapNode:
    def __init__(self):
        ####Some parameters
        # I recommend you keep the current limit from 350 for the lite, and 550 for the full hand
        # Increase KP if the hand is too weak, decrease if it's jittery.
        # kP(default) = 600
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350  ##set this to 550 if you are using full motors!!!!
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))
        #You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        # For example ls /dev/serial/by-id/* to find your LEAP Hand. Then use the result.  
        # For example: /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7W91VW-if00-port0
        self.motors = motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        try:
            self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', 4000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB1', 4000000)
                self.dxl_client.connect()
            except Exception:
                self.dxl_client = DynamixelClient(motors, 'COM13', 4000000)
                self.dxl_client.connect()
        #Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
        self.dxl_client.sync_write(motors, np.ones(len(motors))*5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2) # Pgain stiffness     
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2) # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2) # Igain
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2) # Dgain damping
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2) # Dgain damping for side to side should be a bit less
        #Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    #Receive LEAP pose and directly control the robot
    def set_leap(self, pose):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    #allegro compatibility joint angles.  It adds 180 to make the fully open position at 0 instead of 180
    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #Sim compatibility for policies, it assumes the ranges are [-1,1] and then convert to leap hand ranges.
    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #read position of the robot
    def read_pos(self):
        return self.dxl_client.read_pos()
    #read velocity
    def read_vel(self):
        return self.dxl_client.read_vel()
    #read current
    def read_cur(self):
        return self.dxl_client.read_cur()
    #These combined commands are faster FYI and return a list of data
    def pos_vel(self):
        return self.dxl_client.read_pos_vel()
    #These combined commands are faster FYI and return a list of data
    def pos_vel_eff_srv(self):
        return self.dxl_client.read_pos_vel_cur()

# ================== 配置 ==================
# joints (16 floats)
JOINT_UDP_IP = "127.0.0.1"
JOINT_UDP_PORT = 5009
JOINT_NUM_FLOATS = 16
JOINT_PACKET_SIZE = JOINT_NUM_FLOATS * 4

# servos (5 floats) + multicast
SERVO_BIND_IP = "0.0.0.0"
SERVO_UDP_PORT = 5011
SERVO_NUM_FLOATS = 5
SERVO_PACKET_SIZE = SERVO_NUM_FLOATS * 4
MCAST_GRP = "239.255.42.99"

UPDATE_HZ = 50
# =========================================


def create_joint_socket():
    """接收16路关节角：127.0.0.1:5009"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((JOINT_UDP_IP, JOINT_UDP_PORT))
    sock.setblocking(False)
    return sock


def create_servo_socket():
    """
    接收5路舵机角：
    - 绑定 0.0.0.0:5011
    - 加入组播 239.255.42.99
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

    # 允许端口复用（组播常用）
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    sock.bind((SERVO_BIND_IP, SERVO_UDP_PORT))

    # 加入组播组（默认网卡）
    mreq = struct.pack("4s4s", socket.inet_aton(MCAST_GRP), socket.inet_aton("0.0.0.0"))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    sock.setblocking(False)
    return sock


def clear_console():
    # ANSI：清屏 + 光标回到左上角
    sys.stdout.write("\033[H\033[J")
    sys.stdout.flush()


def drain_latest(sock, expected_size, unpack_fmt):
    """
    非阻塞读光缓冲区，只保留最后一包。
    返回：(latest_tuple_or_None, latest_addr_or_None, last_rx_time_or_None)
    """
    latest = None
    latest_from = None
    last_rx_time = None

    while True:
        try:
            data, addr = sock.recvfrom(4096)
        except BlockingIOError:
            break
        except OSError:
            # socket closed
            break

        if len(data) == expected_size:
            latest = struct.unpack(unpack_fmt, data)
            latest_from = addr
            last_rx_time = time.time()

    return latest, latest_from, last_rx_time


def get_latest_with_wait(sock, expected_size, unpack_fmt, wait_s=5.0):
    """
    在 wait_s 时间内尝试拿到至少一包有效数据（取最新）。
    返回：(latest_tuple_or_None, latest_addr_or_None, last_rx_time_or_None)
    """
    end = time.time() + wait_s
    latest = None
    latest_from = None
    latest_time = None

    while time.time() < end:
        v, v_from, v_t = drain_latest(sock, expected_size, unpack_fmt)
        if v is not None:
            latest, latest_from, latest_time = v, v_from, v_t
            # 再短暂“追一追”，确保拿到更接近此刻的最新包
            t2_end = time.time() + 0.05
            while time.time() < t2_end:
                v2, v2_from, v2_t = drain_latest(sock, expected_size, unpack_fmt)
                if v2 is not None:
                    latest, latest_from, latest_time = v2, v2_from, v2_t
                time.sleep(0.001)
            break
        time.sleep(0.01)

    return latest, latest_from, latest_time


def main():
    leap_hand = LeapNode()
    joint_sock = create_joint_socket()
    servo_sock = create_servo_socket()

    latest_joint = None
    latest_joint_from = None
    latest_joint_time = None

    latest_servo = None
    latest_servo_from = None
    latest_servo_time = None

    # 归零基准
    zero_joint = None
    zero_joint_from = None
    zero_joint_time = None

    zero_servo = None
    zero_servo_from = None
    zero_servo_time = None

    # dt = 1.0 / float(UPDATE_HZ)
    dt = 1.0 / 30

    print(f"Listening joints  : {JOINT_UDP_IP}:{JOINT_UDP_PORT} ({JOINT_NUM_FLOATS} floats)")
    print(f"Listening servos  : {SERVO_BIND_IP}:{SERVO_UDP_PORT} ({SERVO_NUM_FLOATS} floats), multicast {MCAST_GRP}")
    print("Press Ctrl+C to quit.")
    # time.sleep(0.3)
    #
    # # 先读一些数据，让“归零”时更容易抓到最新包
    # for _ in range(50):
    #     j, j_from, j_t = drain_latest(joint_sock, JOINT_PACKET_SIZE, "f" * JOINT_NUM_FLOATS)
    #     if j is not None:
    #         latest_joint, latest_joint_from, latest_joint_time = j, j_from, j_t
    #
    #     s, s_from, s_t = drain_latest(servo_sock, SERVO_PACKET_SIZE, "f" * SERVO_NUM_FLOATS)
    #     if s is not None:
    #         latest_servo, latest_servo_from, latest_servo_time = s, s_from, s_t
    #     time.sleep(0.01)
    #
    # # ============ 归零提示 ============
    # input("\n请先完成【归零操作】，完成后按下 ENTER 继续...")
    #
    # # 在按下 ENTER 的时刻，抓取最新收到的包作为 zero
    # zj, zj_from, zj_t = get_latest_with_wait(joint_sock, JOINT_PACKET_SIZE, "f" * JOINT_NUM_FLOATS, wait_s=5.0)
    # if zj is not None:
    #     zero_joint, zero_joint_from, zero_joint_time = zj, zj_from, zj_t
    # else:
    #     zero_joint = None
    #
    # zs, zs_from, zs_t = get_latest_with_wait(servo_sock, SERVO_PACKET_SIZE, "f" * SERVO_NUM_FLOATS, wait_s=5.0)
    # if zs is not None:
    #     zero_servo, zero_servo_from, zero_servo_time = zs, zs_from, zs_t
    # else:
    #     zero_servo = None
    #
    # print("\n已记录归零基准：")
    #
    #
    # if zero_joint is None:
    #     print("  - joints: 未获取到有效数据（将无法计算 joints 差值）")
    # else:
    #     print(f"  - joints: from {zero_joint_from}, t={time.strftime('%H:%M:%S', time.localtime(zero_joint_time))}")
    #
    # if zero_servo is None:
    #     print("  - servos: 未获取到有效数据（将无法计算 servos 差值）")
    # else:
    #     print(f"  - servos: from {zero_servo_from}, t={time.strftime('%H:%M:%S', time.localtime(zero_servo_time))}")

    zero_joint = []
    zero_servo = []
    for i in range(100):
        zero_joint.append(0)
        zero_servo.append(0)

    zero_joint[0]  = 276
    zero_joint[1]  = 86
    zero_joint[4]  = 283
    zero_joint[5]  = 77.5
    zero_joint[6]  = 346.4
    zero_joint[7]  = 287.8
    zero_joint[8]  = 83.3
    zero_joint[9]  = 185.6
    zero_joint[10] = 82
    zero_joint[11] = 269.6
    zero_joint[12] = 179.9

    zero_servo[0] = 176.309
    zero_servo[1] = 276.064
    zero_servo[2] = 270.791
    zero_servo[3] = 67.324

    time.sleep(0.8)
    # =================================

    try:
        while True:
            # 读最新 joints
            j, j_from, j_t = drain_latest(
                joint_sock, JOINT_PACKET_SIZE, "f" * JOINT_NUM_FLOATS
            )
            if j is not None:
                latest_joint, latest_joint_from, latest_joint_time = j, j_from, j_t

            # 读最新 servos
            s, s_from, s_t = drain_latest(
                servo_sock, SERVO_PACKET_SIZE, "f" * SERVO_NUM_FLOATS
            )
            if s is not None:
                latest_servo, latest_servo_from, latest_servo_time = s, s_from, s_t

            # 刷新显示
            clear_console()

            print("Merged UDP Receiver Display (with ZERO & DELTA)")
            print("=" * 80)

            # ------- joints -------
            print(f"\n[Joints] {JOINT_UDP_IP}:{JOINT_UDP_PORT}  ({JOINT_NUM_FLOATS} floats)")
            if latest_joint is None:
                print("  No valid joint packet received yet.")
            else:
                age_ms = (time.time() - latest_joint_time) * 1000.0 if latest_joint_time else 0.0
                print(f"  Latest from: {latest_joint_from}   age: {age_ms:.1f} ms")

                if zero_joint is None:
                    print("  ZERO reference: (none)  -> delta unavailable")
                    for i, v in enumerate(latest_joint):
                        print(f"  Joint {i:02d}: {v:9.3f} deg")
                else:
                    z_age_ms = (time.time() - zero_joint_time) * 1000.0 if zero_joint_time else 0.0
                    print(f"  ZERO from : {zero_joint_from}   captured age: {z_age_ms:.1f} ms ago")
                    for i, v in enumerate(latest_joint):
                        dv = v - zero_joint[i]
                        print(f"  Joint {i:02d}: {v:9.3f} deg   Δ={dv:+9.3f} deg")

            # ------- servos -------
            print(f"\n[Servos] {SERVO_BIND_IP}:{SERVO_UDP_PORT}  ({SERVO_NUM_FLOATS} floats)  MCAST: {MCAST_GRP}")
            if latest_servo is None:
                print("  No valid servo packet received yet.")
            else:
                age_ms = (time.time() - latest_servo_time) * 1000.0 if latest_servo_time else 0.0
                print(f"  Latest from: {latest_servo_from}   age: {age_ms:.1f} ms")

                if zero_servo is None:
                    print("  ZERO reference: (none)  -> delta unavailable")
                    for i, v in enumerate(latest_servo):
                        print(f"  Servo[{i}]: {v:9.3f} deg")
                else:
                    z_age_ms = (time.time() - zero_servo_time) * 1000.0 if zero_servo_time else 0.0
                    print(f"  ZERO from : {zero_servo_from}   captured age: {z_age_ms:.1f} ms ago")
                    for i, v in enumerate(latest_servo):
                        dv = v - zero_servo[i]
                        print(f"  Servo[{i}]: {v:9.3f} deg   Δ={dv:+9.3f} deg")

            print("\nPress Ctrl+C to exit.")
            

            Joint = []
            for i, v in enumerate(latest_joint):
                dv = v - zero_joint[i]
                Joint.append(dv)
            Servo = []
            for i, v in enumerate(latest_servo):
                dv = v - zero_servo[i]
                Servo.append(dv)

            gamma = pi / 180
            L_leap = [5.3, 3.6, 4.6]
            pose = np.full(16, pi)
            # print(f"theta:{theta}")
            # print(f"sol:{sol}")

            # 食指
            L_glove = [7.0, 6.00, 1.6 + 1]
            theta = [-Servo[1] * gamma, -abs(Joint[5]) * gamma, -abs(Joint[4]) * gamma]
            sol = Calc(L_glove, L_leap, theta, 3)
            pose[1] += -sol[0]
            pose[2] += -sol[1]
            pose[3] += -sol[2]
            # 中指
            L_glove = [7.8, 6.5, 2.6]
            theta = [-Servo[2] * gamma, -abs(Joint[8]) * gamma, -abs(Joint[7]) * gamma]
            sol = Calc(L_glove, L_leap, theta, 2.54)
            pose[5] += -sol[0]
            pose[6] += -sol[1]
            pose[7] += -sol[2]
            # 无名指
            L_glove = [7.3, 6.0, 2.7]
            theta = [-Servo[3] * gamma, -abs(Joint[11]) * gamma, -abs(Joint[10]) * gamma]
            sol = Calc(L_glove, L_leap, theta, 3)
            pose[9] += -sol[0]
            pose[10] += -sol[1]
            pose[11] += -sol[2]

            # 拇指
            L_glove = [5.0, 4.0, 2.7]
            L_leap = [4.6, 6.0]
            theta = [Servo[0] * gamma, abs(Joint[1]) * gamma, abs(Joint[0]) * gamma]
            sol = Calc_thumb(L_glove, L_leap, theta, 3)
            pose[14] += sol[0]
            pose[15] += sol[1]

            pose[12] -= pi / 3
            pose[13] -= pi / 10


            # 弯曲
            Joint[12] = -Joint[12] # 给定正方向，向右为正
            # print(f"偏转角度：{Joint[6]}, {Joint[9]}, {Joint[12]}")
            k = 6
            # pose[0] += Joint[6] * gamma * k
            pose[4] += Joint[9] * gamma * k
            # pose[8] += Joint[12] * gamma * k





            leap_hand.set_leap(pose)

            # time.sleep(dt)

    except KeyboardInterrupt:
        print("\nExiting...")

    finally:
        try:
            joint_sock.close()
        except Exception:
            pass
        try:
            servo_sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
