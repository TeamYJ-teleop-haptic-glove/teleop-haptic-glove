import socket
import struct
import threading
import time

from dynamixel_sdk import *


# ========= Dynamixel 基本配置 =========
ADDR_TORQUE_ENABLE = 64
ADDR_LED_RED = 65

ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4

BAUDRATE = 57600          # 需与舵机实际波特率一致
PROTOCOL_VERSION = 2.0
DXL_ID = [0, 1, 2, 3, 4]  # 5个舵机ID
DEVICENAME = "COM15"      # U2D2端口（按你实际改）

DEFAULT_POS_SCALE = 2.0 * 180.0 / 4096.0  # position->degree（按你的舵机模式/分辨率确认）


# ========= UDP 配置（避免与之前重复：不使用5009/5010）=========
UDP_LOCAL_IP = "127.0.0.1"
UDP_LOCAL_PORT_SERVO = 5011

# 局域网转发：使用多播（不重复之前的 255.255.255.255 广播方案）
UDP_MCAST_GRP = "239.255.42.99"
UDP_MCAST_PORT = 5011
UDP_MCAST_TTL = 1  # 1=仅局域网内传播


READ_INTERVAL_SEC = 0.01  # 10ms一次（约100Hz）


class ServoReader:
    def __init__(self):
        self.running = True
        self.thread = None

        # ---- Dynamixel SDK ----
        self.portHandler = PortHandler(DEVICENAME)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)

        if not self.portHandler.openPort():
            print(f"端口打开失败: {DEVICENAME}")
            self.running = False
            return

        if not self.portHandler.setBaudRate(BAUDRATE):
            print(f"波特率设置失败: {BAUDRATE}")
            self.portHandler.closePort()
            self.running = False
            return

        # BulkRead
        self.groupBulkRead = GroupBulkRead(self.portHandler, self.packetHandler)
        self._init_bulk_read()

        # 扭矩使能 + LED
        self._enable_torque(enable=False)

        # ---- UDP sockets ----
        # 1) 本机回环发送
        self.udp_socket_local = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 2) 多播发送（局域网转发）
        self.udp_socket_mcast = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.udp_socket_mcast.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, UDP_MCAST_TTL)

    def _init_bulk_read(self):
        for dxl_id in DXL_ID:
            ok = self.groupBulkRead.addParam(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            if not ok:
                print(f"[ID:{dxl_id}] BulkRead参数添加失败")
                self.running = False
                return

    def _enable_torque(self, enable: bool):
        val = 1 if enable else 0
        for dxl_id in DXL_ID:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, val
            )
            if dxl_comm_result != COMM_SUCCESS:
                print(f"[ID:{dxl_id}] 扭矩设置失败: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                print(f"[ID:{dxl_id}] 扭矩错误: {self.packetHandler.getRxPacketError(dxl_error)}")
            else:
                # 点亮/熄灭LED辅助验证
                try:
                    self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_LED_RED, 1 if enable else 0)
                except Exception:
                    pass

    @staticmethod
    def _pack_5_floats(angles):
        # 确保长度为5，避免struct.pack报错
        a = list(angles)
        if len(a) < 5:
            a += [180.0] * (5 - len(a))
        if len(a) > 5:
            a = a[:5]
        return struct.pack("fffff", float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4]))

    def _read_bulk_position_deg(self):
        comm_result = self.groupBulkRead.txRxPacket()
        if comm_result != COMM_SUCCESS:
            print(f"BulkRead通信失败: {self.packetHandler.getTxRxResult(comm_result)}")
            return [180.0] * 5

        servo_angles = []
        for dxl_id in DXL_ID:
            if not self.groupBulkRead.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                print(f"[ID:{dxl_id}] 数据不可用")
                servo_angles.append(180.0)
                continue

            raw_pos = self.groupBulkRead.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            angle_deg = raw_pos * DEFAULT_POS_SCALE
            servo_angles.append(float(angle_deg))

        # 防御性补齐
        if len(servo_angles) != 5:
            servo_angles = (servo_angles + [180.0] * 5)[:5]

        return servo_angles

    def _send_udp_local(self, angles):
        try:
            msg = self._pack_5_floats(angles)
            self.udp_socket_local.sendto(msg, (UDP_LOCAL_IP, UDP_LOCAL_PORT_SERVO))
        except Exception as e:
            print(f"UDP本机发送失败: {e}")

    def _forward_udp_mcast(self, angles):
        """转发到本机局域网：UDP多播（地址/端口不与之前重复）"""
        try:
            msg = self._pack_5_floats(angles)
            self.udp_socket_mcast.sendto(msg, (UDP_MCAST_GRP, UDP_MCAST_PORT))
        except Exception as e:
            print(f"UDP多播转发失败: {e}")

    def _loop(self):
        while self.running:
            angles = self._read_bulk_position_deg()
            print("5个舵机角度(deg):", [round(a, 2) for a in angles])

            # 发送给本机 & 局域网
            self._send_udp_local(angles)
            self._forward_udp_mcast(angles)

            time.sleep(READ_INTERVAL_SEC)

    def start(self):
        if not self.running:
            print("初始化失败，无法启动")
            return
        print(f"启动：本机 {UDP_LOCAL_IP}:{UDP_LOCAL_PORT_SERVO} + 多播 {UDP_MCAST_GRP}:{UDP_MCAST_PORT}")
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        print("停止舵机读取...")
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        # 关闭扭矩 + 熄灭LED
        try:
            self._enable_torque(enable=False)
        except Exception:
            pass

        # 清理bulkread参数
        try:
            self.groupBulkRead.clearParam()
        except Exception:
            pass

        # 关闭端口/UDP
        try:
            self.portHandler.closePort()
        except Exception:
            pass

        try:
            self.udp_socket_local.close()
        except Exception:
            pass

        try:
            self.udp_socket_mcast.close()
        except Exception:
            pass

        print("资源已释放，已停止")


if __name__ == "__main__":
    servo_reader = ServoReader()
    try:
        servo_reader.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("用户中断程序")
    finally:
        servo_reader.stop()
