import socket
import struct
import time

NUM_FLOATS = 5
PACKET_SIZE = NUM_FLOATS * 4

# 与发送端保持一致（不使用5009/5010）
LOCAL_BIND_IP = "0.0.0.0"       # 绑定所有网卡，既能收本机也能收局域网
UDP_PORT = 5011

# 组播参数（与你发送端一致）
MCAST_GRP = "239.255.42.99"


def create_receiver_socket():
    """
    创建一个UDP socket：
    - 绑定 0.0.0.0:5011
    - 加入组播 239.255.42.99（用于局域网接收）
    - 非阻塞读取，只保留最新包
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

    # 允许端口复用（组播场景常用）
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # 绑定端口（Windows上组播一般也这样用）
    sock.bind((LOCAL_BIND_IP, UDP_PORT))

    # 加入组播组
    # INADDR_ANY 表示用系统默认网卡加入（多数情况下足够）
    mreq = struct.pack("4s4s", socket.inet_aton(MCAST_GRP), socket.inet_aton("0.0.0.0"))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    # 非阻塞
    sock.setblocking(False)
    return sock


def clear_console():
    # ANSI清屏 + 光标置顶
    print("\033[2J\033[H", end="")


def main(update_hz=50):
    sock = create_receiver_socket()
    latest = None
    latest_from = None
    last_rx_time = None

    dt = 1.0 / float(update_hz)

    print(f"Listening on UDP :{UDP_PORT}, multicast {MCAST_GRP}:{UDP_PORT}")
    print("Press Ctrl+C to quit.\n")
    time.sleep(0.5)

    try:
        while True:
            # 读光缓冲区，保留最新一帧
            while True:
                try:
                    data, addr = sock.recvfrom(2048)
                    if len(data) == PACKET_SIZE:
                        latest = struct.unpack("fffff", data)
                        latest_from = addr
                        last_rx_time = time.time()
                except BlockingIOError:
                    break

            clear_console()
            print(f"UDP Receiver (5 floats)  Port: {UDP_PORT}")
            print(f"Multicast Group: {MCAST_GRP}")
            print("-" * 50)

            if latest is None:
                print("No valid packet received yet.")
            else:
                age_ms = (time.time() - last_rx_time) * 1000.0 if last_rx_time else 0.0
                print(f"Latest packet from: {latest_from}   age: {age_ms:.1f} ms")
                print("Servo angles (deg):")
                for i, v in enumerate(latest):
                    print(f"  Servo[{i}]: {v:8.3f}")

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        try:
            sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    main(update_hz=50)
