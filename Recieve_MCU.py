import socket
import struct
import time
import sys

# ================== 配置 ==================
UDP_IP = "127.0.0.1"
UDP_PORT = 5009
NUM_FLOATS = 16          # 必须和发送端一致
UPDATE_HZ = 50           # 显示刷新频率
# =========================================

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(False)   # 非阻塞，方便丢旧包

    latest = None

    print("Listening for UDP joint data...")
    time.sleep(1)

    try:
        while True:
            # ---------- 清空 UDP 接收缓冲区，只保留最新一帧 ----------
            while True:
                try:
                    data, _ = sock.recvfrom(4 * NUM_FLOATS)
                except BlockingIOError:
                    break

                if len(data) == 4 * NUM_FLOATS:
                    latest = struct.unpack("f" * NUM_FLOATS, data)

            # ---------- 刷新控制台显示 ----------
            if latest is not None:
                # ANSI：清屏 + 光标回到左上角（PowerShell 支持）
                sys.stdout.write("\033[H\033[J")

                print("UDP Joint Angles (deg)")
                print("======================")

                for i, v in enumerate(latest):
                    print(f"Joint {i:02d}: {v:7.2f}")

                print("\nPress Ctrl+C to exit")

            time.sleep(1.0 / UPDATE_HZ)

    except KeyboardInterrupt:
        print("\nExiting...")

    finally:
        sock.close()


if __name__ == "__main__":
    main()
