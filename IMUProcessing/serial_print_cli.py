import serial
import serial.tools.list_ports

# ===================== 配置区 =====================
PORT = "COM7"  # 你自己的串口号
BAUD_RATE = 460800  # 必须和舵机/单片机一致！
# ===================================================

# 打开串口
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)

if ser.is_open:
    print(f"✅ 串口 {PORT} 已打开，可以发送指令")
else:
    print("❌ 串口打开失败")
    exit()

# 循环发送指令
try:
    while True:
        # 输入指令，格式：舵机ID-角度
        cmd = input("输入指令(例:1-90)：")

        # # 加上换行符，发送给串口
        # send_data = (cmd + "\n").encode("utf-8")

        # 发送！
        ser.write((cmd + "\n").encode('utf-8'))

        print(f"📤 已发送：{cmd}")

except KeyboardInterrupt:
    ser.close()
    print("\n🔌 串口已关闭")