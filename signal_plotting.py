import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

port = '/dev/cu.usbmodem11401'  # 사용자 환경에 맞게 수정
baud_rate = 9600
max_points = 500  # x축에 표시할 데이터 포인트 수 

ser = serial.Serial(port, baud_rate)
fig, ax = plt.subplots(figsize=(12, 6))
xs = np.arange(max_points)
ys = np.zeros(max_points)
line, = ax.plot(xs, ys)

ax.set_xlim(0, max_points)
ax.set_ylim(0, 1023)  # 아두이노 아날로그 값 범위, 필요에 따라 조정
ax.set_title('Arduino Serial Data')
ax.set_xlabel('Sample')
ax.set_ylabel('Value')
ax.grid(True)

def update(frame):
    global ys
    if ser.in_waiting:
        try:
            data = ser.readline().decode().strip()
            value = float(data)
            ys = np.append(ys[1:], value)
            line.set_ydata(ys)
        except:
            pass
    return line,

ani = FuncAnimation(fig, update, interval=50, blit=True)
plt.tight_layout()
plt.show()

ser.close()