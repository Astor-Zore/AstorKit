#!/usr/bin/env python3

import wave
import struct

# 参数
sample_rate = 48000
duration = 10.0                 # 秒
num_frames = int(sample_rate * duration)   # 480000 帧（每帧包含左右声道）
sampwidth = 2                   # 16-bit = 2 字节
channels = 2

# 生成数据（小端字节序，符合 WAV 标准）
value = 1                       # 起始数值
data = bytearray()

for _ in range(num_frames):
    left = value
    right = value + 1

    # 将 16-bit 有符号整数限制在 -32768 ~ 32767 范围内（模 65536 回绕）
    # 这里先将值映射到 [0,65535] 无符号范围，再转成有符号
    def to_signed16(x):
        x = x % 65536            # 模 65536
        if x >= 32768:
            x -= 65536
        return x

    left_signed = to_signed16(left)
    right_signed = to_signed16(right)

    # 小端模式打包两个声道（标准 WAV 要求）
    data.extend(struct.pack('<hh', left_signed, right_signed))

    value += 2                   # 下一帧的起始值

# 写入 WAV 文件
with wave.open('output_increase.wav', 'w') as wav:
    wav.setnchannels(channels)
    wav.setsampwidth(sampwidth)
    wav.setframerate(sample_rate)
    wav.writeframes(data)

print("已生成 output_increase.wav，时长 10 秒，小端字节序，左右声道按 (1,2), (3,4), (5,6)... 递增。")
