#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import argparse

def generate_sine_lut(logic_precision, range_cycles=1.0, bit_width=16,
                      amplitude_scale=1.0, freq=None, sample_rate=None):
    """
    生成正弦波查找表（整数数组）

    参数：
        logic_precision : int   – 表的点数（也就是 0~360° 被切成的份数）
        range_cycles     : float – 输出相位范围（以周期为单位）
                                   1.0 = 0~2π（完整周期）
                                   0.5 = 0~π（半周期）

                                   0.25 = 0~π/2（四分之一周期）
        bit_width        : int   – 输出整数的位宽（如 16）
        amplitude_scale  : float – 幅度缩放因子（0~1），默认 1.0
        freq             : float – 信号频率（Hz），仅作注释，不影响表
        sample_rate      : float – 采样率（Hz），仅作注释，不影响表

    返回：
        list[int] – 量化后的整数列表
    """

    if logic_precision <= 0:
        raise ValueError("logic_precision must be positive")

    if range_cycles <= 0:
        raise ValueError("range_cycles must be positive")

    max_val = (1 << (bit_width - 1)) - 1   # 例如 16位 → 32767
    min_val = -(1 << (bit_width - 1))      # 例如 16位 → -32768

    lut = []
    for i in range(logic_precision):
        # 相位从 0 均匀增加到 range_cycles * 2π
        phase = 2.0 * math.pi * (i / logic_precision) * range_cycles
        raw = math.sin(phase)
        # 量化并缩放到整数范围
        quant = int(round(max_val * amplitude_scale * raw))
        # 钳制（防止溢出）
        quant = max(min_val, min(max_val, quant))
        lut.append(quant)
    return lut


def print_c_array(lut, name="sine_lut", bit_width=16, indent=4):
    """
    将列表打印成 C 语言的 static const 数组定义

    """
    prefix = " " * indent
    print(f"static const int{bit_width}_t {name}[{len(lut)}] = {{")
    for i in range(0, len(lut), 8):
        chunk = lut[i:i+8]
        line = ", ".join(f"{v:5d}" for v in chunk)
        print(f"{prefix}{line},")
    print("};")



# ---------- 命令行使用示例 ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成正弦波 LUT 表")
    parser.add_argument("-n", "--points", type=int, default=256,
                        help="逻辑精度（点数），默认 256")
    parser.add_argument("-r", "--range", type=float, default=1.0,
                        help="输出相位范围（周期数），0.5 表示 0~π，默认 1.0")
    parser.add_argument("-b", "--bit", type=int, default=16,
                        help="量化位宽，默认 16")
    parser.add_argument("-a", "--amplitude", type=float, default=1.0,
                        help="幅度缩放，默认 1.0")
    parser.add_argument("-f", "--freq", type=float, default=None,
                        help="信号频率（Hz），仅注释，不影响表")
    parser.add_argument("-s", "--sample-rate", type=float, default=None,
                        help="采样率（Hz），仅注释，不影响表")
    parser.add_argument("--name", type=str, default="sine_lut",
                        help="C 数组名称，默认 sine_lut")

    args = parser.parse_args()

    # 生成表
    lut = generate_sine_lut(
        logic_precision=args.points,
        range_cycles=args.range,
        bit_width=args.bit,
        amplitude_scale=args.amplitude,
        freq=args.freq,
        sample_rate=args.sample_rate
    )


    # 打印信息（如果传入了频率和采样率）
    if args.freq is not None and args.sample_rate is not None:
        print(f"// Signal: {args.freq} Hz @ {args.sample_rate} Hz sample rate")
        print(f"// Step size = {args.freq * (1 << 32) / args.sample_rate:.2f} (if using 32-bit DDS)\n")

    # 打印 C 数组
    print_c_array(lut, name=args.name, bit_width=args.bit)
