#!/usr/bin/env python3
"""
将 Kingst VIS 导出的 TDM 数据（64bit×2 模式）转换为多声道 WAV 文件。

导出时请确保：
- I²S 解码器帧位宽设为 64
- 同时使用两个“声道”采集同一 TDM 数据线
- CSV 格式：Time [s],Channel,Value
  其中 Channel 交替为 2 和 1，Value 为 64 位十六进制数（如 0x000...）

用法示例：
    python tdm_to_wav.py -i data.csv -b 16 -c 8 -s 48000 -o output.wav
"""

import csv
import struct
import sys
import os
import argparse
from typing import List, Tuple, Optional


def read_csv_and_pair(
    filepath: str,
    verbose: bool = True,
) -> Tuple[List[int], List[float], int]:
    """
    读取 CSV，将相邻的 Channel 2 和 Channel 1 合并为 128 位帧。
    返回：帧时间戳列表，合并后的整数值列表，有效帧数
    """
    frames_time = []   # 每帧对应的时间（取 Channel 2 的时刻）
    frames_val = []    # 合并后的 128 位整数

    with open(filepath, 'r', newline='') as f:
        reader = csv.reader(f)
        # 跳过标题行（如果第一行第一列不是数字，就认为是标题）
        first_row = next(reader, None)
        if first_row is None:
            print("错误：CSV 文件为空")
            sys.exit(1)
        try:
            float(first_row[0])
            # 是数字，说明没有标题行，直接处理第一行
            rows = [first_row] + list(reader)
        except ValueError:
            # 不是数字，则跳过标题行
            rows = list(reader)

    total_rows = len(rows)
    if total_rows < 2:
        print("错误：CSV 数据不足，至少需要 2 行（一帧）")
        sys.exit(1)

    expected_next_ch = 2       # 先期待 Channel 2
    saved_ch2_time = None
    saved_ch2_val = None
    pair_count = 0

    def show_progress(current, total):
        """单行进度条"""
        percent = min(current / total * 100, 100)
        bar_len = 30
        filled = int(bar_len * current / total)
        bar = '█' * filled + '░' * (bar_len - filled)
        sys.stdout.write(f"\r处理进度：{bar} {percent:5.1f}% ({current}/{total})")
        sys.stdout.flush()

    for idx, row in enumerate(rows, 1):
        if verbose and idx % 100 == 0:
            show_progress(idx, total_rows)

        try:
            time_val = float(row[0])
            ch = int(row[1])
            hex_str = row[2].strip()
            if hex_str.startswith('0x') or hex_str.startswith('0X'):
                hex_str = hex_str[2:]
            # 64 位十六进制，最多 16 个字符
            int_val = int(hex_str, 16)
        except (ValueError, IndexError) as e:
            print(f"\n警告：第 {idx} 行解析失败，已跳过 ({e})")
            continue

        if ch == 2:
            if expected_next_ch == 1 and saved_ch2_val is not None:
                # 连续两个 Channel 2，丢弃之前未配对的那个
                print(f"\n警告：第 {idx} 行发现连续 Channel 2，之前未配对的 Channel 2 已丢弃")
            saved_ch2_time = time_val
            saved_ch2_val = int_val
            expected_next_ch = 1
        elif ch == 1:
            if expected_next_ch == 1 and saved_ch2_val is not None:
                # 成功配对一帧
                combined = (saved_ch2_val << 64) | int_val
                frames_time.append(saved_ch2_time)   # 用 Ch2 的时间作为帧时间
                frames_val.append(combined)
                pair_count += 1
                saved_ch2_val = None
                expected_next_ch = 2
            else:
                # 没有前导的 Channel 2，丢弃这个 Channel 1
                if expected_next_ch == 2:
                    print(f"\n警告：第 {idx} 行是 Channel 1，但之前缺少 Channel 2，已丢弃")
        else:
            print(f"\n警告：第 {idx} 行通道编号为 {ch}（不是 1 或 2），已跳过")

    if verbose:
        show_progress(total_rows, total_rows)
        print()   # 换行

    if saved_ch2_val is not None:
        print("警告：最后存在未配对的 Channel 2 行，已丢弃")

    print(f"有效帧数：{pair_count}")
    return frames_time, frames_val, pair_count


def split_channels(
    frames_val: List[int],
    bit_width: int,
    num_channels: int,
) -> List[List[int]]:
    """
    将每个 128 位帧按位宽拆分为各通道的样本值（有符号 16 位）。
    返回：二维列表，第一维为通道索引，第二维为样本值列表。
    """
    total_bits_exported = 128   # 两个 64 位合并
    total_bits_expected = bit_width * num_channels
    if total_bits_exported != total_bits_expected:
        print(f"错误：导出总位宽 {total_bits_exported} 不等于 位宽×通道数 = {total_bits_expected}")
        sys.exit(1)

    # 初始化每个通道的列表
    channels = [[] for _ in range(num_channels)]

    # 最大正负值
    max_val = (1 << bit_width) - 1
    half = 1 << (bit_width - 1)

    for combined in frames_val:
        for i in range(num_channels):
            # 从最高位开始，依次提取各通道
            shift = total_bits_expected - (i + 1) * bit_width
            raw = (combined >> shift) & max_val
            # 转换为有符号整数（假设补码）
            if raw >= half:
                raw -= (1 << bit_width)
            # 若位宽不是 16，需要量化到 16 位
            if bit_width == 16:
                sample = raw
            else:
                # 归一化并转 16 位有符号范围
                normalized = raw / (half)   # -1.0 ~ 1.0
                sample = int(max(-32768, min(32767, normalized * 32767)))
            channels[i].append(sample)

    return channels


def estimate_sample_rate(timestamps: List[float]) -> float:
    """从时间戳估算采样率（帧之间的平均周期）"""
    if len(timestamps) < 2:
        print("错误：帧数不足，无法估算采样率，请手动指定 -s/--sample-rate")
        sys.exit(1)
    diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    avg_diff = sum(diffs) / len(diffs)
    if avg_diff <= 0:
        print("错误：时间戳不是严格递增的，无法估算采样率")
        sys.exit(1)
    fs = 1.0 / avg_diff
    print(f"从时间戳估算的采样率：{fs:.2f} Hz（如需精确请用 -s 指定）")
    return fs


def write_multichannel_wav(
    filepath: str,
    sample_rate: int,
    channels: List[List[int]],
) -> None:
    """
    写多声道 PCM 16-bit WAV 文件（纯 Python 实现）。
    channels: 每个元素是一个通道的样本列表，长度必须相同。
    """
    if not channels:
        print("错误：没有通道数据，无法生成 WAV")
        sys.exit(1)

    num_channels = len(channels)
    num_samples = len(channels[0])
    for i, ch in enumerate(channels):
        if len(ch) != num_samples:
            print(f"错误：通道 {i+1} 样本数 {len(ch)} 与通道 1 的 {num_samples} 不一致")
            sys.exit(1)

    byte_rate = sample_rate * num_channels * 2  # 16bit = 2 bytes
    block_align = num_channels * 2
    data_size = num_samples * block_align

    with open(filepath, 'wb') as f:
        # RIFF header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))   # 文件大小 - 8
        f.write(b'WAVE')
        # fmt chunk
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))               # chunk size
        f.write(struct.pack('<H', 1))                # PCM format
        f.write(struct.pack('<H', num_channels))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', byte_rate))
        f.write(struct.pack('<H', block_align))
        f.write(struct.pack('<H', 16))               # bits per sample
        # data chunk
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        # 交错写入样本
        for sample_idx in range(num_samples):
            for ch in range(num_channels):
                val = channels[ch][sample_idx]
                # 限制在 16 位有符号范围
                val = max(-32768, min(32767, val))
                f.write(struct.pack('<h', val))

    print(f"已生成 WAV 文件：{filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="将 Kingst VIS 导出的 TDM 数据（64bit×2 模式）转换为多声道 WAV"
    )
    parser.add_argument('-i', '--input', required=True, help='输入 CSV 文件路径')
    parser.add_argument('-b', '--bit-width', type=int, required=True, help='每个通道的位宽，例如 16')
    parser.add_argument('-c', '--channels', type=int, required=True, help='通道总数，例如 8')
    parser.add_argument('-s', '--sample-rate', type=float, default=None,
                        help='采样率 (Hz)，若不提供则从时间戳自动估算')
    parser.add_argument('-o', '--output', default=None, help='输出 WAV 文件名，默认由输入文件名生成')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"错误：文件不存在 - {args.input}")
        sys.exit(1)

    if args.bit_width <= 0 or args.channels <= 0:
        print("错误：位宽和通道数必须为正整数")
        sys.exit(1)

    # 1. 读取 CSV 并配对
    times, values, pair_count = read_csv_and_pair(args.input)
    if pair_count == 0:
        print("没有有效的帧，程序退出。")
        sys.exit(1)

    # 2. 拆分通道
    channels_data = split_channels(values, args.bit_width, args.channels)

    # 3. 采样率
    if args.sample_rate is None:
        sample_rate = int(round(estimate_sample_rate(times)))
    else:
        sample_rate = int(round(args.sample_rate))
    print(f"使用采样率：{sample_rate} Hz")

    # 4. 输出文件名
    output = args.output
    if output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        output = f"{base}.wav"

    # 5. 写 WAV
    write_multichannel_wav(output, sample_rate, channels_data)


if __name__ == '__main__':
    main()
