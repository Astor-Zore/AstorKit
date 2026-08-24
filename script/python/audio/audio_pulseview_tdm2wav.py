#!/usr/bin/env python3
"""
pulseview_tdm2wav.py
将 PulseView 导出的 TDM 音频数据（8通道，通道号 0~7）转换为独立的 WAV 文件。
用法: python pulseview_tdm2wav.py -i 导出数据.txt -c 11333311 -o tdm_dmic_test
"""

import argparse
import os
import re
import struct
import sys
import wave

def parse_line(line):
    """
    从一行文本中提取通道号（0~7）和有符号 16 位采样值。
    兼容格式:
      Ch6: Channel 6: ff8a
      Ch0: Channel 0: 1234
    """
    # 匹配 "Channel X:" 或 "ChX:"，X 为数字
    match = re.search(r'Ch(?:annel\s+)?(\d+):.*?([0-9a-fA-F]+)\s*$', line)
    if not match:
        return None, None
    ch = int(match.group(1))
    hex_str = match.group(2)
    try:
        val = int(hex_str, 16)
        if val >= 0x8000:
            val -= 0x10000
    except ValueError:
        return None, None
    return ch, val

def main():
    parser = argparse.ArgumentParser(description='PulseView TDM 音频转 WAV')
    parser.add_argument('-i', '--input', required=True, help='输入文件')
    parser.add_argument('-c', '--divider', default='11111111',
                        help='8 位分频比，按通道 0~7 顺序（默认全 1）')
    parser.add_argument('-o', '--output', default='tdm_output',
                        help='输出文件夹路径，同时作为文件名前缀')
    args = parser.parse_args()

    # 检查分频比
    div_str = args.divider
    if len(div_str) != 8 or not all(d.isdigit() and d != '0' for d in div_str):
        print("错误：分频比必须是 8 位非零数字。", file=sys.stderr)
        sys.exit(1)
    dividers = [int(d) for d in div_str]

    # 读取文件
    try:
        with open(args.input, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"无法读取文件: {e}", file=sys.stderr)
        sys.exit(1)

    total = len(lines)
    print(f"共 {total} 行待解析")

    # data[0] 对应通道 0，data[7] 对应通道 7
    data = [[] for _ in range(8)]

    for i, line in enumerate(lines, 1):
        ch, val = parse_line(line)
        if ch is not None and 0 <= ch <= 7:   # 只接受通道 0~7
            data[ch].append(val)
        # 进度条
        percent = i * 100.0 / total
        sys.stdout.write(f"\r解析进度: {percent:.1f}%")
        sys.stdout.flush()
    sys.stdout.write("\n")

    lengths = [len(ch_data) for ch_data in data]
    print(f"各通道样本数 (通道0~7): {lengths}")

    # 有效通道的最小长度，避免因某通道缺失而清空所有数据
    valid_lengths = [l for l in lengths if l > 0]
    if not valid_lengths:
        print("错误：未解析到任何数据，请检查文件格式。", file=sys.stderr)
        sys.exit(1)

    min_len = min(valid_lengths)
    if len(set(valid_lengths)) > 1:
        print(f"注意：通道样本数不一致，按最短有效通道 {min_len} 截齐。")

    os.makedirs(args.output, exist_ok=True)
    prefix = os.path.basename(args.output.rstrip('/\\'))
    sample_rate_base = 48000

    for ch_idx in range(8):
        if lengths[ch_idx] == 0:
            print(f"通道 {ch_idx} 无数据，跳过。")
            continue

        raw = data[ch_idx][:min_len]        # 截齐
        div = dividers[ch_idx]               # 使用与通道对应的分频比
        decimated = raw[::div]
        out_rate = sample_rate_base // div

        # 文件名：通道 0 -> channel_1，通道 7 -> channel_8
        wav_name = f"{prefix}_channel_{ch_idx+1}.wav"
        wav_path = os.path.join(args.output, wav_name)

        with wave.open(wav_path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(out_rate)
            wf.writeframes(b''.join(struct.pack('<h', s) for s in decimated))

        print(f"已生成: {wav_path}  (采样率 {out_rate} Hz, 样本数 {len(decimated)})")

    print("转换完成。")

if __name__ == '__main__':
    main()
