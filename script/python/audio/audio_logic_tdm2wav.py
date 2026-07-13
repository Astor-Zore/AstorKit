#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import time
import wave
from array import array
from collections import defaultdict

def estimate_total_lines(filename):

    """快速估算文件总行数（仅用于进度条）"""
    try:
        with open(filename, 'rb') as f:
            head = f.read(2048)
            first_newline = head.find(b'\n')
            if first_newline == -1:
                return 0
            avg_line_len = first_newline + 1
            f.seek(0, os.SEEK_END)

            file_size = f.tell()
            return file_size // avg_line_len

    except:
        return 0

def convert_logic_to_wav(csv_file, wav_file, samplerate=None, split=False, channels=None):
    print(f"正在处理: {csv_file}")
    start_time = time.time()


    total_est = estimate_total_lines(csv_file)
    if total_est > 0:
        print(f"估算总行数: {total_est:,}")

    # 使用 lambda 返回带类型码的 array
    channels_data = defaultdict(lambda: array('h'))
    processed = 0
    last_report = 0
    report_interval = 50000

    with open(csv_file, 'r') as f:
        for line in f:
            processed += 1
            parts = line.split(',')
            if len(parts) < 4:
                continue


            try:
                ch = int(parts[1].strip())
                val = int(parts[2].strip())
            except ValueError:
                continue  # 跳过标题行或异常行

            # 转为有符号 16 位
            if val >= 32768:
                val -= 65536

            # 如果用户指定了通道列表，只收集这些通道
            if channels is not None and ch not in channels:
                continue

            channels_data[ch].append(val)


            # 进度条 (单行更新)
            if processed - last_report >= report_interval:
                last_report = processed
                elapsed = time.time() - start_time
                if total_est > 0 and processed > 0:
                    percent = (processed / total_est) * 100
                    eta = (elapsed / processed) * (total_est - processed)
                    progress_str = (f"\r已处理 {processed:,} 行 ({percent:.1f}%)  "
                                    f"耗时 {elapsed:.1f}s  ETA {eta:.1f}s   ")
                else:
                    progress_str = f"\r已处理 {processed:,} 行  耗时 {elapsed:.1f}s   "
                sys.stdout.write(progress_str)
                sys.stdout.flush()

    sys.stdout.write('\n')


    if not channels_data:
        print("错误：未解析到任何有效数据。")
        sys.exit(1)

    # 自动过滤无效通道：保留数据长度 ≥ 最长长度 * 1% 的通道
    max_len = max(len(arr) for arr in channels_data.values())
    threshold = max_len * 0.01  # 1% 阈值
    valid_channels = {ch: arr for ch, arr in channels_data.items() if len(arr) >= threshold}

    if not valid_channels:
        print("错误：过滤后无有效通道，请检查数据。")

        sys.exit(1)


    # 如果用户指定了通道但过滤后为空，给出提示
    if channels is not None:
        # 只保留用户指定且通过过滤的通道
        final_channels = {ch: arr for ch, arr in valid_channels.items() if ch in channels}
        if not final_channels:
            print(f"错误：指定的通道 {channels} 均无有效数据。")

            sys.exit(1)
        valid_channels = final_channels

    else:
        # 如果未指定，但过滤后通道数远少于原始数，给出提示
        if len(valid_channels) < len(channels_data):
            discarded = set(channels_data.keys()) - set(valid_channels.keys())
            print(f"⚠️  自动丢弃无效通道: {sorted(discarded)} (数据量过少)")

    sorted_channels = sorted(valid_channels.keys())
    num_channels = len(sorted_channels)
    print(f"有效通道: {sorted_channels} (共{num_channels}个)")

    # 对齐所有通道的长度（取最小长度）
    min_len = min(len(arr) for arr in valid_channels.values())
    if min_len == 0:
        print("错误：所有通道数据为空。")
        sys.exit(1)

    for ch in sorted_channels:

        del valid_channels[ch][min_len:]

    # 采样率处理

    if samplerate is None:
        samplerate = 48000
        print("⚠️  未指定采样率，使用默认 48000 Hz")
    else:
        print(f"使用采样率: {samplerate} Hz")


    if split:
        base, ext = os.path.splitext(wav_file)
        for ch in sorted_channels:
            single_file = f"{base}_ch{ch}{ext}"

            data = valid_channels[ch]
            with wave.open(single_file, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(samplerate)
                w.writeframes(data.tobytes())
            print(f"✅ 已写入通道 {ch}: {single_file}")

    else:
        total_samples = min_len * num_channels
        stereo = array('h', [0]) * total_samples
        for idx, ch in enumerate(sorted_channels):
            stereo[idx::num_channels] = valid_channels[ch]

        with wave.open(wav_file, 'wb') as w:
            w.setnchannels(num_channels)
            w.setsampwidth(2)
            w.setframerate(samplerate)
            w.writeframes(stereo.tobytes())

        elapsed = time.time() - start_time
        print(f"✅ 转换完成！")
        print(f"   输出文件: {wav_file}")
        print(f"   通道数:   {num_channels}")
        print(f"   采样点数: {min_len:,}")

        print(f"   总耗时:   {elapsed:.2f} 秒")
        print(f"   文件大小: {os.path.getsize(wav_file) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(

        description="逻辑分析仪 CSV → 多通道WAV 转换器（自动过滤无效通道）"
    )
    parser.add_argument("input", help="输入的 CSV 文件名")
    parser.add_argument("-o", "--output", default="output.wav", help="输出 WAV 文件名（多通道模式）或前缀（拆分模式）")
    parser.add_argument("-r", "--samplerate", type=int, default=None,
                        help="采样率 (Hz)，如 48000，不指定则使用默认 48000")
    parser.add_argument("--split", action="store_true",
                        help="拆分每个通道为独立的单声道WAV文件")
    parser.add_argument("--channels", type=str, default=None,
                        help="手动指定要提取的通道号，逗号分隔，如 '1,2,3,4'")

    args = parser.parse_args()

    channels_list = None
    if args.channels:
        channels_list = [int(x.strip()) for x in args.channels.split(',')]

    convert_logic_to_wav(args.input, args.output, args.samplerate, args.split, channels_list)
