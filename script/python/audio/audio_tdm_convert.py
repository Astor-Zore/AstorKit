#!/usr/bin/env python3
"""
convert.py — Multi-channel WAV decimation tool.

Extracts each channel from a multi-channel 48kHz WAV file and downsamples
to 16kHz (or other rate) by keeping 1 of every N samples, where N is the
upsampling factor for that channel.  Useful when 16kHz audio was upsampled
to 48kHz by sample repetition and recorded as a multi-channel WAV.

Usage:
  convert input.wav -c 8 -b 16 -s 48000 -x 33333333 -o output/
"""

import argparse
import os
import sys
import wave
import struct
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
class Color:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def _ts() -> str:
    """Return a dimmed timestamp prefix for log lines."""
    from datetime import datetime
    return f"{Color.DIM}[{datetime.now().strftime('%H:%M:%S')}]{Color.RESET}"


def log_info(msg: str) -> None:
    print(f" {Color.BLUE}\u25b6{Color.RESET} {msg}")


def log_ok(msg: str) -> None:
    print(f" {Color.GREEN}\u2714{Color.RESET} {msg}")


def log_warn(msg: str) -> None:
    print(f" {Color.YELLOW}\u26a0{Color.RESET} {msg}")


def log_error(msg: str) -> None:
    print(f" {Color.RED}\u2718{Color.RESET} {msg}")


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------
def read_wav(path: str):
    """Open a WAV file and return (wave_obj, raw_bytes)."""
    w = wave.open(path, "rb")
    data = w.readframes(w.getnframes())
    return w, data


def write_wav(path: str, sample_rate: int, sample_width: int,
              channels: int, raw_data: bytes) -> None:
    """Write raw PCM data to a mono WAV file."""
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(raw_data)


def channels_from_factors(s: str) -> int:
    """Return the number of channels implied by the -x string."""
    return len(s.strip())


# ---------------------------------------------------------------------------
# Decimation
# ---------------------------------------------------------------------------
def decimate_channel(raw_data: bytes, sample_width: int,
                     total_channels: int, channel_index: int,
                     factor: int) -> bytes:
    """
    Extract channel_index from interleaved multi-channel PCM, keeping every
    *factor*-th sample (simple decimation, starting at sample 0).
    """
    if factor <= 0:
        raise ValueError(f"Decimation factor must be > 0, got {factor}")

    fmt = {1: "<b", 2: "<h", 4: "<i"}.get(sample_width)
    if fmt is None:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    frame_size = total_channels * sample_width
    total_frames = len(raw_data) // frame_size
    out_samples = []

    for i in range(0, total_frames, factor):
        offset = i * frame_size + channel_index * sample_width
        sample = struct.unpack(fmt, raw_data[offset:offset + sample_width])[0]
        out_samples.append(sample)

    return struct.pack(f"<{len(out_samples)}{fmt[1]}", *out_samples)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decimate multi-channel WAV into per-channel mono files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        examples:
          %(prog)s dmicall.wav -c 8 -b 16 -s 48000 -x 33333333 -o out/
              Decimate all 8 channels by factor 3 (48k -> 16k).

          %(prog)s input.wav -c 4 -b 16 -s 48000 -x 3322 -o out/
              Channels 0-1 decimated by 3, channels 2-3 by 2.

        notes:
          - The -x string must have exactly one digit per channel.
          - Each digit is the decimation factor: keep 1 of every N samples.
          - Output sample rate = input_sample_rate / factor.
          - Output files are named <input_stem>_ch<index>.wav.
        """),
    )

    parser.add_argument(
        "input",
        help="Path to the input multi-channel WAV file.",
    )
    parser.add_argument(
        "-c", "--channels", type=int, required=True,
        help="Number of channels in the input WAV file.",
    )
    parser.add_argument(
        "-b", "--bits", type=int, required=True,
        help="Bits per sample (e.g. 16).",
    )
    parser.add_argument(
        "-s", "--sample-rate", type=int, required=True,
        help="Sample rate of the input WAV file (e.g. 48000).",
    )
    parser.add_argument(
        "-x", "--factors", type=str, required=True,
        help="Decimation factors per channel, one digit per channel "
             "(e.g. '33333333' for 8 channels all factor 3).",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Directory to write output per-channel WAV files.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Validate arguments
    # ------------------------------------------------------------------
    factors_str = args.factors.strip()
    num_channels = args.channels

    if len(factors_str) != num_channels:
        log_error(f"-x string has {len(factors_str)} digits but "
                  f"-c specifies {num_channels} channels")
        sys.exit(1)

    factors: list[int] = []
    for ch in factors_str:
        if not ch.isdigit() or int(ch) == 0:
            log_error(f"Invalid factor '{ch}' in -x string. "
                      f"Each character must be a digit 1-9.")
            sys.exit(1)
        factors.append(int(ch))

    sample_width = args.bits // 8
    if args.bits % 8 != 0:
        log_error(f"Bits per sample must be a multiple of 8, got {args.bits}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Validate input file
    # ------------------------------------------------------------------
    input_path = Path(args.input)
    if not input_path.is_file():
        log_error(f"Input file not found: {input_path}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Print header
    # ------------------------------------------------------------------
    print()
    print(f" {Color.BOLD}{Color.CYAN}"
          f"\u2554\u2550\u2550 CLOVER Audio Channel Decimator \u2550\u2550\u2557"
          f"{Color.RESET}")
    print(f" {Color.BOLD}{Color.CYAN}\u255a\u2550\u2550\u2550\u2550\u2550"
          f"\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
          f"\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
          f"\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
          f"\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d"
          f"{Color.RESET}")
    print()
    log_info(f"Input file   : {Color.BOLD}{input_path}{Color.RESET}")
    log_info(f"Output dir   : {Color.BOLD}{args.output}{Color.RESET}")
    log_info(f"Channels     : {Color.BOLD}{num_channels}{Color.RESET}")
    log_info(f"Sample rate  : {Color.BOLD}{args.sample_rate} Hz{Color.RESET}")
    log_info(f"Bit depth    : {Color.BOLD}{args.bits}-bit{Color.RESET}")
    log_info(f"Factors (-x) : {Color.BOLD}{factors_str}{Color.RESET}")
    print()

    # ------------------------------------------------------------------
    # 4. Read WAV
    # ------------------------------------------------------------------
    log_info("Reading input WAV file ...")
    wav_in, raw_data = read_wav(str(input_path))

    actual_channels = wav_in.getnchannels()
    actual_rate = wav_in.getframerate()
    actual_width = wav_in.getsampwidth()
    actual_frames = wav_in.getnframes()
    wav_in.close()

    log_info(f"  Actual: {actual_channels} ch, {actual_rate} Hz, "
             f"{actual_width * 8}-bit, {actual_frames} frames "
             f"({actual_frames / actual_rate:.2f}s)")

    if actual_channels != num_channels:
        log_warn(f"Input file has {actual_channels} channels, "
                 f"but -c specifies {num_channels}. Using actual channel count.")
        # Recalculate factors if needed — but keep the user's specified count
        # for the -x string validation. We'll just use what the file has.

    if actual_width != sample_width:
        log_error(f"Input file is {actual_width * 8}-bit, "
                  f"but -b specifies {args.bits}-bit.")
        sys.exit(1)

    if actual_rate != args.sample_rate:
        log_warn(f"Input file is {actual_rate} Hz, "
                 f"but -s specifies {args.sample_rate} Hz. "
                 f"Using actual rate {actual_rate} Hz.")

    # ------------------------------------------------------------------
    # 5. Prepare output directory
    # ------------------------------------------------------------------
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 6. Decimate each channel
    # ------------------------------------------------------------------
    total_size = 0
    input_stem = input_path.stem
    sample_rate_out = actual_rate  # will be overridden per channel

    print()
    log_info(f"{'Channel':>8}  {'Factor':>6}  {'In Rate':>8}  "
             f"{'Out Rate':>8}  {'In Frames':>10}  "
             f"{'Out Frames':>10}  {'Output File'}")
    sep = "\u2500" * 88
    print(f" {Color.DIM}{sep}{Color.RESET}")

    for ch_idx in range(actual_channels):
        factor = factors[ch_idx] if ch_idx < len(factors) else factors[-1]
        out_rate = actual_rate // factor

        out_samples = decimate_channel(
            raw_data, actual_width, actual_channels, ch_idx, factor
        )
        out_frames = len(out_samples) // actual_width

        out_name = f"{input_stem}_ch{ch_idx}.wav"
        out_path = output_dir / out_name

        write_wav(str(out_path), out_rate, actual_width, 1, out_samples)
        total_size += out_path.stat().st_size

        print(f" {Color.GREEN}ch{ch_idx:>5}{Color.RESET}  "
              f"{factor:>6}  {actual_rate:>8}  {out_rate:>8}  "
              f"{actual_frames:>10}  {out_frames:>10}  "
              f"{Color.BOLD}{out_name}{Color.RESET}")

    sep = "\u2500" * 88
    print(f" {Color.DIM}{sep}{Color.RESET}")
    print()
    log_ok(f"Decimation complete — {actual_channels} channel(s) written "
           f"to {Color.BOLD}{output_dir}/{Color.RESET}")
    log_info(f"Total output size: {Color.BOLD}"
             f"{total_size / 1024:.1f} KB{Color.RESET}")
    print()


if __name__ == "__main__":
    main()