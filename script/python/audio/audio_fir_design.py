#!/usr/bin/env python3
"""
嵌入式音频重采样 FIR 系数设计器 (专业版 v2.0)
=================================================
修复:
  - 多相分解: stride 抽取 (非 reshape)
  - 通带比率: 0~1.0 占有效带宽比例 (非 0~0.5)
  - 增益补偿: 插值 ×L, 抽取 ×1
新增:
  - 质量预设 (语音/通用/高品质)
  - Q15/Q31 自动移位计算
  - 每相位 DC 增益验证
  - 性能验证警告
  - 通带放大视图
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# ============================================================
#  1. 滤波器规格计算
# ============================================================

def compute_specs(fs_in, fs_out, passband_fraction=0.9):
    """
    计算重采样滤波器的归一化频率参数。

    参数:
        fs_in:  输入采样率 (Hz)
        fs_out: 输出采样率 (Hz)
        passband_fraction: 通带占有效带宽的比例 (0~1.0)
                          0.9  = 保留 90% 有效带宽 (推荐)
                          0.85 = 保留 85% (更低开销)
                          0.95 = 保留 95% (更高音质)

    返回:
        L, M, fs_inter, f_pass, f_stop,
        f_pass_hz, f_stop_hz, useful_nyq, mode, num_phases
    """
    gcd = np.gcd(fs_in, fs_out)
    L = fs_out // gcd
    M = fs_in // gcd
    fs_inter = fs_in * L

    useful_nyq = 0.5 * min(fs_in, fs_out)

    f_pass = passband_fraction * useful_nyq / fs_inter
    f_stop = useful_nyq / fs_inter
    f_pass_hz = passband_fraction * useful_nyq
    f_stop_hz = useful_nyq

    if L > 1 and M == 1:
        mode = 'interpolation'
        num_phases = L
    elif L == 1 and M > 1:
        mode = 'decimation'
        num_phases = M       # 抽取用 M 相位
    else:
        mode = 'rational'
        num_phases = L

    return (L, M, fs_inter, f_pass, f_stop,
            f_pass_hz, f_stop_hz, useful_nyq, mode, num_phases)


# ============================================================
#  2. 抽头数估算
# ============================================================

def estimate_taps(A_stop, f_pass, f_stop, num_phases):
    """Kaiser 窗经验公式估算所需抽头数"""
    transition = f_stop - f_pass
    if transition <= 0:
        return num_phases

    A = max(A_stop, 21)
    delta_omega = 2 * np.pi * transition
    N = (A - 8) / (2.285 * delta_omega) + 1
    N = int(np.ceil(N))

    N = int(np.ceil(N / num_phases)) * num_phases
    if N % 2 == 0:
        N += num_phases
    return N


# ============================================================
#  3. 滤波器设计
# ============================================================

def design_filter(num_taps, f_pass, f_stop, method='remez',
                  A_stop=60, stopband_weight=10.0):
    """设计原型低通 FIR 滤波器 (DC 增益 = 1.0)"""
    num_taps = int(num_taps)
    cutoff = (f_pass + f_stop) / 2

    if method == 'remez':
        coeff = signal.remez(num_taps, [0, f_pass, f_stop, 0.5], [1, 0],
                             weight=[1.0, stopband_weight], fs=1.0)
    elif method == 'kaiser':
        A = max(A_stop, 21)
        if A > 50:
            beta = 0.1102 * (A - 8.7)
        elif A >= 21:
            beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21)
        else:
            beta = 0.0
        coeff = signal.firwin(num_taps, cutoff,
                              window=('kaiser', beta), fs=1.0)
    elif method == 'hamming':
        coeff = signal.firwin(num_taps, cutoff, window='hamming', fs=1.0)
    elif method == 'blackman':
        coeff = signal.firwin(num_taps, cutoff, window='blackman', fs=1.0)
    else:
        raise ValueError(f"未知设计方法: {method}")

    return coeff


# ============================================================
#  4. 多相分解 (核心修复: stride 而非 reshape)
# ============================================================

def polyphase_decompose(coeff, num_phases):
    """
    多相分解: phase i 取 coeff[i], coeff[i+L], coeff[i+2L], ...

    reshape(L, -1) 取连续元素 → DC 增益不均 (错误!)
    stride 抽取取间隔元素   → DC 增益均衡 (正确!)
    """
    N = len(coeff)
    if N % num_phases != 0:
        coeff = np.pad(coeff, (0, num_phases - (N % num_phases)))
        N = len(coeff)

    phase_len = N // num_phases
    poly = np.zeros((num_phases, phase_len))
    for i in range(num_phases):
        poly[i] = coeff[i::num_phases]   # stride 抽取
    return poly


# ============================================================
#  5. 滤波器分析
# ============================================================

def analyze_filter(coeff, f_pass, f_stop, fs_inter, num_phases, mode):
    """全面分析滤波器性能指标"""
    w, h = signal.freqz(coeff, worN=65536, fs=1.0)
    mag = np.abs(h)
    mag_db = 20 * np.log10(np.maximum(mag, 1e-16))
    phase = np.unwrap(np.angle(h))

    # 通带
    pb_mask = w <= f_pass
    pb_ripple = (np.max(mag_db[pb_mask]) - np.min(mag_db[pb_mask])
                 if np.any(pb_mask) else 0.0)

    pb_detail = {}
    for ratio in [0.5, 0.8, 1.0]:
        mask = w <= f_pass * ratio
        if np.any(mask):
            pb_detail[f'{int(ratio*100)}%'] = (
                np.max(mag_db[mask]) - np.min(mag_db[mask]))

    # 阻带
    sb_mask = w >= f_stop
    sb_atten = -np.max(mag_db[sb_mask]) if np.any(sb_mask) else 0.0

    # 截止频率
    idx_3db = np.where(mag_db <= -3.0)[0]
    f_3db = w[idx_3db[0]] if len(idx_3db) > 0 else None

    # DC 增益
    dc_gain = np.abs(h[0])

    # 群延迟
    pb_w = w[pb_mask]
    pb_phase = phase[pb_mask]
    if len(pb_w) > 2:
        gd = -np.diff(pb_phase) / np.diff(pb_w * 2 * np.pi)
        gd_samples = np.median(gd)
    else:
        gd_samples = 0.0

    # 多相 DC 增益
    poly = polyphase_decompose(coeff, num_phases)
    phase_sums = np.sum(poly, axis=1)
    total_dc = np.sum(phase_sums)

    return {
        'w': w, 'mag_db': mag_db, 'num_taps': len(coeff),
        'pb_ripple': pb_ripple, 'pb_detail': pb_detail,
        'sb_atten': sb_atten, 'f_3db': f_3db,
        'dc_gain': dc_gain, 'group_delay': gd_samples,
        'phase_sums': phase_sums, 'total_dc': total_dc,
        'f_pass': f_pass, 'f_stop': f_stop,
        'transition_bw': f_stop - f_pass,
    }


# ============================================================
#  6. Q15 / Q31 定点缩放
# ============================================================

def scale_to_fixed(poly, mode, num_phases, bits=15):
    """
    将浮点系数缩放为定点格式。

    插值: 每相位和 = 2^shift (每个输出只用 1 个相位)
    抽取: 总和 = 2^shift     (用户代码对所有相位求和)
    """
    max_val = (1 << bits) - 1

    if mode in ('interpolation', 'rational'):
        ref = np.mean(np.sum(poly, axis=1))
    else:
        ref = np.sum(poly)

    max_abs_coeff = np.max(np.abs(poly))
    best_shift = 1
    for shift in range(bits, 0, -1):
        scale = (1 << shift) / ref
        if max_abs_coeff * scale < max_val:
            best_shift = shift
            break

    scale = (1 << best_shift) / ref
    if bits == 15:
        poly_q = np.round(poly * scale).astype(np.int16)
    else:
        poly_q = np.round(poly * scale).astype(np.int32)

    return poly_q, best_shift, np.sum(poly_q, axis=1)


# ============================================================
#  7. 可视化
# ============================================================

def plot_response(analysis, fs_inter, title):
    """绘制幅频响应, 含通带放大视图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    w = analysis['w']
    mag_db = analysis['mag_db']
    f_pass = analysis['f_pass']
    f_stop = analysis['f_stop']
    freq_hz = w * fs_inter

    # ---- 全景幅频响应 ----
    ax1.plot(freq_hz, mag_db, 'b', linewidth=0.8)
    ax1.axvspan(0, f_pass * fs_inter, alpha=0.08, color='green', label='Passband')
    ax1.axvspan(f_stop * fs_inter, 0.5 * fs_inter, alpha=0.08, color='red', label='Stopband')
    ax1.axvline(f_pass * fs_inter, color='green', linestyle='--', alpha=0.5, lw=0.8)
    ax1.axvline(f_stop * fs_inter, color='red', linestyle='--', alpha=0.5, lw=0.8)

    if analysis['f_3db']:
        ax1.axvline(analysis['f_3db'] * fs_inter, color='orange',
                    linestyle='--', alpha=0.7, lw=1,
                    label=f"-3dB @ {analysis['f_3db']*fs_inter:.0f} Hz")

    ax1.axhline(-3, color='orange', linestyle=':', alpha=0.3)
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.3)

    ax1.set_title(title, fontsize=11)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_ylim([-100, 5])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=8)

    # ---- 通带放大 ----
    pb_mask = w <= f_pass * 1.15
    ax2.plot(freq_hz[pb_mask], mag_db[pb_mask], 'b', linewidth=1.2)
    ax2.axvspan(0, f_pass * fs_inter, alpha=0.08, color='green')
    ax2.axvline(f_pass * fs_inter, color='green', linestyle='--', alpha=0.5)

    ripple = analysis['pb_ripple']
    ax2.set_title(f'Passband Zoom (ripple: {ripple:.4f} dB)', fontsize=11)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude (dB)')
    ylim = max(0.5, ripple * 3 + 0.2)
    ax2.set_ylim([-ylim, ylim])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
#  8. C 代码生成
# ============================================================

def generate_c_code(poly_q, L, M, num_phases, fs_in, fs_out,
                    shift, mode, method, analysis):
    """生成 C 语言系数表代码"""
    phase_len = poly_q.shape[1]
    dtype = 'int16_t' if poly_q.dtype == np.int16 else 'int32_t'
    w = 6 if poly_q.dtype == np.int16 else 12

    if mode == 'interpolation':
        prefix, ratio = 'UPSAMPLE', f'{L}X'
    elif mode == 'decimation':
        prefix, ratio = 'DOWNSAMPLE', f'{M}X'
    else:
        prefix, ratio = 'RESAMPLE', f'{L}_{M}'
    macro = f'{prefix}_{ratio}'

    lines = []
    lines.append("/*")
    lines.append(f" * FIR Polyphase Resampler: {fs_in} Hz -> {fs_out} Hz")
    lines.append(f" * Mode: {mode}, Phases: {num_phases}, Taps/Phase: {phase_len}")
    lines.append(f" * Method: {method}, Shift: >> {shift}")
    lines.append(f" *")
    lines.append(f" * Performance:")
    lines.append(f" *   Passband ripple:  {analysis['pb_ripple']:.4f} dB")
    lines.append(f" *   Stopband atten:   {analysis['sb_atten']:.2f} dB")
    if analysis['f_3db']:
        lines.append(f" *   -3dB cutoff:      {analysis['f_3db']*fs_in*L:.1f} Hz")
    lines.append(f" *   Group delay:      {analysis['group_delay']:.1f} samples")
    lines.append(" */")
    lines.append("")
    lines.append(f"#define {macro}_PHASES     {num_phases}U")
    lines.append(f"#define {macro}_TAPS       {phase_len}U")
    if mode == 'interpolation':
        lines.append(f"#define {macro}_HIST_DEPTH ({macro}_TAPS - 1U)")
    else:
        lines.append(f"#define {macro}_HIST_DEPTH {macro}_TAPS")
    lines.append(f"#define {macro}_SHIFT      {shift}U")
    lines.append("")
    lines.append(f"static const {dtype} {prefix.lower()}_{ratio.lower()}_coef")
    lines.append(f"    [{macro}_PHASES][{macro}_TAPS] = {{")

    for i in range(num_phases):
        vals = ', '.join(f'{x:>{w}d}' for x in poly_q[i])
        comma = ',' if i < num_phases - 1 else ''
        lines.append(f"    {{ {vals} }}{comma}")
    lines.append("};")
    lines.append("")

    target = 1 << shift
    lines.append("/* DC gain verification:")
    if mode in ('interpolation', 'rational'):
        for i in range(num_phases):
            s = int(np.sum(poly_q[i]))
            lines.append(f" *   Phase {i}: sum = {s:>8d}  (target: {target}, err: {s-target:+d})")
    else:
        total = int(np.sum(poly_q))
        lines.append(f" *   Total sum = {total:>8d}  (target: {target}, err: {total-target:+d})")
        for i in range(num_phases):
            s = int(np.sum(poly_q[i]))
            lines.append(f" *   Phase {i}: sum = {s:>8d}")
    lines.append(" */")
    return '\n'.join(lines)


# ============================================================
#  9. 预设方案
# ============================================================

PRESETS = {
    '1': {'name': '语音通信 (低开销)', 'pb_fraction': 0.80,
          'A_stop': 45, 'method': 'remez', 'stopband_weight': 5,
          'desc': '通带到 80%, 45dB 衰减, 适合语音'},
    '2': {'name': '通用音频 (推荐)',   'pb_fraction': 0.85,
          'A_stop': 55, 'method': 'remez', 'stopband_weight': 10,
          'desc': '通带到 85%, 55dB 衰减, 性价比最优'},
    '3': {'name': '高品质音乐',        'pb_fraction': 0.92,
          'A_stop': 70, 'method': 'remez', 'stopband_weight': 20,
          'desc': '通带到 92%, 70dB 衰减, 抽头数多'},
    '4': {'name': '自定义参数', 'pb_fraction': None,
          'A_stop': None, 'method': None, 'stopband_weight': None,
          'desc': '手动输入所有参数'},
}


# ============================================================
#  10. 交互主程序
# ============================================================

def main():
    print()
    print("=" * 62)
    print("     嵌入式音频重采样 FIR 系数设计器 (专业版 v2.0)")
    print("=" * 62)
    print()

    fs_in = int(input("  输入采样率 (Hz) [默认 16000]: ") or 16000)
    fs_out = int(input("  输出采样率 (Hz) [默认 48000]: ") or 48000)

    (L, M, fs_inter, f_pass, f_stop,
     f_pass_hz, f_stop_hz, useful_nyq, mode, num_phases) = \
        compute_specs(fs_in, fs_out, 0.9)

    mode_cn = {'interpolation': '插值', 'decimation': '抽取',
               'rational': '有理重采样'}[mode]

    print()
    print(f"  转换类型:   {mode_cn} ({mode})")
    print(f"  L = {L}, M = {M}, 中间采样率 = {fs_inter} Hz")
    print(f"  有效奈奎斯特: {useful_nyq:.0f} Hz")
    print(f"  多相相位数: {num_phases}")

    # 预设选择
    print()
    print("  ── 质量预设 ──────────────────────────────────")
    for k, v in PRESETS.items():
        print(f"    ({k}) {v['name']}: {v['desc']}")
    preset_choice = input("  选择预设 [默认 2]: ").strip() or '2'
    preset = PRESETS.get(preset_choice, PRESETS['2'])

    if preset_choice == '4':
        pb_fraction = float(input(
            f"  通带边缘比率 (0~1.0) [默认 0.85]: ") or 0.85)
        A_stop = float(input(
            f"  目标阻带衰减 (dB) [推荐 50-80, 默认 60]: ") or 60)
        print()
        print("  设计方法: (0)Remez (1)Kaiser (2)Hamming (3)Blackman")
        method_choice = input("  选择 [默认 0]: ").strip() or '0'
        methods = {'0': 'remez', '1': 'kaiser', '2': 'hamming', '3': 'blackman'}
        method = methods.get(method_choice, 'remez')
        if method == 'remez':
            sw = input("  阻带权重 [默认 10]: ").strip() or '10'
            stopband_weight = float(sw)
        else:
            stopband_weight = 10.0
    else:
        pb_fraction = preset['pb_fraction']
        A_stop = preset['A_stop']
        method = preset['method']
        stopband_weight = preset['stopband_weight']
        print(f"\n  已选: {preset['name']}")

    (L, M, fs_inter, f_pass, f_stop,
     f_pass_hz, f_stop_hz, useful_nyq, mode, num_phases) = \
        compute_specs(fs_in, fs_out, pb_fraction)

    print(f"\n  通带截止: {f_pass_hz:.0f} Hz, 阻带起始: {f_stop_hz:.0f} Hz")
    print(f"  过渡带:   {f_stop_hz - f_pass_hz:.0f} Hz")

    N_rec = estimate_taps(A_stop, f_pass, f_stop, num_phases)
    print(f"  推荐抽头: {N_rec} (每相位 {N_rec // num_phases} taps)")

    current_coeff = None
    current_analysis = None
    current_poly = None

    while True:
        print()
        taps_input = input(
            f"  输入抽头数 (回车={N_rec}, 'q'退出): ").strip()
        if taps_input.lower() == 'q':
            break
        if taps_input == '':
            num_taps = N_rec
        else:
            try:
                num_taps = int(taps_input)
            except ValueError:
                print("  无效输入")
                continue
            if num_taps % num_phases != 0:
                num_taps = int(np.ceil(num_taps / num_phases)) * num_phases
                print(f"  → 对齐到 {num_taps}")
            if num_taps % 2 == 0:
                num_taps += num_phases
                print(f"  → 调整为 {num_taps} (奇数)")

        print("  设计中...", end='', flush=True)
        try:
            coeff = design_filter(num_taps, f_pass, f_stop, method=method,
                                  A_stop=A_stop, stopband_weight=stopband_weight)
        except Exception as e:
            print(f" 失败: {e}")
            continue

        gain_factor = L if mode in ('interpolation', 'rational') else 1
        coeff_scaled = coeff * gain_factor
        analysis = analyze_filter(coeff_scaled, f_pass, f_stop,
                                  fs_inter, num_phases, mode)
        poly = polyphase_decompose(coeff_scaled, num_phases)

        current_coeff = coeff_scaled
        current_analysis = analysis
        current_poly = poly
        print(" 完成.")

        print()
        print(" >>>>>> 性能指标")
        print(f" 总抽头数:          {analysis['num_taps']:>6}")
        print(f" 每相位抽头:        {poly.shape[1]:>6}")
        print(f" 通带纹波:          {analysis['pb_ripple']:>8.4f} dB")
        print(f" 阻带最小衰减:      {analysis['sb_atten']:>8.2f} dB")
        if analysis['f_3db']:
            print(f" -3 dB 截止频率:    {analysis['f_3db']*fs_inter:>8.1f} Hz")
        print(f" 过渡带宽度:        {analysis['transition_bw']*fs_inter:>8.1f} Hz")
        print(f" 群延迟:            {analysis['group_delay']:>8.1f} samples")
        print(" 通带纹波分解:")
        for label, r in analysis['pb_detail'].items():
            print(f"    0~{label:<4}通带:    {r:>8.4f} dB")
        print(" 多相 DC 增益验证:")
        for i, g in enumerate(analysis['phase_sums']):
            print(f"    Phase {i}:         {g:>10.6f}")
        print(f" 总 DC 增益:        {analysis['total_dc']:>10.6f}")

        warnings = []
        if analysis['f_3db'] and analysis['f_3db'] * fs_inter < useful_nyq * 0.7:
            warnings.append(f"⚠ -3dB={analysis['f_3db']*fs_inter:.0f}Hz 过低! 应>{useful_nyq*0.7:.0f}Hz")
        if analysis['sb_atten'] < 40:
            warnings.append(f"⚠ 阻带衰减 {analysis['sb_atten']:.1f}dB 偏低, 建议>50dB")
        if analysis['pb_ripple'] > 0.5:
            warnings.append(f"⚠ 通带纹波 {analysis['pb_ripple']:.2f}dB 过大, 增加抽头或降低通带比率")
        if warnings:
            print()
            for w_msg in warnings:
                print(f"  {w_msg}")
        else:
            print("\n  ✓ 所有指标合格")

        print()
        plot_response(analysis, fs_inter,
                      f"{fs_in}→{fs_out} Hz, N={analysis['num_taps']}, {method}, {mode}")

        ans = input("\n  满意? (y/n, 默认 y): ").strip().lower()
        if ans in ('', 'y'):
            break

    if current_coeff is None:
        print("\n  未生成系数。")
        return

    print()
    print("  输出格式: (0)float (1)Q15 (2)Q31 [默认 1]: ", end='')
    fmt_choice = input().strip() or '1'

    if fmt_choice == '0':
        phase_len = current_poly.shape[1]
        lines = [f"/* {fs_in} -> {fs_out} Hz, L={L}, M={M}, "
                 f"phases={num_phases}, taps/phase={phase_len} */",
                 f"const float fir_poly_coeff[{num_phases}][{phase_len}] = {{"]
        for i in range(num_phases):
            vals = ', '.join(f'{x:.10f}f' for x in current_poly[i])
            comma = ',' if i < num_phases - 1 else ''
            lines.append(f"    {{ {vals} }}{comma}")
        lines.append("};")
        c_str = '\n'.join(lines)

    elif fmt_choice == '1':
        poly_q, shift, sums = scale_to_fixed(
            current_poly, mode, num_phases, bits=15)
        print(f"\n  Q15: >> {shift}, max={np.max(np.abs(poly_q))}")
        target = 1 << shift
        if mode in ('interpolation', 'rational'):
            for i, s in enumerate(sums):
                print(f"    Phase {i}: {s:>6d} (err: {s-target:+d})")
        else:
            total = int(np.sum(poly_q))
            print(f"    Total: {total:>6d} (err: {total-target:+d})")
        c_str = generate_c_code(poly_q, L, M, num_phases, fs_in, fs_out,
                                shift, mode, method, current_analysis)

    elif fmt_choice == '2':
        poly_q, shift, sums = scale_to_fixed(
            current_poly, mode, num_phases, bits=31)
        print(f"\n  Q31: >> {shift}")
        target = 1 << shift
        if mode in ('interpolation', 'rational'):
            for i, s in enumerate(sums):
                print(f"    Phase {i}: {s:>12d} (err: {s-target:+d})")
        else:
            total = int(np.sum(poly_q))
            print(f"    Total: {total:>12d} (err: {total-target:+d})")
        c_str = generate_c_code(poly_q, L, M, num_phases, fs_in, fs_out,
                                shift, mode, method, current_analysis)

    print()
    print("─" * 62)
    print(c_str)
    print("─" * 62)

    save = input("\n  保存到文件? (文件名或回车跳过): ").strip()
    if save:
        with open(save, 'w') as f:
            f.write(c_str)
        print(f"  已保存至 {save}")
    print("\n  完成!")


if __name__ == '__main__':
    main()

