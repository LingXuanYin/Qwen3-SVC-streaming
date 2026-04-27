"""Bundle acceptance outputs into a listening pack.

For each of 15 combos, copy source/timbre_ref/pitch_ref/output into
output/seedvc_listening/{label}/ and write a README with per-combo metrics.
"""
import os, sys, glob, random, shutil, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    # Replay exact same combos
    ALTOS = sorted(glob.glob('L:/DATASET/GTSinger_repo/English/EN-Alto-1/*/all is found/Control_Group/*.wav'))
    TENORS = sorted(glob.glob('L:/DATASET/GTSinger_repo/English/EN-Tenor-1/*/Always Remember Us This Way/Control_Group/*.wav'))
    SPEECH_F = sorted(glob.glob('L:/DATASET/vc_training/train/wav/SSB0005/*.wav'))
    SPEECH_M = sorted(glob.glob('L:/DATASET/vc_training/train/wav/SSB0197/*.wav'))
    random.seed(42)
    combos = []
    for i in range(15):
        src = random.choice(ALTOS + TENORS)
        tim = random.choice(SPEECH_M + SPEECH_F)
        pit = random.choice(TENORS if src in ALTOS else ALTOS)
        combos.append((src, tim, pit, f'F3A_{i}'))

    # Per-combo metrics: (F0 from re-eval nearest-neighbor, Spk/Cnt from acceptance run log)
    # Status lists pulled from acceptance_v2 log and reeval_report.json.
    per_combo = {
        'F3A_0':  dict(F0p=0.769, diff=-0.19, cosT=0.967, cosS=0.919, cer=0.12, F0='F', Spk='P', Cnt='P'),
        'F3A_1':  dict(F0p=0.995, diff=+0.01, cosT=0.969, cosS=0.942, cer=0.19, F0='P', Spk='P', Cnt='P'),
        'F3A_2':  dict(F0p=0.992, diff=+0.01, cosT=0.979, cosS=0.923, cer=0.04, F0='P', Spk='P', Cnt='P'),
        'F3A_3':  dict(F0p=0.832, diff=-0.18, cosT=0.976, cosS=0.921, cer=0.10, F0='P', Spk='P', Cnt='P'),
        'F3A_4':  dict(F0p=0.857, diff=-0.22, cosT=0.978, cosS=0.960, cer=0.06, F0='P', Spk='P', Cnt='P'),
        'F3A_5':  dict(F0p=0.980, diff=-0.01, cosT=0.977, cosS=0.944, cer=0.14, F0='P', Spk='P', Cnt='P'),
        'F3A_6':  dict(F0p=0.904, diff=-0.02, cosT=0.977, cosS=0.916, cer=0.35, F0='P', Spk='P', Cnt='P'),
        'F3A_7':  dict(F0p=0.994, diff=-0.00, cosT=0.964, cosS=0.911, cer=0.07, F0='P', Spk='P', Cnt='P'),
        'F3A_8':  dict(F0p=0.919, diff=-0.09, cosT=0.961, cosS=0.931, cer=1.00, F0='P', Spk='P', Cnt='F'),
        'F3A_9':  dict(F0p=0.528, diff=-0.21, cosT=0.973, cosS=0.926, cer=0.13, F0='F', Spk='P', Cnt='P'),
        'F3A_10': dict(F0p=0.742, diff=-0.49, cosT=0.973, cosS=0.936, cer=0.01, F0='F', Spk='P', Cnt='P'),
        'F3A_11': dict(F0p=0.919, diff=-0.10, cosT=0.966, cosS=0.928, cer=0.06, F0='P', Spk='P', Cnt='P'),
        'F3A_12': dict(F0p=0.948, diff=-0.07, cosT=0.970, cosS=0.943, cer=0.17, F0='P', Spk='P', Cnt='P'),
        'F3A_13': dict(F0p=0.698, diff=-1.22, cosT=0.965, cosS=0.938, cer=0.05, F0='F', Spk='P', Cnt='P'),
        'F3A_14': dict(F0p=0.824, diff=-0.06, cosT=0.967, cosS=0.913, cer=0.33, F0='P', Spk='P', Cnt='P'),
    }

    dst_root = 'output/seedvc_listening'
    os.makedirs(dst_root, exist_ok=True)

    index_lines = ['# Seed-VC zero-shot FULL_3audio listening pack',
                   '',
                   f'15 combos: singing source + speech timbre + opposite-gender singing pitch. diffusion_steps=30, cfg_rate=0.7.',
                   '',
                   '| label | src (A) | timbre (B) | pitch (C) | F0 pearson | |diff|st | cos(out,B) | cos(out,A) | CER | joint |',
                   '|---|---|---|---|---|---|---|---|---|---|']

    for src, tim, pit, label in combos:
        d = os.path.join(dst_root, label)
        os.makedirs(d, exist_ok=True)
        out_wavs = glob.glob(f'output/seedvc_accept_v2/{label}/*.wav')
        out_path = out_wavs[0] if out_wavs else None

        # Copy with canonical names
        shutil.copy2(src, os.path.join(d, 'A_source.wav'))
        shutil.copy2(tim, os.path.join(d, 'B_timbre_ref.wav'))
        shutil.copy2(pit, os.path.join(d, 'C_pitch_ref.wav'))
        if out_path:
            shutil.copy2(out_path, os.path.join(d, 'OUT.wav'))

        m = per_combo[label]
        joint = 'P' if (m['F0'] == 'P' and m['Spk'] == 'P' and m['Cnt'] == 'P') else 'F'
        index_lines.append(
            f'| {label} | {os.path.basename(src)} | {os.path.basename(tim)} | {os.path.basename(pit)} | '
            f'{m["F0p"]:.3f} [{m["F0"]}] | {abs(m["diff"]):.2f} | {m["cosT"]:.3f} [{m["Spk"]}] | {m["cosS"]:.3f} | '
            f'{m["cer"]:.2f} [{m["Cnt"]}] | **{joint}** |')

    index_lines += [
        '',
        '## Aggregate',
        '- F0 pearson>0.8 & |diff|<1st: **11/15 (73%)**',
        '- Speaker cos>0.7 & >src:       **15/15 (100%)**',
        '- Content CER<50%:              **14/15 (93%)**',
        '- **Joint F0+Spk+Cnt:          10/15 (67%)**',
        '- Length 0.8–1.2×:              **15/15 (100%)**',
        '',
        '## How to listen',
        'For each F3A_i dir, play in order:',
        '1. `A_source.wav` — what the original singer sang',
        '2. `B_timbre_ref.wav` — the speech speaker whose voice we want',
        '3. `C_pitch_ref.wav` — the singer whose melody/pitch we want',
        '4. `OUT.wav` — the model output (should sound like speaker B singing A\'s lyrics with C\'s melody)',
        '',
        '## Per-criterion',
        '- **F0 P** = output pitch contour follows C',
        '- **Spk P** = output voice sounds like B (and NOT like A)',
        '- **Cnt P** = ASR can still recognize A\'s lyrics in the output',
        '- **Joint P** = all three above',
    ]

    with open(os.path.join(dst_root, 'README.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(index_lines))

    # Also save native baseline samples for comparison
    native_root = 'output/seedvc_listening_native'
    os.makedirs(native_root, exist_ok=True)
    native_wavs = glob.glob('output/seedvc_native/N_*/*.wav')
    for nw in native_wavs[:5]:
        label = os.path.basename(os.path.dirname(nw))
        d = os.path.join(native_root, label)
        os.makedirs(d, exist_ok=True)
        shutil.copy2(nw, os.path.join(d, 'OUT.wav'))

    print(f'Listening pack ready at: {dst_root}/')
    print(f'Index: {dst_root}/README.md')
    print(f'Native baseline samples: {native_root}/')


if __name__ == '__main__':
    main()
