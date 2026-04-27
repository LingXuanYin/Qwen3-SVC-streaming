"""Feasibility check B: does Qwen3-TTS voice_clone preserve content on singing source?

Pipeline: src_singing → Whisper ASR → text → voice_clone(text, ref=timbre_ref) → out.wav

If out.wav preserves the lyrics (content cos > 0.7 with src) AND sounds like timbre_ref,
then Qwen3-TTS has the codec-level capability; our job is to add F0 conditioning via LoRA.
If out.wav is garbage on singing text, Qwen3-TTS backbone itself is inadequate for this task.
"""
import os, sys, glob, numpy as np, torch, librosa, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts.svc.content_encoder import HubertContentEncoder


def main():
    device = 'cuda:0'
    m = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-Base', torch_dtype=torch.bfloat16, device_map=device)
    hubert = HubertContentEncoder(device=device, dtype=torch.float16)

    from faster_whisper import WhisperModel
    asr = WhisperModel('medium', device='cpu', compute_type='int8')

    # 3 English singing sources + 3 timbre refs
    srcs = sorted(glob.glob('L:/DATASET/GTSinger_repo/English/EN-Alto-1/*/all is found/Control_Group/*.wav'))[:3]
    tims = sorted(glob.glob('L:/DATASET/vc_training/train/wav/SSB0005/*.wav'))[:3]

    os.makedirs('output/voice_clone_baseline', exist_ok=True)
    for i, (src, tim) in enumerate(zip(srcs, tims)):
        src_a, src_sr = sf.read(src, dtype='float32')
        if src_a.ndim > 1: src_a = src_a.mean(-1)

        # ASR source
        if src_sr != 16000:
            wav16 = librosa.resample(src_a.astype(np.float32), orig_sr=src_sr, target_sr=16000)
        else:
            wav16 = src_a.astype(np.float32)
        segs, _ = asr.transcribe(wav16, beam_size=1, language='en')
        text = ''.join(s.text for s in segs).strip()
        print(f'\n[{i}] src={os.path.basename(src)}')
        print(f'    ASR text: {text}')

        if not text:
            print(f'    skip (empty text)')
            continue

        try:
            with torch.inference_mode():
                wavs, fs = m.generate_voice_clone(
                    text=text,
                    ref_audio=tim,
                    language='english',
                    x_vector_only_mode=True,
                    non_streaming_mode=True,
                )
            out = wavs[0]
            if out.ndim > 1: out = out[0]
            out_path = f'output/voice_clone_baseline/out_{i}.wav'
            sf.write(out_path, out.astype(np.float32), fs)

            # ASR output to check content
            if fs != 16000:
                out16 = librosa.resample(out.astype(np.float32), orig_sr=fs, target_sr=16000)
            else:
                out16 = out.astype(np.float32)
            segs_out, _ = asr.transcribe(out16, beam_size=1, language='en')
            text_out = ''.join(s.text for s in segs_out).strip()
            print(f'    out ASR:  {text_out}')

            # HuBERT content cosine src vs out (common T via target_frames)
            with torch.inference_mode():
                src_codes = m.model.speech_tokenizer.encode(src_a, sr=src_sr).audio_codes[0]
                T_src = src_codes.shape[0]
                out_codes = m.model.speech_tokenizer.encode(out, sr=fs).audio_codes[0]
                T_out = out_codes.shape[0]
            # Use the shorter T and interp both to same length
            T = min(T_src, T_out)
            c_src = hubert.encode(src_a, src_sr, target_frames=T)
            c_out = hubert.encode(out, fs, target_frames=T)
            cos = torch.nn.functional.cosine_similarity(c_src, c_out, dim=-1).mean().item()
            print(f'    content cos(src, vc_out) [T_src={T_src}, T_out={T_out}, cmp@T={T}]: {cos:.3f}')
            print(f'    out len: {len(out)/fs:.2f}s  (src {len(src_a)/src_sr:.2f}s)')
        except Exception as e:
            print(f'    FAIL: {e}')


if __name__ == '__main__':
    main()
