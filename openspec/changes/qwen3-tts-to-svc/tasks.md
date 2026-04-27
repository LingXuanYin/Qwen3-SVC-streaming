## 1. F0 Processing Module

- [x] 1.1 Create `qwen_tts/svc/f0_extractor.py` with FCPE-based F0 extraction class: load weights lazily to cuda:0, extract F0 at 100Hz from 16kHz resampled audio, return 1-D float tensor
- [x] 1.2 Implement F0 frame rate alignment function: downsample 100Hz F0 to 12Hz using segment-wise mean pooling over voiced frames, with exact frame count matching codec token length
- [x] 1.3 Implement pitch shift function: apply semitone offset to F0 contour (multiply voiced F0 by 2^(shift/12)), default shift=0
- [x] 1.4 Create `qwen_tts/svc/f0_projector.py` with F0-to-embedding projector: log(1+F0) transform → Linear(1, 1024), learned zero-pitch embedding for unvoiced frames
- [x] 1.5 Write unit tests for F0 extraction, alignment, pitch shift, and projection (test with synthetic and real audio)

## 2. Configuration Extensions

- [x] 2.1 Add `Qwen3TTSSVCConfig` dataclass to `configuration_qwen3_tts.py` with fields: f0_extractor_type, f0_hop_size, f0_sample_rate, lora_rank, lora_alpha, lora_target_modules, sub_talker_lora_rank, max_audio_duration_seconds
- [x] 2.2 Add SVC-related fields to `Qwen3TTSConfig`: svc_adapter_path (optional), svc_enabled flag

## 3. Model Architecture Extensions

- [x] 3.1 Create `qwen_tts/svc/svc_adapter.py` with LoRA application logic: apply LoRA (rank=32, alpha=64) to talker q/k/v/o/gate/up_proj, LoRA (rank=16) to sub-talker code_predictor, using peft library
- [ ] 3.2 Implement SVC input embedding construction in `qwen_tts/svc/svc_model.py`: sum source codec embeddings across 16 codebooks + F0 embedding + speaker embedding broadcast, producing (T, 1024) input  — **验收不通过，实际实现偏离 design，需重做**
- [ ] 3.3 Implement SVC prefill logic: build input_embeds from three-input pipeline (source codes, F0 embed, speaker embed) compatible with existing talker.forward() signature  — **验收不通过**

## 4. SVC Inference Pipeline

- [ ] 4.1 Add `generate_svc()` method to `Qwen3TTSModel` in `qwen3_tts_model.py`: accept timbre_ref, source_audio, pitch_ref as AudioLike inputs, optional pitch_shift and streaming parameters  — **验收不通过，输出不响应 F0/speaker**
- [ ] 4.2 Implement non-streaming SVC inference: extract speaker embedding → encode source to codes → extract F0 from pitch_ref → align F0 → build prefill embeds → autoregressive generate → decode all codes → return (wavs, sr)  — **验收不通过**
- [ ] 4.3 Implement streaming SVC inference: same prefill as non-streaming, but yield audio chunks every N codec frames (default 4) via generator  — **验收不通过**
- [ ] 4.4 Implement LoRA adapter loading: `from_pretrained()` extension to load SVC LoRA weights and F0 projector from svc_adapter_path on top of base model  — **验收不通过**
- [ ] 4.5 Add input validation: check audio duration ≤ 60s, validate audio readability, raise ValueError with clear messages  — **部分通过（长度检查），但上游功能失败**
- [ ] 4.6 Write integration tests for SVC inference: test three-input generation, streaming vs non-streaming consistency, pitch shift, error handling  — **测试存在但未覆盖条件响应验证**

## 5. SVC Training Pipeline

- [x] 5.1 Create `finetuning/svc_dataset.py` with `SVCDataset` class: load JSONL with source_audio/target_audio/pitch_audio fields, preprocess to aligned tensors (source_codes, f0_embed, spk_embed, target_codes)
- [x] 5.2 Implement training data preprocessing: encode audios via tokenizer, extract F0, extract speaker embedding, align frame counts by truncation to shorter length
- [ ] 5.3 Create `finetuning/sft_svc.py` training script: load base model → apply LoRA → initialize F0 projector → training loop with combined loss (main + 0.3 * sub-talker)  — **训练能跑但模型不学 F0 响应**
- [ ] 5.4 Implement SVC training forward pass: construct input_embeds from source codes + F0 + speaker, compute main talker CE loss on target codec_0, compute sub-talker CE loss on target codec_1-15  — **训练结构与 design 偏离，需重做**
- [x] 5.5 Add TensorBoard logging: log total_loss, main_loss, sub_talker_loss, lr, grad_norm every 10 steps to output_dir/runs/
- [x] 5.6 Implement checkpoint saving: save only LoRA state_dict + F0 projector state_dict + training config, verify checkpoint size < 200MB
- [x] 5.7 Add gradient checkpointing and memory optimization: ensure peak GPU usage < 30GB on PG199 32GB with bf16 + LoRA

## 6. End-to-End Validation

- [ ] 6.1 Create self-reconstruction test: train SVC on same-audio pairs for 100 steps, verify loss decreases, generate output, verify audio is valid wav at 24kHz  — **未真正验证音频可辨识**
- [ ] 6.2 Validate streaming output: compare streaming concatenated output with non-streaming output, verify they match within numerical tolerance (atol=1e-4)  — **未做数值对比**
- [ ] 6.3 Validate F0 control: generate with different pitch_shift values (-12, 0, +12), extract F0 from outputs, verify pitch ratios match expected semitone shifts  — **实测失败：输出 F0 不跟随输入，逐帧差 -6.25st when input +6st**
- [ ] 6.4 Validate timbre control: generate with two different timbre references, extract speaker embeddings from outputs, verify they are closer to their respective references than to each other  — **未做 cosine similarity 定量验证**
- [x] 6.5 Validate TTS compatibility: verify original TTS generate_voice_clone() still works correctly after SVC modules are loaded (no regression)
- [ ] 6.6 Memory profiling: run training and inference, log peak GPU memory, verify within 32GB constraint
