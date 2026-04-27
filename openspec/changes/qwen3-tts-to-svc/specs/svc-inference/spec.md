## ADDED Requirements

### Requirement: Three-input SVC inference interface
The system SHALL provide a `generate_svc()` method on `Qwen3TTSModel` that accepts three audio inputs: timbre reference audio (for target voice), source audio (content to convert), and pitch reference audio (for target pitch contour). All three inputs SHALL accept the same `AudioLike` types as the existing TTS API (file path, URL, base64, numpy array, (waveform, sr) tuple).

#### Scenario: Basic SVC generation with three audio files
- **WHEN** `generate_svc(timbre_ref="spk.wav", source_audio="source.wav", pitch_ref="pitch.wav")` is called
- **THEN** the system extracts speaker embedding from timbre_ref, codec tokens from source_audio, F0 contour from pitch_ref, and generates converted audio with the timbre of spk.wav and pitch of pitch.wav applied to the content of source.wav

#### Scenario: Pitch reference same as source
- **WHEN** `generate_svc(timbre_ref="spk.wav", source_audio="source.wav", pitch_ref="source.wav")` is called
- **THEN** the system preserves the original pitch of source.wav while applying the timbre of spk.wav

#### Scenario: Pitch reference same as timbre reference
- **WHEN** `generate_svc(timbre_ref="spk.wav", source_audio="source.wav", pitch_ref="spk.wav")` is called
- **THEN** the system applies both timbre and pitch characteristics of spk.wav to the content of source.wav

### Requirement: Non-streaming SVC output
The system SHALL support non-streaming output mode where the complete converted audio is returned after full generation. This is the default mode.

#### Scenario: Non-streaming generation
- **WHEN** `generate_svc(..., streaming=False)` is called (or streaming parameter omitted)
- **THEN** the system returns a tuple `(wavs: List[np.ndarray], sample_rate: int)` containing the complete converted audio waveform, consistent with the existing TTS API return format

### Requirement: Streaming SVC output
The system SHALL support streaming output mode where audio packets are yielded incrementally as codec tokens are generated.

#### Scenario: Streaming generation
- **WHEN** `generate_svc(..., streaming=True)` is called
- **THEN** the system yields audio chunks as a generator/iterator, each chunk being a numpy array of audio samples. The concatenation of all chunks SHALL equal the non-streaming output (within numerical tolerance)

#### Scenario: Streaming chunk size
- **WHEN** streaming mode is active
- **THEN** each yielded audio chunk corresponds to a configurable number of codec frames (default: 4 frames at 12Hz ≈ 333ms), decoded by the tokenizer

### Requirement: LoRA weight loading for SVC
The system SHALL load SVC-specific LoRA weights and F0 projector weights separately from the base TTS model. The system SHALL support on-the-fly switching between TTS mode (no LoRA) and SVC mode (with LoRA).

#### Scenario: Load SVC adapter on top of base model
- **WHEN** `Qwen3TTSModel.from_pretrained(base_path, svc_adapter_path="path/to/svc_lora")` is called
- **THEN** the base TTS model is loaded first, then SVC LoRA weights and F0 projector are loaded on top, enabling SVC inference

#### Scenario: SVC inference without adapter
- **WHEN** `generate_svc()` is called without SVC adapter loaded
- **THEN** the system raises a clear error message indicating that SVC adapter weights are required

### Requirement: Pitch shift parameter
The system SHALL accept an optional `pitch_shift` parameter (in semitones) on `generate_svc()` to transpose the pitch reference F0 contour before generation.

#### Scenario: Generate with pitch transposition
- **WHEN** `generate_svc(..., pitch_shift=5)` is called
- **THEN** the F0 contour from pitch_ref is shifted up by 5 semitones before being injected as a condition

### Requirement: Output audio format consistency
The system SHALL output audio at the same sample rate as the tokenizer (24kHz for 12Hz tokenizer). The output format SHALL be consistent with existing TTS outputs.

#### Scenario: Output sample rate
- **WHEN** SVC generation completes
- **THEN** the output audio is at 24000 Hz sample rate, float32, mono channel

### Requirement: Input audio length validation
The system SHALL validate that input audio lengths do not exceed the maximum supported duration (60 seconds). The system SHALL provide clear error messages for invalid inputs.

#### Scenario: Audio exceeding maximum length
- **WHEN** source audio longer than 60 seconds is provided
- **THEN** the system raises a ValueError with a message indicating the maximum supported duration

#### Scenario: Empty or corrupt audio
- **WHEN** an empty audio file or unreadable path is provided
- **THEN** the system raises a ValueError with a descriptive error message
