## ADDED Requirements

### Requirement: F0 extraction from audio
The system SHALL extract fundamental frequency (F0) contour from any input audio using RMVPE algorithm. The extraction SHALL operate at 16kHz sample rate with hop_size=160 (100Hz output rate). The system SHALL handle unvoiced segments by marking them with F0=0.

#### Scenario: Extract F0 from voiced audio
- **WHEN** a valid audio waveform (mono, any sample rate) is provided to the F0 extractor
- **THEN** the system resamples to 16kHz, runs RMVPE, and returns a 1-D float tensor of F0 values in Hz at 100Hz frame rate, with unvoiced frames set to 0.0

#### Scenario: Extract F0 from silent audio
- **WHEN** a silent or near-silent audio waveform is provided
- **THEN** the system returns an all-zero F0 tensor of appropriate length without errors

### Requirement: F0 frame rate alignment to codec rate
The system SHALL downsample extracted F0 contours from native extraction rate (100Hz) to 12Hz codec frame rate. The downsampling SHALL use segment-wise mean pooling over voiced frames (F0 > 0), preserving unvoiced markers.

#### Scenario: Downsample 100Hz F0 to 12Hz
- **WHEN** a 100Hz F0 contour of length T_100 is provided
- **THEN** the system outputs a 12Hz F0 contour of length T_12 = ceil(T_100 * 12 / 100), where each frame is the mean of corresponding voiced source frames (or 0 if all source frames are unvoiced)

#### Scenario: Alignment with codec tokens
- **WHEN** an audio is encoded by the 12Hz tokenizer producing T codec frames, and F0 is extracted from the same audio
- **THEN** the aligned F0 contour SHALL have exactly T frames

### Requirement: F0 to continuous embedding projection
The system SHALL convert F0 values to continuous embeddings compatible with the talker hidden_size (1024). The projection SHALL apply log-scale transformation (log(1 + F0)) followed by a learnable linear projection.

#### Scenario: Project F0 to embedding
- **WHEN** a 12Hz F0 contour of shape (T,) is provided
- **THEN** the system outputs an embedding tensor of shape (T, 1024) where unvoiced frames (F0=0) produce a learned zero-pitch embedding

### Requirement: F0 pitch shifting
The system SHALL support pitch shifting by a semitone offset applied to the F0 contour before embedding. This allows the user to transpose the pitch reference up or down.

#### Scenario: Shift pitch up by 12 semitones
- **WHEN** an F0 contour and pitch_shift=12 is provided
- **THEN** all voiced F0 values are multiplied by 2.0 (one octave up) before embedding, unvoiced frames remain 0

#### Scenario: No pitch shift (default)
- **WHEN** an F0 contour is provided without pitch_shift parameter (or pitch_shift=0)
- **THEN** F0 values are passed through unchanged

### Requirement: F0 extractor GPU acceleration
The system SHALL run RMVPE inference on GPU (cuda:0) when available. The RMVPE model weights SHALL be loaded lazily on first use and cached for subsequent calls.

#### Scenario: First-time F0 extraction on GPU
- **WHEN** F0 extraction is called for the first time with CUDA available
- **THEN** the RMVPE model is loaded to cuda:0, inference runs on GPU, and the model is cached in memory

#### Scenario: Subsequent F0 extraction
- **WHEN** F0 extraction is called again after initial load
- **THEN** the cached RMVPE model is reused without reloading
