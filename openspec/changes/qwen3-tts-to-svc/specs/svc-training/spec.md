## ADDED Requirements

### Requirement: SVC training dataset format
The system SHALL accept training data as a JSONL file where each line contains `{"source_audio": "<path>", "target_audio": "<path>"}`. For self-reconstruction training, source_audio and target_audio MAY be the same file. An optional `"pitch_audio": "<path>"` field allows specifying a separate pitch reference (defaults to target_audio if omitted).

#### Scenario: Load self-reconstruction dataset
- **WHEN** a JSONL file with entries `{"source_audio": "a.wav", "target_audio": "a.wav"}` is provided
- **THEN** the dataset loads successfully, using the same audio as source, target, and pitch reference

#### Scenario: Load parallel corpus dataset
- **WHEN** a JSONL file with entries `{"source_audio": "a_spk1.wav", "target_audio": "a_spk2.wav", "pitch_audio": "a_spk2.wav"}` is provided
- **THEN** the dataset loads source content from spk1, target and pitch from spk2

#### Scenario: Invalid audio path in dataset
- **WHEN** a JSONL entry references a non-existent audio file
- **THEN** the dataset skips the entry with a warning (does not crash)

### Requirement: SVC training data preprocessing
The system SHALL preprocess each training sample by: (1) extracting codec tokens from source audio via 12Hz tokenizer, (2) extracting F0 contour from pitch reference audio, (3) extracting speaker embedding from target audio, (4) extracting target codec tokens from target audio as labels. All features SHALL be aligned to the same frame count.

#### Scenario: Preprocess a training sample
- **WHEN** a training sample with source, target, and pitch audio is loaded
- **THEN** the system produces aligned tensors: source_codes (T, 16), f0_embed (T, 1024), spk_embed (1024,), target_codes (T, 16), all at 12Hz frame rate

#### Scenario: Length mismatch handling
- **WHEN** source audio and target audio have different lengths
- **THEN** the system truncates both to the shorter length after codec encoding, ensuring frame-level alignment

### Requirement: LoRA-based SVC finetuning
The system SHALL finetune the talker model using LoRA adapters applied to the self-attention and MLP layers. The F0 projector module SHALL be trained with full parameters. The speaker encoder and tokenizer SHALL remain frozen.

#### Scenario: Initialize LoRA training
- **WHEN** the SVC training script is launched with a pretrained base model path
- **THEN** LoRA adapters (rank=32, alpha=64) are applied to q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj of the talker, and LoRA (rank=16) to the sub-talker code predictor. Only LoRA parameters and F0 projector parameters are set as trainable

#### Scenario: Verify frozen modules
- **WHEN** training begins
- **THEN** speaker_encoder.parameters() all have requires_grad=False, tokenizer model parameters all have requires_grad=False, and only LoRA/F0 projector parameters have requires_grad=True

### Requirement: SVC training loss computation
The system SHALL compute a combined loss: main talker cross-entropy loss on codec_0 prediction + weighted sub-talker cross-entropy loss on codec_1-15 prediction. The loss weight for sub-talker SHALL be configurable (default: 0.3).

#### Scenario: Compute training loss
- **WHEN** a forward pass produces logits for codec_0 and codec_1-15 given source codes + F0 embedding + speaker embedding as input and target codes as labels
- **THEN** the total loss = CE_loss(codec_0_logits, target_codec_0) + 0.3 * CE_loss(codec_1-15_logits, target_codec_1-15)

### Requirement: SVC training forward pass
The system SHALL construct training input embeddings by: (1) summing source codec embeddings across all codebook positions, (2) adding F0 continuous embedding per frame, (3) adding speaker embedding (broadcast to all frames). The target is the next-frame codec token prediction.

#### Scenario: Construct training input
- **WHEN** source_codes (T, 16), f0_embed (T, 1024), spk_embed (1024,) are provided
- **THEN** input_embeds = sum(codec_embed[i](source_codes[:, i]) for i in range(16)) + f0_embed + spk_embed.unsqueeze(0).expand(T, -1), with shape (T, 1024)

### Requirement: SVC training hyperparameters
The system SHALL use the following default training hyperparameters: optimizer=AdamW, lr=2e-5, warmup_steps=100, bf16 mixed precision, gradient_accumulation_steps=4, max_steps=10000, save_checkpoint_every=1000 steps.

#### Scenario: Training with default hyperparameters
- **WHEN** the training script is launched without custom hyperparameter overrides
- **THEN** training uses AdamW optimizer, lr=2e-5, warmup_steps=100, bf16, grad_accum=4

#### Scenario: Custom hyperparameter override
- **WHEN** the training script is launched with --lr 1e-4 --max_steps 5000
- **THEN** the specified values override defaults while other parameters remain at default

### Requirement: SVC adapter checkpoint saving
The system SHALL save only the LoRA adapter weights and F0 projector weights as the SVC checkpoint (not the full model). The checkpoint SHALL be loadable by the inference pipeline for SVC generation.

#### Scenario: Save SVC checkpoint
- **WHEN** a checkpoint save is triggered during training
- **THEN** the system saves LoRA adapter state dict and F0 projector state dict to the output directory, along with training config

#### Scenario: Checkpoint size
- **WHEN** an SVC checkpoint is saved
- **THEN** the checkpoint size is significantly smaller than the full model (expected < 200MB for rank=32 LoRA on 1.7B model)

### Requirement: TensorBoard training logging
The system SHALL log training metrics to TensorBoard including: training loss, main talker loss, sub-talker loss, learning rate, and gradient norm. Logs SHALL be written every N steps (configurable, default: 10).

#### Scenario: TensorBoard logging during training
- **WHEN** training is in progress
- **THEN** TensorBoard logs are written to the output directory under a "runs/" subdirectory, viewable via `tensorboard --logdir <output_dir>/runs`

### Requirement: GPU memory management for training
The system SHALL operate within 32GB GPU memory on PG199 (cuda:0). The system SHALL use gradient checkpointing if needed to fit within memory constraints.

#### Scenario: Training on PG199 32GB
- **WHEN** SVC training is launched on a 32GB GPU with default settings
- **THEN** peak GPU memory usage stays below 30GB (with ~2GB headroom), using bf16 + LoRA + gradient checkpointing as needed
