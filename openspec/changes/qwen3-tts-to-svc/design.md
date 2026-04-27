## Context

Qwen3 TTS 是一个基于 Transformer 的多 codebook 语音合成系统，核心架构包括：

1. **Speech Tokenizer（12Hz Mimi Codec）**：将音频编码为 16 个量化码本的离散 token（shape: `[T, 16]`），12Hz 帧率
2. **Talker（LM）**：20 层 Transformer，接受文本嵌入 + codec 嵌入，自回归生成 codec_0，再由 sub-talker（5 层）生成 codec_1-15
3. **Speaker Encoder（ECAPA-TDNN）**：从 mel 谱提取 1024 维说话人嵌入，注入 codec position 6
4. **双轨架构**：main talker 生成 codec_0，sub-talker 补全剩余 codebook，支持 token 级流式

当前系统以文本为输入条件。SVC 改造的核心是将条件从「文本」切换为「源音频内容 + F0 音高轮廓 + 目标音色」，同时保留流式能力。

**硬件约束**：PG199 32GB (cuda:0) 唯一 GPU，CPU 96 核/RAM 32GB 上限。

## Goals / Non-Goals

**Goals:**
- 实现三路输入（音色参考、源音频、音高参考）到转换音频的完整 SVC 管线
- F0 提取与量化编码，对齐到 12Hz codec 帧率
- 基于 LoRA 的高效微调，最大化复用预训练 TTS 权重
- 同时支持流式（逐 token 解码）和非流式（完整生成后解码）推理
- 保持原 TTS 推理路径不受影响

**Non-Goals:**
- 不改造 25Hz V1 tokenizer，仅支持 12Hz V2
- 不实现实时音频流输入（输入仍为完整音频文件）
- 不实现多说话人批量推理
- 不修改 Mimi codec 编解码器本身
- 不支持跨语言歌唱转换（限同语言）

## Decisions

### 决策 1：F0 提取算法选择 — RMVPE

**选择**：RMVPE（Robust Model for Vocal Pitch Estimation）

**替代方案**：
- DIO/Harvest（WORLD vocoder）：CPU-only，速度慢，对气声/混声处理差
- CREPE/TorchCREPE：GPU 加速但模型较大（~20MB），需额外依赖
- ParselMouth(Praat)：经典但对歌唱音频 pitch 跟踪不稳定

**理由**：RMVPE 是 SVC 社区（RVC/So-VITS-SVC）验证最广泛的 F0 提取器，对歌唱音频的鲁棒性最好，支持 GPU 推理，模型轻量（~3.6MB），且已有成熟的 PyTorch 实现可直接集成。

### 决策 2：F0 条件注入方式 — 连续嵌入加法注入

**选择**：将 F0 值通过对数变换 + 线性投影为与 hidden_size 相同维度的连续嵌入，以加法方式注入到 talker 的 inputs_embeds 中（与 speaker embedding 和 text embedding 的注入方式一致）。

**替代方案**：
- F0 量化为离散 token 占用新 codebook 位置：需要修改 codebook 结构，引入量化误差，且离散化丢失 F0 的连续性
- 作为 cross-attention 条件注入：需大幅修改 Transformer 层结构，LoRA 难以覆盖
- 作为额外的 prefix token 序列：会增加序列长度，影响推理速度

**理由**：加法注入与现有架构完全兼容——speaker embedding 在 position 6 已经用加法注入，text embedding 也是加法注入到 codec embedding。F0 嵌入用同样方式逐帧加到对应 codec 帧位置。这使得 LoRA 只需适配 attention/FFN 层学习 F0 条件响应，不需要引入新的结构性组件。

### 决策 3：源音频内容表示 — Codec Token 去音高

**选择**：使用现有 12Hz tokenizer encoder 将源音频编码为 codec tokens 作为内容表示。源音频的 F0 信息由音高参考音频的 F0 替代，实现音高与内容的解耦。

**替代方案**：
- 使用 HuBERT/ContentVec 等预训练特征提取器：需额外模型（300MB+），特征维度不匹配需要额外适配层
- 使用 Whisper encoder 特征：模型已有但输出在 16kHz/25Hz，需重采样对齐

**理由**：Codec tokens 是 talker 的原生输入格式，直接复用无需特征适配。内容/音高/音色的三要素解耦通过以下方式实现：
- **内容**：源音频 codec tokens（携带语言/音素信息）
- **音高**：音高参考音频的 F0 轮廓（连续嵌入注入）
- **音色**：音色参考音频的 speaker embedding（codec position 6 注入）

### 决策 4：微调策略 — LoRA on Talker + F0 Projector 全参数

**选择**：
- Talker Transformer 层：LoRA（rank=32, alpha=64）应用于 q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj
- F0 Projector（新增模块）：全参数训练
- Speaker encoder：冻结
- Tokenizer：冻结
- Sub-talker code predictor：LoRA（rank=16）

**替代方案**：
- 全参数微调：1.7B 参数在 32GB GPU 上无法 fit（即使 bf16 也需 ~3.4GB 仅权重，加上梯度/优化器状态需 ~20GB+）
- 仅训练 F0 projector：容量不足以学习复杂的内容-音高-音色交互

**理由**：LoRA 显存友好，32GB GPU 可训练 1.7B 模型。LoRA 权重可按需加载/卸载，不影响原 TTS 推理。F0 projector 是全新模块必须全参训练。

### 决策 5：推理管线设计 — 复用 TTS generate() 框架

**选择**：SVC 推理复用 `Qwen3TTSForConditionalGeneration.generate()` 的自回归循环，仅替换输入条件构建逻辑：
- 原 TTS：text_embed + speaker_embed → prefill → autoregressive codec generation
- SVC：source_codec_embed + f0_embed + speaker_embed → prefill → autoregressive codec generation

**理由**：现有 generate() 已集成 KV cache、streaming、sub-talker 调用等复杂逻辑。重写成本高且容易引入 bug。通过在 prefill 阶段替换输入嵌入，可以最小改动实现 SVC 推理。

### 决策 6：流式支持 — Token 级流式输出

**选择**：
- **非流式输入**：三路音频完整输入，一次性编码
- **流式输出**：复用 talker 现有的 token-by-token 生成机制，每生成 N 个 codec frame 立即解码输出音频包
- **非流式输出**：完整生成所有 codec tokens 后一次解码

**理由**：输入音频需完整才能提取全局 F0 和 speaker embedding，无法真正流式输入。输出流式天然由 autoregressive generation 支持。

## 失败方案记录（实测不可行，后续不得重复尝试）

### 失败 1：Codec embedding sum 作为 content signal（原 design 决策 3）
- 方案：`sum(codec_0..15)` 作为 text track，F0 embed 相加
- 实测：
  - 训练/推理的 prefix embedding 存在 diff=0.5 不一致
  - 教师强制 loss → 0 但 AR 推理 codec_0 accuracy = 0%
  - 输出 F0 ~= source F0，完全不响应输入 F0
- 根因：codec embeddings 在 codec 空间，与模型期望的 text-projected 空间尺度/语义都不匹配

### 失败 2：ContentProjector（codec → text-projected）
- 方案：加可训练 MLP 投影 codec embeddings
- 实测：过拟合 loss 0.0015，推理 accuracy 0%，输出含混不清
- 根因：exposure bias + 投影无法解耦 codec 中的 speaker/pitch 信息

### 失败 3：ICL prefill 复用
- 方案：源 codec 作为 ICL context，F0 作为 trailing
- 实测：AR 生成在早期停止或 accuracy 0%
- 根因：ICL 设计用于文本→声音，不适合声音→声音

### 失败 4：非 AR codec mapper（单 codebook）
- 方案：bidirectional transformer 预测 target codec_0
- 实测：过拟合 100% accuracy，但换 F0/speaker 时解码音频听感完全一致
- 根因：只改 codec_0 不改 codec_1-15，后者携带 98.5% 声学信息

### 失败 5：非 AR codec mapper（16 codebook）
- 方案：每 codebook 独立 head，同时预测全 16 个
- 实测：训练时模型走捷径复制 source codec，忽略 F0/speaker 输入
- 根因：source codec 携带完整 pitch/speaker 信息，无信息瓶颈

### 失败 6：Info bottleneck + pitch-parallel（±3st）
- 方案：训练时对 source 加 mask+noise，用 pitch-parallel 数据
- 实测：token 级 F0 响应仅 2%，输出 F0 固定不变
- 根因：训练只有 2 档 shift，模型学成二值分类；bottleneck 强度不够

### 失败 7：HuBERT content + parallel data
- 方案：HuBERT 特征替代 codec 作为 content（理论上 pitch-invariant）
- 实测：逐帧 F0 跟随测试：input +6st → output -6.25st（完美抵消）
- 根因：HuBERT cosine similarity 仅 0.78 for ±5st shift，保留足够 pitch 残留让模型恢复

## 根本障碍（所有失败的共同根因）

Mimi codec decoder **没有 F0 物理通路**：
- codec tokens 以离散方式编码了 pitch 全部信息
- mapper 预测 codec 是**生成**任务（选 pitch 值）而非**控制**任务（跟随 F0 输入）
- 无论多好的 pitch-invariant content encoder，最终 codec 要重建音频必须包含 pitch → 模型倾向于从 content 恢复 pitch

业界工作的 SVC（RVC/so-vits-svc/DDSP-SVC）**不用 codec decoder**：
- 用 HiFi-GAN/BigVGAN 声码器从 mel+F0（sine-exc）生成波形
- F0 是**物理正弦激励**直接注入声码器，不经过离散 token
- 在 codec 架构下做 F0 控制在学界也无成熟方案

## 历史架构决策修正

**决策 1（F0 提取器）**：实际使用 FCPE 替代 RMVPE（pip 可安装，等效功能）

**决策 3（源内容表示）**：原决策"codec tokens"在实测中不可行。目前无可用替代方案未经用户批准。

**决策 4（微调策略）**：硬件恢复为 PG199 32GB（cuda:0），V100 16GB (cuda:1) 保留用于推理验证。

## Risks / Trade-offs

- **[F0 对齐精度]** F0 提取在 16kHz/hop_size=160 (100Hz)，需下采样到 12Hz codec 帧率 → **缓解**：使用分段均值池化，保留 F0 轮廓趋势
- **[内容泄漏]** Codec tokens 可能残留源音频音色信息 → **缓解**：LoRA 微调中模型会学习忽略 codec 中的音色成分，依赖 speaker embedding 控制音色
- **[显存压力]** 长音频（>30s）的 codec 序列可能超出 KV cache 容量 → **缓解**：限制最大输入时长 60s，或分段处理
- **[LoRA 能力瓶颈]** Rank=32 的 LoRA 可能不足以学习复杂的 SVC 映射 → **缓解**：可逐步提升 rank，监控验证损失
- **[训练数据需求]** SVC 需要平行语料（同内容不同音色）或大量单音色数据 → **缓解**：先用自我重建任务（输入=输出=同一音频）预训练 SVC 管线，再用少量平行数据微调
