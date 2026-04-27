# SVC 实施工作流规则

## 核心循环（严格遵循，不得跳步）
1. **工作** — 实现当前任务
2. **审查** — 检查代码正确性、与 spec 一致性、无遗漏
3. **验证** — 运行测试/检查确认功能正常
4. **循环** — 标记完成，进入下一任务

## 训练流程（严格顺序，不得跳步）
1. 确定 batch_size → GPU 利用率 ≥60%
2. 网格超参搜索（在目标 bs 下）→ 选最优
3. 用最优超参做过拟合验证（≤20 样本）→ loss 收敛 + 推理输出可听
4. **条件响应验证**（必做，用真实三路输入，不得用自身变换）
5. **训练/推理分布一致性验证**（必做）：训练时 condition 信号的来源/分布必须与推理一致
6. 过拟合+条件响应+分布一致验证通过后才允许全量训练
7. 全量训练后做完备推理验证（含歌声场景）
**任何一步不通过都不得进入下一步。**

## 条件响应验证（过拟合后必做，验证失败不允许全量）
SVC 模型必须**真正响应**条件信号，不得学捷径。

**真实 SVC 场景验证**（必做，不得用自身变换替代）：
- 必须用**三个不同音频**：source (A), timbre_ref (B), pitch_ref (C)
- F0 必须来自 **真实 C 音频提取**，不是 `pitch_shift(source_F0)`
- Speaker 必须来自 **真实 B 音频提取**，不是源同人
- 训练和推理必须使用**同分布**的条件信号（不得训练用变换，推理用真实）

**F0 控制验证**：
- 固定 source A + timbre_ref B，变换 pitch_ref C
- 量化：`mean(log2(f0_out_voiced/f0_pitch_ref_voiced))*12 ∈ [-1, +1]`（输出 F0 轮廓跟随 C）
- 额外：`pearson_corr(f0_out, f0_pitch_ref) > 0.8`（逐帧轮廓相关性）

**Speaker 控制验证**：
- 固定 source A + pitch_ref C，变换 timbre_ref B
- 量化：`cosine(spk_emb(output), spk_emb(B)) > 0.7`

**内容保持验证**：
- ASR 字错率 WER < 20%，或人工听辨内容一致

**歌声联合验证**：
- source = 歌声 audio, timbre_ref = 不同说话人, pitch_ref = 另一段歌声
- 同时满足上述三项合格标准

**错误指标禁用**（已犯的错）：
- ❌ mean F0（歌声时变轮廓）
- ❌ token accuracy（teacher-forced acc 与推理能力脱钩）
- ❌ `pitch_shift(source_F0)`（与真实 SVC 分布不一致）
- ❌ 听感"有变化"（需量化）

## 架构/方案变更规则
任何架构变更必须：
1. 提出方案前先向用户说明变更原因和预期效果
2. 用户确认后再开始
3. 不得"先试试看"再补沟通
4. 失败的方案必须保留到变更文档（design.md / 本规则），避免重复踩坑

## 自我审查规则（汇报前必做）
提交任何结果给用户前：
1. 用**用户的视角**检查：是否真解决了原始需求？
2. 用**量化指标**自验证：不只是 loss/acc，要测条件响应
3. 若指标未达标，**不允许声称"验证通过"**
4. 发现问题主动说，不等用户指出

## 强制规则
- 每次开始新任务前，重新阅读相关 spec 和 design 决策
- 每个任务完成后必须有可验证的产出（测试通过/代码可运行）
- 不可退让的需求：三路输入、流式+非流式、F0 控制、LoRA 微调
- 硬件约束：PG199 32GB (cuda:0) 训练，V100 16GB (cuda:1) 推理验证，CPU≤96核，RAM≤32GB
- 信息不足时自行获取（读源码、查文档），不猜测
- 所有步骤必须有可信依据

## 验收指标（全部必须通过）
1. **F0 控制**：同一源音频 + 不同 pitch_shift → 输出听感音高有明显变化
2. **跨 Speaker**：同一源音频 + 不同 timbre_ref → 输出听感音色有明显变化
3. **内容保持**：转换后内容（歌词/语句）可辨识，与源一致
4. **歌声 + 跨 Speaker + F0**：歌声输入 + 不同音色 + 不同音高 → 三者联合生效
5. **流式/非流式一致**：两种模式输出一致
6. **输出长度合理**：输出时长与源音频时长比 0.8x~1.2x

## 技术约束
- 仅支持 12Hz V2 tokenizer
- F0 提取用 FCPE (torchfcpe)
- F0 注入用连续嵌入加法（与 speaker/text embedding 同方式）
- 实际模型 hidden_size = 2048（非默认 1024）
- LoRA: talker rank=32/alpha=64, sub-talker rank=16
- 输出 24kHz float32 mono
- 最大输入音频 60s
- TensorBoard 日志
- 原 TTS 推理路径不受影响

## 超参搜索触发规则
**任何以下变更都必须重新搜索超参数，不得跳过：**
- batch_size 变更
- 模型结构变更（LoRA rank/alpha、F0 projector 等）
- 数据集变更（新增/删除数据、过滤条件变更）
- 训练序列结构变更（prefill/trailing/EOS 等）
- 优化器/scheduler 变更

**数据复用规则：**
- 预处理数据(.pt)必须复用，不得无故重新预处理
- 新增数据用增量预处理（preprocess_svc.py 已支持）
- 过滤数据集通过 manifest 操作，不重新预处理

## 训练参数（网格搜索+OLS确认 2026-04-20）
- GPU 0 PG199 32GB 训练, GPU 1 V100 16GB 推理（V100须用float16）
- batch_size=32, grad_accum=1, 数据按 p80 序列长度过滤（T<=90）
- **lr=5e-4, rank=32, alpha=64, sub_talker_rank=16, sub_weight=0.15**
- warmup=100, max_grad_norm=1.0（LoRA下100% clip是正常的，起正则化作用）
- 数据集: 75K 样本（p80过滤后，含歌声+语音+pitch aug）
- 预处理数据用 cached dataset + num_workers=4 + pin_memory
- JSONL 文件必须用 encoding='utf-8' 打开
- soundfile 在 Windows 上读中文路径可能失败，回退用 librosa
