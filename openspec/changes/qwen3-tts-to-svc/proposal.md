## Why

Qwen3 TTS 目前仅支持文本到语音合成，不具备歌唱/语音转换能力。将其改造为 SVC（Singing Voice Conversion）系统，增加 F0 音高控制，可实现音色迁移与音高控制的解耦——用户提供音色参考、源音频和音高参考，即可生成目标音色+目标音高的转换结果。这填补了当前架构在可控语音转换方面的空白。

## What Changes

- **新增 F0 提取模块**：集成 RMVPE/DIO 等 F0 提取算法，从音高参考音频中提取 F0 轮廓，从源音频中提取内容特征
- **修改 Talker 模型架构**：在 codec embedding 中新增 F0 条件注入通道（类似现有 speaker embedding 在 codec position 6 的注入方式），使模型能接受连续 F0 信号作为生成条件
- **新增 SVC 推理管线**：替换 TTS 的文本输入为音频输入，支持三路输入（音色参考、源音频、音高参考），同时支持流式和非流式两种模式
- **新增 SVC 训练管线**：基于 LoRA 微调策略，训练模型学习 F0 条件下的音色转换，最大化复用预训练 TTS 权重
- **新增端到端推理 API**：提供 `generate_svc()` 高层接口，支持流式输出音频包和非流式完整输出
- **保留 TTS 原有能力**：所有 SVC 扩展以 LoRA/adapter 形式附加，不破坏原 TTS 推理路径

## Capabilities

### New Capabilities
- `f0-processing`: F0 基频提取与量化编码——从音频中提取 F0 轮廓，量化为离散 token 或连续嵌入，对齐到 codec 帧率（12Hz），支持 pitch shift 变换
- `svc-inference`: SVC 推理管线——接受音色参考音频、源音频、音高参考音频三路输入，输出转换后的音频；支持流式（逐 token 生成并解码）和非流式（完整生成后一次解码）两种模式
- `svc-training`: SVC LoRA 微调训练——基于预训练 TTS talker 模型，通过 LoRA 适配层学习 F0 条件注入和音频到音频的映射，构建训练数据集与训练循环

### Modified Capabilities
（无现有 spec 需要修改——所有变更以新增能力形式实现，不修改原 TTS spec）

## Impact

- **核心模型文件**：`modeling_qwen3_tts.py` 需扩展 Talker 前向传播以接受 F0 条件；`configuration_qwen3_tts.py` 需新增 F0 相关配置项
- **推理层**：`qwen3_tts_model.py` 需新增 `generate_svc()` 方法；`qwen3_tts_tokenizer.py` 需扩展编码接口以提取内容特征
- **训练层**：`finetuning/` 下需新增 SVC 数据集类和 LoRA 微调脚本
- **新增依赖**：`peft`（LoRA）、`torchcrepe` 或 `rmvpe`（F0 提取）
- **硬件约束**：仅使用 PG199 32GB GPU (cuda:0)，训练时 CPU 限 96 核、RAM 限 32GB
- **API 兼容性**：原 TTS API 不变，SVC 功能以独立入口暴露
