# coding=utf-8
"""LoRA adapter application and management for SVC."""

import json
import os
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model, PeftModel

from ..core.models.configuration_qwen3_tts import Qwen3TTSSVCConfig
from .content_projector import ContentProjector
from .f0_projector import F0Projector


def apply_svc_lora(
    model,
    svc_config: Qwen3TTSSVCConfig,
) -> tuple:
    """Apply LoRA adapters to talker and sub-talker for SVC training.

    Args:
        model: Qwen3TTSForConditionalGeneration instance.
        svc_config: SVC configuration.

    Returns:
        (f0_projector, trainable_param_count)
    """
    talker = model.talker

    # Apply LoRA to main talker
    talker_lora_config = LoraConfig(
        r=svc_config.lora_rank,
        lora_alpha=svc_config.lora_alpha,
        target_modules=svc_config.lora_target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=None,
    )
    model.talker = get_peft_model(talker, talker_lora_config)

    # Apply LoRA to sub-talker code predictor
    code_predictor = model.talker.base_model.model.code_predictor
    sub_lora_config = LoraConfig(
        r=svc_config.sub_talker_lora_rank,
        lora_alpha=svc_config.sub_talker_lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=None,
    )
    model.talker.base_model.model.code_predictor = get_peft_model(
        code_predictor, sub_lora_config
    )

    # Create F0 projector and Content projector
    hidden_size = talker.config.hidden_size
    f0_projector = F0Projector(hidden_size)
    content_projector = ContentProjector(hidden_size)

    # Freeze everything except LoRA params, F0 projector, and content projector
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    for param in f0_projector.parameters():
        param.requires_grad = True
    for param in content_projector.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable += sum(p.numel() for p in f0_projector.parameters())
    trainable += sum(p.numel() for p in content_projector.parameters())

    return f0_projector, content_projector, trainable


def save_svc_checkpoint(
    model,
    f0_projector: F0Projector,
    svc_config: Qwen3TTSSVCConfig,
    output_dir: str,
    step: Optional[int] = None,
    content_projector: Optional[ContentProjector] = None,
):
    """Save SVC adapter checkpoint (LoRA weights + projectors).

    Args:
        model: The model with LoRA applied.
        f0_projector: The F0 projector module.
        svc_config: SVC configuration.
        output_dir: Directory to save checkpoint.
        step: Optional training step for naming.
        content_projector: Optional content projector module.
    """
    save_dir = os.path.join(output_dir, f"checkpoint-{step}" if step else "checkpoint")
    os.makedirs(save_dir, exist_ok=True)

    # Save LoRA adapters
    if hasattr(model.talker, 'save_pretrained'):
        model.talker.save_pretrained(os.path.join(save_dir, "talker_lora"))

    code_predictor = _get_code_predictor(model)
    if hasattr(code_predictor, 'save_pretrained'):
        code_predictor.save_pretrained(os.path.join(save_dir, "sub_talker_lora"))

    # Save projectors
    torch.save(f0_projector.state_dict(), os.path.join(save_dir, "f0_projector.pt"))
    if content_projector is not None:
        torch.save(content_projector.state_dict(), os.path.join(save_dir, "content_projector.pt"))

    # Save config
    with open(os.path.join(save_dir, "svc_config.json"), "w") as f:
        json.dump(svc_config.to_dict(), f, indent=2)


def load_svc_checkpoint(
    model,
    checkpoint_dir: str,
    device: str = "cuda:0",
) -> tuple:
    """Load SVC adapter checkpoint onto a base model.

    Args:
        model: Base Qwen3TTSForConditionalGeneration model.
        checkpoint_dir: Path to saved SVC checkpoint.
        device: Device to load onto.

    Returns:
        (model_with_lora, f0_projector, svc_config)
    """
    # Load config
    config_path = os.path.join(checkpoint_dir, "svc_config.json")
    with open(config_path) as f:
        svc_config = Qwen3TTSSVCConfig.from_dict(json.load(f))

    # Load talker LoRA
    talker_lora_path = os.path.join(checkpoint_dir, "talker_lora")
    if os.path.exists(talker_lora_path):
        model.talker = PeftModel.from_pretrained(
            model.talker, talker_lora_path
        )

    # Load sub-talker LoRA
    sub_lora_path = os.path.join(checkpoint_dir, "sub_talker_lora")
    code_predictor = _get_code_predictor(model)
    if os.path.exists(sub_lora_path):
        new_predictor = PeftModel.from_pretrained(code_predictor, sub_lora_path)
        _set_code_predictor(model, new_predictor)

    # Load projectors
    hidden_size = model.talker.config.hidden_size if not isinstance(model.talker, PeftModel) else model.talker.base_model.model.config.hidden_size
    model_dtype = next(model.parameters()).dtype

    f0_projector = F0Projector(hidden_size)
    f0_projector.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, "f0_projector.pt"), map_location=device, weights_only=True)
    )
    f0_projector.to(device=device, dtype=model_dtype)

    content_projector = None
    cp_path = os.path.join(checkpoint_dir, "content_projector.pt")
    if os.path.exists(cp_path):
        content_projector = ContentProjector(hidden_size)
        content_projector.load_state_dict(
            torch.load(cp_path, map_location=device, weights_only=True)
        )
        content_projector.to(device=device, dtype=model_dtype)

    return model, f0_projector, content_projector, svc_config


def _get_code_predictor(model):
    """Get code_predictor handling both PeftModel and raw model."""
    talker = model.talker
    if isinstance(talker, PeftModel):
        return talker.base_model.model.code_predictor
    return talker.code_predictor


def _set_code_predictor(model, predictor):
    """Set code_predictor handling both PeftModel and raw model."""
    talker = model.talker
    if isinstance(talker, PeftModel):
        talker.base_model.model.code_predictor = predictor
    else:
        talker.code_predictor = predictor
