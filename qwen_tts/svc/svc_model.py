# coding=utf-8
"""SVC prefill logic based on ICL (In-Context Learning) mechanism.

Instead of creating a new content signal pipeline, reuses the model's
pretrained ICL capability:
- Source audio codec tokens serve as ICL reference (ref_code)
- Target speaker embedding replaces the voice
- F0 embedding is added to the codec track as pitch conditioning

This matches how the model already knows to work: see reference audio,
then generate output that preserves content but changes voice.
"""

import torch
from peft import PeftModel


def build_svc_prefill(
    model,
    source_codes: torch.Tensor,
    f0_embedding: torch.Tensor,
    speaker_embedding: torch.Tensor,
    non_streaming_mode: bool = False,
    content_projector=None,  # kept for API compat, not used in ICL approach
) -> tuple:
    """Build SVC prefill using ICL mechanism.

    Structure (streaming mode):
        Prefill: [prefix(6)] [ref_codec_bos + ref_codec_sum (source audio ICL context)]
        Trailing: [f0_frame(0), f0_frame(1), ..., f0_frame(T-1), tts_eos]

    The model sees the source audio as ICL context, then generates output
    conditioned on F0 frames (delivered via trailing) and speaker embedding.
    """
    config = model.config
    talker = model.talker
    base_talker = talker.base_model.model if isinstance(talker, PeftModel) else talker
    device = next(base_talker.parameters()).device
    dtype = next(base_talker.parameters()).dtype

    T = source_codes.shape[0]
    source_codes = source_codes.to(device)

    # Special text embeddings
    tts_bos_embed, tts_eos_embed, tts_pad_embed = base_talker.text_projection(
        base_talker.get_text_embeddings()(
            torch.tensor(
                [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]],
                device=device, dtype=torch.long,
            )
        )
    ).chunk(3, dim=1)  # 3 * (1, 1, D)

    # === Prefix (6 tokens): same as TTS x_vector_only ===
    codec_prefix_ids = [
        config.talker_config.codec_nothink_id,
        config.talker_config.codec_think_bos_id,
        config.talker_config.codec_think_eos_id,
    ]
    codec_prefix_embed = base_talker.get_input_embeddings()(
        torch.tensor([codec_prefix_ids], device=device, dtype=torch.long)
    )
    spk_embed = speaker_embedding.view(1, 1, -1).to(device=device, dtype=dtype)
    codec_pad_bos = base_talker.get_input_embeddings()(
        torch.tensor([[config.talker_config.codec_pad_id, config.talker_config.codec_bos_id]],
                     device=device, dtype=torch.long)
    )
    prefix_codec = torch.cat([codec_prefix_embed, spk_embed, codec_pad_bos], dim=1)
    prefix_text = torch.cat([tts_pad_embed.expand(-1, 5, -1), tts_bos_embed], dim=1)
    prefix_combined = prefix_text + prefix_codec  # (1, 6, D)

    # === ICL context: source audio codec tokens ===
    # Sum all 16 codebook embeddings per frame (same as generate_icl_prompt)
    codec_embeds = []
    for i in range(source_codes.shape[1]):
        if i == 0:
            emb = base_talker.get_input_embeddings()(source_codes[:, i:i+1])
        else:
            emb = base_talker.code_predictor.get_input_embeddings()[i-1](source_codes[:, i:i+1])
            if isinstance(emb, tuple):
                emb = emb[0]
        codec_embeds.append(emb)
    # Sum across codebooks: (T, 16 embeddings) → (T, D)
    ref_codec_embed = torch.stack([e.squeeze(1) for e in codec_embeds], dim=0).sum(dim=0)  # (T, D)
    ref_codec_embed = ref_codec_embed.unsqueeze(0)  # (1, T, D)

    # Prepend codec_bos
    codec_bos_embed = base_talker.get_input_embeddings()(
        torch.tensor([[config.talker_config.codec_bos_id]], device=device, dtype=torch.long)
    )
    ref_with_bos = torch.cat([codec_bos_embed, ref_codec_embed], dim=1)  # (1, T+1, D)

    # Text track for ICL context: tts_pad for all ref frames
    ref_text_track = tts_pad_embed.expand(-1, T + 1, -1)  # (1, T+1, D)

    # Combined ICL context
    icl_context = ref_text_track + ref_with_bos  # (1, T+1, D)

    # === F0 as trailing signal ===
    # F0 embedding serves as the "text track" during generation
    # It tells the model what pitch to use at each output frame
    f0_embed = f0_embedding.unsqueeze(0).to(device=device, dtype=dtype)  # (1, T, D)

    # Project F0 through content_projector if available (maps to text-projected space)
    if content_projector is not None:
        f0_embed = content_projector(f0_embed)

    # === Assemble ===
    # Prefill: prefix + ICL context + first generation frame
    # The first generation frame starts with codec_bos + f0[0]
    gen_start = f0_embed[:, :1] + base_talker.get_input_embeddings()(
        torch.tensor([[config.talker_config.codec_bos_id]], device=device, dtype=torch.long)
    )

    input_embeds = torch.cat([prefix_combined, icl_context, gen_start], dim=1)

    # Trailing: remaining F0 frames + tts_eos
    if T > 1:
        trailing_text_hidden = torch.cat([f0_embed[:, 1:], tts_eos_embed], dim=1)
    else:
        trailing_text_hidden = tts_eos_embed

    attention_mask = torch.ones(1, input_embeds.shape[1], device=device, dtype=torch.long)
    return input_embeds, attention_mask, trailing_text_hidden, tts_pad_embed
