#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import torch
if not hasattr(torch.nn, "Buffer"):
    torch.nn.Buffer = lambda data, persistent=True: torch.nn.Parameter(data, requires_grad=False)
_torch_is_autocast_enabled = torch.is_autocast_enabled
if not getattr(_torch_is_autocast_enabled, "_accepts_device_type", False):
    def _compat_is_autocast_enabled(device_type=None):
        return _torch_is_autocast_enabled()
    _compat_is_autocast_enabled._accepts_device_type = True
    torch.is_autocast_enabled = _compat_is_autocast_enabled
from transformers import AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seq-len", type=int, default=16)
    args = ap.parse_args()

    model_dir = Path(args.model)
    print("torch", torch.__version__)
    print("model", model_dir)
    print("config", json.loads((model_dir / "config.json").read_text()).get("architectures"))

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    input_ids = torch.ones((1, args.seq_len), dtype=torch.long)
    print("loaded_parameters", sum(p.numel() for p in model.parameters()))
    print("export_start")

    # Export the decoder path directly.  This avoids the Transformers forward
    # wrapper's dynamic mask/rope preparation and lets us pass fixed RoPE tables.
    rotary = model.model.rotary_emb
    position_ids = torch.arange(args.seq_len, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        rope_cos, rope_sin = rotary(input_ids, position_ids)
    rope_dtype = next(model.parameters()).dtype
    rope_cos = rope_cos.to(dtype=rope_dtype)
    rope_sin = rope_sin.to(dtype=rope_dtype)

    class _DirectLogits(torch.nn.Module):
        def __init__(self, m, pos_emb):
            super().__init__()
            self.m = m
            self.register_buffer("rope_cos", pos_emb[0])
            self.register_buffer("rope_sin", pos_emb[1])

        def forward(self, input_ids):
            hidden_states = self.m.model.embed_tokens(input_ids)
            position_embeddings = (self.rope_cos, self.rope_sin)
            for layer in self.m.model.layers:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=None,
                    position_ids=None,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                )
            hidden_states = self.m.model.norm(hidden_states)
            return self.m.lm_head(hidden_states)

    exported = torch.export.export(
        _DirectLogits(model, (rope_cos, rope_sin)),
        args=(input_ids,),
        strict=False,
    )
    torch.export.save(exported, args.output)
    print("exported", args.output, Path(args.output).stat().st_size)


if __name__ == "__main__":
    main()
