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
    def _forward_with_positions(m, input_ids):
        # Pre-compute RoPE tables once and pass them through, so the exported
        # graph contains plain mul/add (rotate_half) instead of aten::cos/sin.
        rotary = m.model.rotary_emb
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        cos, sin = rotary(hidden_states=input_ids, position_ids=position_ids)
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)
        out = m.model(
            input_ids=input_ids,
            position_embeddings=(cos, sin),
            use_cache=False,
        )
        return m.lm_head(out.last_hidden_state)

    class _LogitsOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids):
            return _forward_with_positions(self.m, input_ids)

    exported = torch.export.export(
        _LogitsOnly(model),
        args=(input_ids,),
        strict=False,
    )
    torch.export.save(exported, args.output)
    print("exported", args.output, Path(args.output).stat().st_size)


if __name__ == "__main__":
    main()
