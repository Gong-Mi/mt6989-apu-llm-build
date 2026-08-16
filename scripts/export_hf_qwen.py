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

    # Freeze RoPE tables as constants: the exported graph then contains plain
    # mul/add (rotate_half) instead of aten::cos/aten::sin/aten::pow/exp,
    # which the mtk pytorch importer does not support.
    rotary = model.model.rotary_emb
    position_ids = torch.arange(args.seq_len, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        rope_cos, rope_sin = rotary(input_ids, position_ids)
    rope_cos = rope_cos.to(torch.float32)
    rope_sin = rope_sin.to(torch.float32)

    def _frozen_rope_forward(*args, **kwargs):
        return rope_cos, rope_sin

    type(rotary).forward = _frozen_rope_forward

    class _LogitsOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids):
            out = self.m(input_ids=input_ids, use_cache=False)
            return out.logits

    exported = torch.export.export(
        _LogitsOnly(model),
        args=(input_ids,),
        strict=False,
    )
    torch.export.save(exported, args.output)
    print("exported", args.output, Path(args.output).stat().st_size)


if __name__ == "__main__":
    main()
