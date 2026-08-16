#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText


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

    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    input_ids = torch.ones((1, args.seq_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    print("loaded_parameters", sum(p.numel() for p in model.parameters()))
    print("export_start")
    exported = torch.export.export(
        model,
        args=(),
        kwargs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        },
        strict=False,
    )
    torch.export.save(exported, args.output)
    print("exported", args.output, Path(args.output).stat().st_size)


if __name__ == "__main__":
    main()
