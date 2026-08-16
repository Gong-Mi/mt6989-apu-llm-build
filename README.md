# MT6989 APU LLM Build

This repository is the public build relay for MT6989 MediaTek APU/Neuron validation.

It deliberately does **not** contain NeuroPilot SDK archives, vendor `.so` files,
credentials, or proprietary model artifacts. GitHub Actions downloads official
resources during a run, verifies them, builds host/Android artifacts, and uploads
only reproducible build outputs.

## Workflow inputs

Run `MT6989 APU build` with:

- `sdk_url`: the final direct NeuroPilot SDK archive URL from the official public
  NeuroPilot resource page. The documentation page itself is not an archive.
- `sdk_sha256`: optional SHA-256 of the archive; required for a release-grade run.
- `model_url`: optional direct URL for a public model or precompiled MT6989 model.
- `model_sha256`: optional SHA-256 of the model.
- `mode`: `audit` (download/verify/inspect only), `build` (build host artifacts),
  or `package` (also split large outputs for artifact upload).

The first run should use `audit`. Do not put the SDK archive in a repository,
secret, environment variable, cache, or release. Put only the direct URL in the
workflow input or repository variable. If the URL requires a token/cookie, use
an Actions secret and never print it.

## Device-side acceptance

GitHub Actions has no MT6989 APU. Its artifacts must be downloaded to the phone
and tested there for:

- Neuron/APU compilation
- cold/warm TTFT
- prefill and decode token/s
- KV-cache growth
- delegated vs CPU-fallback nodes
- memory and sustained execution

The local independent measurement and current boundary are recorded in
`APU_CAPACITY_VALIDATION_2026-08-16.md` in the audit workspace.

## Large artifact handling

`package` mode creates a manifest and 512 MiB chunks under `out/chunks/` before
uploading. The chunks are never treated as a model format; they are only an
artifact transport fallback and must be reassembled and SHA-256 checked on the
phone.
