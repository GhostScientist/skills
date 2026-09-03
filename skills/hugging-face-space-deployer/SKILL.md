---
name: hugging-face-space-deployer
description: Use this skill when deploying a machine learning model to a Hugging Face Space, creating an interactive ML demo, or publishing a model with a web UI. Covers Gradio and Streamlit apps, chat and image-generation interfaces, LoRA/PEFT adapters, ZeroGPU and paid GPU hardware, and managing existing Spaces. Use it whenever the user mentions Hugging Face Spaces, wants to "put a model online", "make a demo for my model", "share my fine-tune", or deploy a model they just trained — even if they don't say "Space" or "Gradio" explicitly.
license: Apache-2.0
metadata:
  author: GhostScientist
  version: "1.1"
---

# Hugging Face Space Deployer

Create, configure, and deploy interactive ML demos on Hugging Face Spaces.

Deployment fails in predictable ways when the model's type is misidentified. Always analyze the
model before generating any code.

## Workflow

1. Analyze the model — full model, LoRA adapter, or Inference API supported.
2. Choose a deployment strategy from that analysis.
3. Ask the user if the model type or cost preference is unclear.
4. Deploy with `scripts/deploy_model.py`, or assemble the Space manually.
5. Set the runtime hardware in Space Settings.
6. Verify the Space builds and actually answers a request.

## Step 1: Analyze the model

Inspect the model's files on the Hub before anything else:

```bash
hf download username/model-name --local-dir /tmp/check --dry-run 2>&1 | grep -E '\.(safetensors|bin|json)'
```

```
Does it have adapter_config.json?
├── YES → LoRA adapter. Read base_model_name_or_path from that file.
│         Deploy with peft + ZeroGPU.
└── NO
    ├── Has model.safetensors / pytorch_model.bin?
    │   ├── Inference widget on the model page? → Inference API, cpu-basic
    │   └── No widget?                          → ZeroGPU, load with transformers
    └── Neither → incomplete upload. Ask the user.
```

**Read `references/model-analysis.md`** for the full file-to-type table, Inference API detection
signals, the hardware tier table, and how to fix a missing `pipeline_tag`.

Adapters never have direct Inference API support. If you find `adapter_config.json`, the
Inference API path is ruled out — do not attempt it.

## Step 2: Deploy

`scripts/deploy_model.py` performs the analysis in Step 1 automatically and picks the strategy,
hardware, and dependencies. Prefer it:

```bash
# Full model or Inference API model — strategy auto-detected
python scripts/deploy_model.py meta-llama/Llama-3-8B-Instruct --type chat

# LoRA adapter — base model read from adapter_config.json
python scripts/deploy_model.py username/my-lora-adapter --type chat

# Override the auto-detected base model
python scripts/deploy_model.py username/my-lora --type chat --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct
```

`--type` is required and accepts `chat`, `image-classification`, `text-to-image`,
`text-generation`, or `embedding`. Also available: `--name`, `--hardware`, `--private`, `--org`,
and `--force-zerogpu`.

Requires `huggingface_hub` and an authenticated `hf` CLI (`hf auth login`).

**To assemble a Space by hand instead** — no network access to run the script, an unsupported
model type, or the user asks to see the files — **read `references/manual-deployment.md`**. It
maps each situation to the right starting files in `assets/` and covers the placeholders to
replace.

## Step 3: Set hardware

**GPU templates do not work until hardware is set manually.** Go to
`https://huggingface.co/spaces/USERNAME/SPACE_NAME/settings` → "Space Hardware" and select
ZeroGPU (free, on-demand) or a paid tier. `suggested_hardware` in the README is only a hint.

This single step causes the most post-deploy failures, including the common "No API found" error.

## Step 4: Verify

Watch the build logs, then send one real request. A Space can build cleanly and still fail at
inference — a wrong LoRA base model produces garbage output rather than an error.

**If anything fails, read `references/troubleshooting.md`.**

## Managing existing Spaces

`scripts/manage_space.py` handles hardware changes, secrets, and lifecycle:

```bash
python scripts/manage_space.py status username/my-space
python scripts/manage_space.py hardware username/my-space --tier t4-small
python scripts/manage_space.py secret username/my-space --key API_KEY --value xxx
python scripts/manage_space.py pause username/my-space      # stops billing
python scripts/manage_space.py restart username/my-space
```

`scripts/create_space.py` creates an empty, correctly-scaffolded Space when you want to upload
files separately.

## Gotchas

- Use `gradio>=5.0.0` and `huggingface_hub>=0.26.0`. Do not pin Gradio to an old exact patch
  version — `gradio==4.44.0` raises `ImportError: cannot import name 'HfFolder'`.
- Gradio 5.x requires nested lists for examples: `[["ex1"], ["ex2"]]`, not `["ex1", "ex2"]`.
- LoRA Spaces must include `peft` in `requirements.txt`.
- Keep `sdk_version` in the README frontmatter consistent with the `gradio` version in
  `requirements.txt`.
- Hardware pricing changes; point users to <https://huggingface.co/docs/hub/spaces-gpus>
  rather than quoting rates.

## Bundled files

| Path | Contents |
|---|---|
| `scripts/deploy_model.py` | Auto-detecting end-to-end deployment (preferred path) |
| `scripts/create_space.py` | Create a scaffolded, empty Space |
| `scripts/manage_space.py` | Hardware, secrets, pause/restart for existing Spaces |
| `references/model-analysis.md` | Model type detection, Inference API signals, hardware tiers |
| `references/manual-deployment.md` | Hand-assembly walkthrough and template selection |
| `references/troubleshooting.md` | Build and runtime error fixes |
| `assets/` | Gradio/Streamlit app templates, requirements files, README templates |
