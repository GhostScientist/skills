# Model Analysis Reference

How to determine a model's type, Inference API availability, and hardware needs before deploying.

## Detecting model type

Inspect the model's file list on the Hub. If the HF MCP tool is available:

```
hf-skills - Hub Repo Details (repo_ids: ["username/model"], repo_type: "model")
```

Otherwise use the CLI:

```bash
hf download username/model-name --local-dir /tmp/check --dry-run 2>&1 | grep -E '\.(safetensors|bin|json)'
```

| Files present | Model type | How to load |
|---|---|---|
| `model.safetensors` or `pytorch_model.bin` (or sharded) | Full model | `AutoModelForCausalLM.from_pretrained(...)` directly |
| `adapter_model.safetensors` + `adapter_config.json` | LoRA/PEFT adapter | Load base model first, then apply adapter with `peft` |
| Config files only, no weights | Broken/incomplete | Stop and ask the user to verify the upload |

For adapters, read `base_model_name_or_path` out of `adapter_config.json` to identify the base model. This value is required — an adapter cannot be deployed without it.

```python
# Full model
model = AutoModelForCausalLM.from_pretrained("username/model")

# LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained("base-model-id")
model = PeftModel.from_pretrained(base_model, "username/adapter")
model = model.merge_and_unload()  # merge for faster inference
```

## Detecting Inference API availability

Check the model page for an "Inference Providers" widget in the right-hand column.

**Likely available:**
- Inference widget visible on the model page
- Model from an established provider namespace (`meta-llama`, `mistralai`, `HuggingFaceH4`, `google`, `stabilityai`, `Qwen`)
- High download count with a standard architecture

**Likely unavailable:**
- Personal namespace (e.g. `GhostScientist/my-model`)
- Any LoRA/PEFT adapter — adapters never have direct Inference API
- Missing `pipeline_tag` in model metadata
- No inference widget on the model page

When available, `InferenceClient` is the simplest path and runs free on `cpu-basic`.

## Hardware selection

| Model size | Recommended hardware |
|---|---|
| < 3B parameters | ZeroGPU (free) or CPU |
| 3B – 7B parameters | ZeroGPU or T4 |
| > 7B parameters | A10G or A100 |

Full tier list, for cost tradeoffs:

| Hardware | Use case | Cost |
|---|---|---|
| `cpu-basic` | Simple demos, Inference API apps | Free |
| `cpu-upgrade` | Faster CPU inference | Paid, lowest tier |
| **`zero-a10g`** | **On-demand GPU — recommended default** | **Free (with daily quota)** |
| `t4-small` | Small GPU models (<7B) | Paid |
| `t4-medium` | Medium GPU models | Paid |
| `a10g-small` | Large models (7B–13B) | Paid |
| `a10g-large` | Very large models (30B+) | Paid |
| `a100-large` | Largest models | Paid, highest tier |

Hardware pricing changes over time. Check the current rates at
<https://huggingface.co/docs/hub/spaces-gpus> rather than quoting figures to the user.

**ZeroGPU behavior:** the Space idles on CPU and allocates a GPU only when a user triggers
inference (~60–120s). It must be enabled manually in Space Settings after deployment; setting
`suggested_hardware` in the README is only a hint.

## Enabling Inference API via pipeline_tag

If a model should have an inference widget but doesn't, its metadata may be incomplete:

```bash
hf download username/model-name README.md --local-dir /tmp/fix
```

Add to the README's YAML frontmatter, then re-upload:

```yaml
---
pipeline_tag: text-generation
tags:
  - conversational
---
```

```bash
hf upload username/model-name /tmp/fix/README.md README.md
```

Correct tags do not guarantee Inference API access — it also depends on Hugging Face's own
infrastructure decisions. If the widget still doesn't appear, fall back to ZeroGPU.

## When the model type is unclear

Do not guess. Ask the user directly:

> I'm analyzing your model to determine the best deployment strategy. I found:
> - [what you found about the files]
> - [what you found about Inference API]
>
> Is this model:
> 1. A full model you trained/uploaded?
> 2. A LoRA/PEFT adapter on top of another model?
> 3. Something else?
>
> Also, would you prefer:
> A. Free deployment with ZeroGPU (may have queue times)
> B. A paid dedicated GPU for faster response
