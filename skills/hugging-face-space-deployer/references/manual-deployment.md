# Manual Deployment Reference

Use this when `scripts/deploy_model.py` can't be used — no network access to run it, a model
type it doesn't handle, or the user wants to assemble the Space themselves.

A Space needs three files at minimum: `app.py`, `requirements.txt`, and `README.md` (whose YAML
frontmatter configures the Space).

## Choosing the template

| Situation | app.py | requirements.txt | README.md |
|---|---|---|---|
| Model **has** Inference API | `assets/gradio_chat.py` | `assets/requirements_inference_api.txt` | `assets/README_template.md` |
| Full model, **no** Inference API | `assets/gradio_zerogpu_chat.py` | `assets/requirements_zerogpu.txt` | `assets/README_zerogpu.md` |
| LoRA/PEFT adapter | `assets/gradio_lora_chat.py` | `assets/requirements_lora.txt` | `assets/README_zerogpu.md` |
| Text-to-image | `assets/gradio_image_gen.py` | `assets/requirements_inference_api.txt` | `assets/README_template.md` |
| User prefers Streamlit | `assets/streamlit_app.py` | `assets/requirements_inference_api.txt` | `assets/README_template.md` |

Copy the chosen files, rename them to `app.py` / `requirements.txt` / `README.md`, then edit
the placeholders.

## Editing the templates

**Inference API template** — set `MODEL_ID` to a model that actually supports the Inference API.
Pointing it at an unsupported model is the most common mistake.

**ZeroGPU full-model template** — set `MODEL_ID`. The model loads lazily inside the
`@spaces.GPU`-decorated function so the Space starts quickly; keep that structure.

**LoRA template** — set **both** `ADAPTER_ID` (the adapter repo) and `BASE_MODEL_ID` (read from
the adapter's `adapter_config.json` → `base_model_name_or_path`). The template merges the
adapter with `merge_and_unload()` for faster inference.

**README templates** use `{{PLACEHOLDER}}` markers. Replace every one — a leftover `{{TITLE}}`
renders literally on the Space page. Keep `sdk_version` consistent with the `gradio` version in
`requirements.txt`.

## Creating and uploading

```bash
# Create the Space
hf repo create my-space-name --repo-type space --space-sdk gradio

# Upload the assembled folder
hf upload username/space-name ./local-folder --repo-type space
```

`scripts/create_space.py` wraps the create step with correct README scaffolding and accepts
`--sdk`, `--hardware`, `--private`, `--description`, `--emoji`, and `--org`.

## After uploading

1. **Set the runtime hardware.** Go to
   `https://huggingface.co/spaces/USERNAME/SPACE_NAME/settings` → "Space Hardware" and select
   ZeroGPU (or the paid tier). This is required for any GPU template — `suggested_hardware` in
   the README is only a hint, not a setting.
2. **Watch the build logs** for import and dependency errors.
3. **Send one test request** to confirm the model actually loads. A Space can build cleanly and
   still fail at inference time.

If anything fails, read `references/troubleshooting.md`.
