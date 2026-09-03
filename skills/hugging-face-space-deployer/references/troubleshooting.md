# Troubleshooting Reference

Known failure modes when deploying Hugging Face Spaces, and their fixes.

## Build and runtime errors

### "No API found"
**Cause:** The Gradio app isn't exposing its API, usually because of a hardware mismatch.
**Fix:** Open Space Settings and set the runtime to ZeroGPU (or the appropriate GPU tier).
Deploying a GPU app without setting hardware is the most common cause.

### `OSError: does not appear to have a file named pytorch_model.bin, model.safetensors`
**Cause:** Trying to load a LoRA adapter as if it were a full model.
**Fix:** Check for `adapter_config.json`. If present, load through PEFT:

```python
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("base-model")
model = PeftModel.from_pretrained(base_model, "adapter-id")
```

Also confirm `peft` is listed in `requirements.txt` — adapter Spaces fail without it.

### `ImportError: cannot import name 'HfFolder'`
**Cause:** A `gradio` / `huggingface_hub` version mismatch. `HfFolder` was removed from newer
`huggingface_hub` releases, so older pinned Gradio versions break against it.
**Fix:** Use `gradio>=5.0.0` with `huggingface_hub>=0.26.0`. Avoid pinning Gradio to an exact
old patch version; the `4.44.0` pin is a known-bad combination.

### `ValueError: examples must be nested list`
**Cause:** Gradio 5.x changed the `examples` format.
**Fix:** Use nested lists, one inner list per example row:

```python
examples=[["Example 1"], ["Example 2"]]   # correct
examples=["Example 1", "Example 2"]       # raises ValueError
```

### Inference API errors on a custom model
**Cause:** The model isn't served on the serverless Inference API.
**Fix:** Either add `pipeline_tag` to the model README (see `references/model-analysis.md`),
or load the model directly with `transformers` on ZeroGPU instead of using `InferenceClient`.

### Space builds but the model never loads
**Cause:** Missing `peft` for adapters, or the wrong base model.
**Fix:** Re-read `adapter_config.json` and confirm `base_model_name_or_path` matches the
`BASE_MODEL_ID` in the app. A mismatched base model produces garbage output rather than a
clean error, so verify this even when the Space appears healthy.

## Post-deploy checklist

| Symptom | Cause | Fix |
|---|---|---|
| "No API found" | Hardware mismatch | Set runtime to ZeroGPU in Settings |
| Model not loading | LoRA vs full model confusion | Re-check for `adapter_config.json`, use the matching template |
| Inference API errors | Model not on serverless | Load directly with `transformers` |
| Slow first response | ZeroGPU cold start | Expected — GPU allocation takes ~60–120s |

## Gradio version notes

Templates in `assets/` pin `sdk_version` in their README frontmatter. That value ages; before
deploying, check the current Gradio release and update `sdk_version` to match the `gradio`
version in `requirements.txt`. A mismatch between the two is a frequent build failure.
