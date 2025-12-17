---
name: hugging-face-space-deployer
description: Create, configure, and deploy Hugging Face Spaces for showcasing ML models. Supports Gradio, Streamlit, and Docker SDKs with templates for common use cases like chat interfaces, image generation, and model comparisons.
---

# Hugging Face Space Deployer

A skill for AI engineers to create, configure, and deploy interactive ML demos on Hugging Face Spaces.

## Overview

Hugging Face Spaces provide free hosting for ML demos. This skill helps you:
- Create new Spaces with proper configuration
- Choose the right SDK (Gradio, Streamlit, Docker)
- Select appropriate hardware (CPU, GPU tiers)
- Deploy models with professional UIs
- Manage secrets and environment variables

## Dependencies

```
huggingface_hub>=0.26.0
gradio>=4.0.0
```

## Core Capabilities

### 1. Space Creation & Configuration
- Initialize Spaces with correct SDK and hardware
- Configure `README.md` metadata (YAML frontmatter)
- Set up secrets and environment variables
- Manage visibility (public/private)

### 2. SDK Support
- **Gradio**: Best for ML demos, auto-generates UI from functions
- **Streamlit**: Best for data apps and dashboards
- **Docker**: Full control, any framework

### 3. Hardware Options
| Hardware | Use Case | Cost |
|----------|----------|------|
| `cpu-basic` | Simple demos, text models | Free |
| `cpu-upgrade` | Faster CPU inference | ~$0.03/hr |
| `t4-small` | Small GPU models (<7B) | ~$0.60/hr |
| `t4-medium` | Medium GPU models | ~$0.90/hr |
| `a10g-small` | Large models (7B-13B) | ~$1.50/hr |
| `a10g-large` | Very large models (30B+) | ~$3.15/hr |
| `a100-large` | Largest models | ~$4.50/hr |

## Space README.md Format

Every Space requires a `README.md` with YAML frontmatter:

```yaml
---
title: My Awesome Demo
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: A demo of my cool model
---

# My Awesome Demo

Description of what this Space does...
```

### Required Fields
- `title`: Display name for the Space
- `emoji`: Single emoji for the Space card
- `colorFrom`, `colorTo`: Gradient colors (red, yellow, green, blue, indigo, purple, pink, gray)
- `sdk`: One of `gradio`, `streamlit`, `docker`, `static`
- `app_file`: Entry point (default: `app.py`)

### Optional Fields
- `sdk_version`: Pin specific SDK version
- `pinned`: Pin to profile (true/false)
- `license`: SPDX license identifier
- `short_description`: Brief tagline (max 60 chars)
- `suggested_hardware`: Default hardware tier
- `suggested_storage`: Persistent storage tier
- `hf_oauth`: Enable HF OAuth (true/false)
- `disable_embedding`: Prevent iframe embedding

## Gradio Templates

### Basic Chat Interface

```python
import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

def respond(message, history, system_message, max_tokens, temperature, top_p):
    messages = [{"role": "system", "content": system_message}]

    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": message})

    response = ""
    for token in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        delta = token.choices[0].delta.content or ""
        response += delta
        yield response

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a helpful assistant.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max tokens"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p"),
    ],
)

if __name__ == "__main__":
    demo.launch()
```

### Image Classification

```python
import gradio as gr
from transformers import pipeline

classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

def classify(image):
    results = classifier(image)
    return {r["label"]: r["score"] for r in results}

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="Image Classifier",
    description="Upload an image to classify it using ViT",
    examples=["example1.jpg", "example2.jpg"],
)

if __name__ == "__main__":
    demo.launch()
```

### Text-to-Image Generation

```python
import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient()

def generate(prompt, negative_prompt, width, height, guidance_scale, num_steps):
    image = client.text_to_image(
        prompt,
        negative_prompt=negative_prompt,
        model="stabilityai/stable-diffusion-xl-base-1.0",
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
    )
    return image

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="A photo of a cat..."),
        gr.Textbox(label="Negative Prompt", placeholder="blurry, low quality"),
        gr.Slider(512, 1024, value=1024, step=64, label="Width"),
        gr.Slider(512, 1024, value=1024, step=64, label="Height"),
        gr.Slider(1, 20, value=7.5, step=0.5, label="Guidance Scale"),
        gr.Slider(10, 50, value=30, step=1, label="Steps"),
    ],
    outputs=gr.Image(label="Generated Image"),
    title="SDXL Image Generator",
)

if __name__ == "__main__":
    demo.launch()
```

### Model Comparison

```python
import gradio as gr
from huggingface_hub import InferenceClient

models = {
    "Zephyr 7B": "HuggingFaceH4/zephyr-7b-beta",
    "Mistral 7B": "mistralai/Mistral-7B-Instruct-v0.2",
    "Llama 3 8B": "meta-llama/Meta-Llama-3-8B-Instruct",
}

def generate(prompt, model_name, max_tokens, temperature):
    client = InferenceClient(models[model_name])
    response = client.text_generation(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )
    return response

with gr.Blocks() as demo:
    gr.Markdown("# Model Comparison")

    with gr.Row():
        prompt = gr.Textbox(label="Prompt", lines=3)

    with gr.Row():
        with gr.Column():
            model1 = gr.Dropdown(list(models.keys()), value="Zephyr 7B", label="Model 1")
            output1 = gr.Textbox(label="Output 1", lines=10)
        with gr.Column():
            model2 = gr.Dropdown(list(models.keys()), value="Mistral 7B", label="Model 2")
            output2 = gr.Textbox(label="Output 2", lines=10)

    with gr.Row():
        max_tokens = gr.Slider(50, 500, value=200, label="Max Tokens")
        temperature = gr.Slider(0.1, 1.5, value=0.7, label="Temperature")

    btn = gr.Button("Generate", variant="primary")

    btn.click(generate, [prompt, model1, max_tokens, temperature], output1)
    btn.click(generate, [prompt, model2, max_tokens, temperature], output2)

if __name__ == "__main__":
    demo.launch()
```

## Streamlit Template

```python
import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="My ML App", page_icon="🤖")

st.title("🤖 My ML App")

@st.cache_resource
def get_client():
    return InferenceClient("HuggingFaceH4/zephyr-7b-beta")

client = get_client()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = client.chat_completion(
            messages=st.session_state.messages,
            max_tokens=500,
        )
        reply = response.choices[0].message.content
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
```

## Docker Template

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir gradio huggingface_hub transformers torch

COPY app.py .

EXPOSE 7860

CMD ["python", "app.py"]
```

### Docker README.md

```yaml
---
title: My Docker Space
emoji: 🐳
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 7860
---
```

## Command Reference

### Create a New Space

```python
from huggingface_hub import create_repo, upload_file

# Create Space repository
create_repo(
    repo_id="username/my-space",
    repo_type="space",
    space_sdk="gradio",  # or "streamlit", "docker"
    private=False,
)

# Upload files
upload_file(
    path_or_fileobj="app.py",
    path_in_repo="app.py",
    repo_id="username/my-space",
    repo_type="space",
)
```

### Configure Hardware

```python
from huggingface_hub import request_space_hardware

request_space_hardware(
    repo_id="username/my-space",
    hardware="t4-small",  # See hardware options above
)
```

### Set Secrets

```python
from huggingface_hub import add_space_secret

add_space_secret(
    repo_id="username/my-space",
    key="API_KEY",
    value="your-secret-value",
)
```

### Pause/Restart Space

```python
from huggingface_hub import pause_space, restart_space

# Pause to stop billing
pause_space("username/my-space")

# Restart when needed
restart_space("username/my-space")
```

### Duplicate a Space

```python
from huggingface_hub import duplicate_space

duplicate_space(
    from_id="original/space",
    to_id="username/my-copy",
    private=False,
)
```

## Workflow Examples

### Workflow 1: Quick Gradio Demo

1. Create `app.py` with Gradio interface
2. Create `README.md` with Space metadata
3. Create `requirements.txt` with dependencies
4. Push to Hugging Face:

```bash
huggingface-cli repo create my-demo --type space --sdk gradio
cd my-demo
# Add your files
git add .
git commit -m "Initial Space"
git push
```

### Workflow 2: Deploy Existing Model

1. Identify the model on Hugging Face Hub
2. Create Space with appropriate hardware
3. Use `InferenceClient` or load model directly
4. Build UI around model capabilities

### Workflow 3: Private Space with Secrets

1. Create private Space
2. Add API keys as secrets
3. Access secrets via environment variables:

```python
import os
api_key = os.environ.get("API_KEY")
```

## Best Practices

### Performance
- Use `@gr.cache` or `@st.cache_resource` for model loading
- Choose appropriate hardware for model size
- Use `InferenceClient` for serverless inference when possible
- Implement streaming for long-running generations

### User Experience
- Add clear titles and descriptions
- Include example inputs
- Show loading indicators for slow operations
- Handle errors gracefully with user-friendly messages

### Security
- Never hardcode API keys—use Space secrets
- Validate user inputs
- Set appropriate rate limits
- Use private Spaces for sensitive demos

### Cost Management
- Start with CPU, upgrade only if needed
- Use `pause_space()` when not in use
- Consider ZeroGPU for intermittent GPU needs
- Monitor usage in Space settings

## Troubleshooting

### Space Won't Build
- Check `requirements.txt` for typos
- Ensure compatible package versions
- Check build logs in Space settings

### Out of Memory
- Reduce model precision (use fp16 or int8)
- Use smaller batch sizes
- Upgrade to larger hardware tier
- Use `InferenceClient` instead of loading locally

### Slow Startup
- Use `@spaces.GPU` decorator for ZeroGPU
- Lazy-load models on first request
- Pre-download models in Dockerfile

### CORS/Embedding Issues
- Set `disable_embedding: false` in README
- Check for conflicting `app_port` settings
