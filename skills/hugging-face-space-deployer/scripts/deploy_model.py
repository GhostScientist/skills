#!/usr/bin/env python3
"""
Deploy a Hugging Face model to a Space with auto-generated UI.

This script automatically detects:
1. Whether the model is a LoRA adapter or full model
2. Whether it supports the Inference API
3. Chooses the appropriate deployment strategy

Deployment strategies:
- Models WITH Inference API: Uses InferenceClient (free, cpu-basic)
- Full models WITHOUT Inference API: Uses transformers + ZeroGPU (free with quota)
- LoRA adapters: Uses peft + transformers + ZeroGPU (free with quota)

Usage:
    python deploy_model.py meta-llama/Llama-3-8B-Instruct --type chat
    python deploy_model.py GhostScientist/my-finetuned-model --type chat
    python deploy_model.py GhostScientist/my-lora-adapter --type chat --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct
"""

import argparse
import json
from huggingface_hub import HfApi, create_repo, upload_file, model_info, hf_hub_download


# Known providers that typically have Inference API support
INFERENCE_API_PROVIDERS = {
    "meta-llama",
    "mistralai",
    "HuggingFaceH4",
    "google",
    "stabilityai",
    "openai",
    "microsoft",
    "facebook",
    "sentence-transformers",
    "Qwen",
}


def detect_model_type(model_id: str) -> dict:
    """
    Detect whether a model is a full model or LoRA adapter.

    Returns dict with:
    - is_adapter: bool
    - base_model: str or None (if adapter)
    - has_full_weights: bool
    """
    api = HfApi()
    result = {
        "is_adapter": False,
        "base_model": None,
        "has_full_weights": False,
    }

    try:
        # List files in the repo
        files = api.list_repo_files(model_id, repo_type="model")
        file_names = [f.split("/")[-1] for f in files]

        # Check for adapter files
        has_adapter_config = "adapter_config.json" in file_names
        has_adapter_model = any("adapter_model" in f for f in file_names)

        # Check for full model files
        has_full_weights = any(
            f in file_names for f in [
                "model.safetensors",
                "pytorch_model.bin",
                "model-00001-of-00001.safetensors",  # Single shard
            ]
        ) or any(
            "model-" in f and ".safetensors" in f for f in file_names  # Sharded
        )

        result["has_full_weights"] = has_full_weights

        # If it has adapter files but no full weights, it's a LoRA adapter
        if has_adapter_config and has_adapter_model and not has_full_weights:
            result["is_adapter"] = True

            # Try to get base model from adapter_config.json
            try:
                config_path = hf_hub_download(model_id, "adapter_config.json")
                with open(config_path) as f:
                    adapter_config = json.load(f)
                result["base_model"] = adapter_config.get("base_model_name_or_path")
            except Exception:
                pass

    except Exception as e:
        print(f"Warning: Could not fully analyze model: {e}")

    return result


def has_inference_api(model_id: str) -> bool:
    """
    Check if a model likely has Inference API support.

    This uses heuristics since the API doesn't directly expose this info:
    1. Check if the model is from a known provider
    2. User-uploaded models (personal namespaces) typically don't have it
    3. LoRA adapters never have direct Inference API
    """
    # First check if it's an adapter - adapters never have Inference API
    model_type = detect_model_type(model_id)
    if model_type["is_adapter"]:
        return False

    org = model_id.split("/")[0] if "/" in model_id else None

    if org in INFERENCE_API_PROVIDERS:
        return True

    # Try to check model info for inference status
    try:
        info = model_info(model_id)
        # Models with "inference" in pipeline_tag or with widget usually have API
        if info.pipeline_tag and info.pipeline_tag in [
            "text-generation", "text2text-generation", "conversational",
            "fill-mask", "text-classification", "image-classification",
            "text-to-image", "image-to-text"
        ]:
            # Popular models from big orgs usually have it
            if info.downloads and info.downloads > 10000:
                return True
    except Exception:
        pass

    return False


# ============================================================================
# TEMPLATES FOR INFERENCE API (models that support it)
# ============================================================================

CHAT_APP_INFERENCE = '''"""Chat interface for {model_id} using Inference API"""
import gradio as gr
from huggingface_hub import InferenceClient

MODEL_ID = "{model_id}"
client = InferenceClient(MODEL_ID)


def respond(message, history, system_message, max_tokens, temperature, top_p):
    messages = [{{"role": "system", "content": system_message}}]

    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({{"role": "user", "content": user_msg}})
        if assistant_msg:
            messages.append({{"role": "assistant", "content": assistant_msg}})

    messages.append({{"role": "user", "content": message}})

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
    title="{title}",
    description="Chat with {model_id}",
    additional_inputs=[
        gr.Textbox(value="You are a helpful assistant.", label="System message"),
        gr.Slider(minimum=1, maximum=4096, value=512, step=1, label="Max tokens"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p"),
    ],
    # IMPORTANT: Gradio 5.x requires nested lists for examples
    examples=[
        ["Hello! How are you?"],
        ["Write a Python function to sort a list"],
        ["Explain this concept simply"],
    ],
)

if __name__ == "__main__":
    demo.launch()
'''


# ============================================================================
# TEMPLATES FOR ZEROGPU (models without Inference API)
# ============================================================================

CHAT_APP_ZEROGPU = '''"""Chat interface for {model_id} using ZeroGPU (free!)"""
import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "{model_id}"

# Load at startup (on CPU)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
)


@spaces.GPU  # GPU allocated on-demand, released after function returns
def generate_response(message, history, system_message, max_tokens, temperature, top_p):
    """Generate response - GPU is allocated only during this function."""
    messages = [{{"role": "system", "content": system_message}}]

    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({{"role": "user", "content": user_msg}})
        if assistant_msg:
            messages.append({{"role": "assistant", "content": assistant_msg}})

    messages.append({{"role": "user", "content": message}})

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate (no streaming with ZeroGPU)
    outputs = model.generate(
        **inputs,
        max_new_tokens=int(max_tokens),
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    return response


demo = gr.ChatInterface(
    generate_response,
    title="{title}",
    description="Chat with {model_id} (powered by ZeroGPU - free!)",
    additional_inputs=[
        gr.Textbox(value="You are a helpful assistant.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max tokens"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p"),
    ],
    # IMPORTANT: Gradio 5.x requires nested lists for examples
    examples=[
        ["Hello! How are you?"],
        ["Write a Python function to sort a list"],
        ["Explain this concept simply"],
    ],
)

if __name__ == "__main__":
    demo.launch()
'''


# ============================================================================
# TEMPLATE FOR LORA ADAPTERS
# ============================================================================

CHAT_APP_LORA = '''"""Chat interface for {model_id} (LoRA adapter) using ZeroGPU"""
import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# LoRA adapter
ADAPTER_ID = "{model_id}"
# Base model (from adapter_config.json)
BASE_MODEL_ID = "{base_model}"

# Load tokenizer from adapter
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)

# Load base model and apply adapter
print(f"Loading base model: {{BASE_MODEL_ID}}")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
)

print(f"Applying adapter: {{ADAPTER_ID}}")
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)

# Merge for faster inference
print("Merging adapter weights...")
model = model.merge_and_unload()
print("Model ready!")


@spaces.GPU
def generate_response(message, history, system_message, max_tokens, temperature, top_p):
    """Generate response - GPU is allocated only during this function."""
    messages = [{{"role": "system", "content": system_message}}]

    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({{"role": "user", "content": user_msg}})
        if assistant_msg:
            messages.append({{"role": "assistant", "content": assistant_msg}})

    messages.append({{"role": "user", "content": message}})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=int(max_tokens),
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    return response


demo = gr.ChatInterface(
    generate_response,
    title="{title}",
    description="LoRA fine-tuned model powered by ZeroGPU (free!)",
    additional_inputs=[
        gr.Textbox(value="You are a helpful assistant.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max tokens"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p"),
    ],
    examples=[
        ["Hello! How are you?"],
        ["Write a Python function to sort a list"],
        ["Explain this concept simply"],
    ],
)

if __name__ == "__main__":
    demo.launch()
'''


# ============================================================================
# OTHER TEMPLATES (unchanged patterns)
# ============================================================================

IMAGE_CLASSIFICATION_APP = '''"""Image classification with {model_id}"""
import gradio as gr
from transformers import pipeline

MODEL_ID = "{model_id}"
classifier = pipeline("image-classification", model=MODEL_ID)


def classify(image):
    if image is None:
        return {{}}
    results = classifier(image)
    return {{r["label"]: r["score"] for r in results}}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="{title}",
    description="Upload an image to classify it using {model_id}",
    # IMPORTANT: Gradio 5.x requires nested lists for examples
    examples=[],
)

if __name__ == "__main__":
    demo.launch()
'''

TEXT_TO_IMAGE_APP = '''"""Text-to-image generation with {model_id}"""
import gradio as gr
from huggingface_hub import InferenceClient

MODEL_ID = "{model_id}"
client = InferenceClient()


def generate(prompt, negative_prompt, width, height, guidance_scale, num_steps):
    if not prompt:
        return None

    image = client.text_to_image(
        prompt,
        negative_prompt=negative_prompt or None,
        model=MODEL_ID,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
    )
    return image


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="A beautiful sunset over mountains..."),
        gr.Textbox(label="Negative Prompt", placeholder="blurry, low quality"),
        gr.Slider(512, 1024, value=1024, step=64, label="Width"),
        gr.Slider(512, 1024, value=1024, step=64, label="Height"),
        gr.Slider(1, 20, value=7.5, step=0.5, label="Guidance Scale"),
        gr.Slider(10, 50, value=30, step=1, label="Steps"),
    ],
    outputs=gr.Image(label="Generated Image"),
    title="{title}",
    description="Generate images using {model_id}",
)

if __name__ == "__main__":
    demo.launch()
'''

TEXT_GENERATION_APP = '''"""Text generation with {model_id}"""
import gradio as gr
from huggingface_hub import InferenceClient

MODEL_ID = "{model_id}"
client = InferenceClient(MODEL_ID)


def generate(prompt, max_tokens, temperature, top_p):
    if not prompt:
        return ""

    response = client.text_generation(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return response


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", lines=5, placeholder="Enter your prompt..."),
        gr.Slider(50, 1000, value=200, step=10, label="Max Tokens"),
        gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p"),
    ],
    outputs=gr.Textbox(label="Generated Text", lines=10),
    title="{title}",
    description="Generate text using {model_id}",
)

if __name__ == "__main__":
    demo.launch()
'''

EMBEDDING_APP = '''"""Text embeddings with {model_id}"""
import gradio as gr
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_ID = "{model_id}"
model = SentenceTransformer(MODEL_ID)


def get_embeddings(texts):
    if not texts.strip():
        return "Please enter some text."

    lines = [line.strip() for line in texts.split("\\n") if line.strip()]
    embeddings = model.encode(lines)

    output = []
    for i, (text, emb) in enumerate(zip(lines, embeddings)):
        output.append(f"Text {{i+1}}: {{text[:50]}}...")
        output.append(f"Embedding shape: {{emb.shape}}")
        output.append(f"First 5 values: {{emb[:5].tolist()}}")
        output.append("")

    if len(lines) > 1:
        output.append("Similarity Matrix:")
        sim_matrix = np.inner(embeddings, embeddings)
        for i in range(len(lines)):
            row = [f"{{sim_matrix[i][j]:.3f}}" for j in range(len(lines))]
            output.append(f"  Text {{i+1}}: {{' | '.join(row)}}")

    return "\\n".join(output)


demo = gr.Interface(
    fn=get_embeddings,
    inputs=gr.Textbox(label="Input Texts (one per line)", lines=5),
    outputs=gr.Textbox(label="Embeddings", lines=15),
    title="{title}",
    description="Generate embeddings using {model_id}. Enter multiple lines to see similarity scores.",
)

if __name__ == "__main__":
    demo.launch()
'''


# ============================================================================
# REQUIREMENTS
# ============================================================================

REQUIREMENTS_INFERENCE = """gradio>=5.0.0
huggingface_hub>=0.26.0
"""

REQUIREMENTS_ZEROGPU = """gradio>=5.0.0
torch
transformers
accelerate
spaces
"""

REQUIREMENTS_LORA = """gradio>=5.0.0
torch
transformers
accelerate
spaces
peft
"""

REQUIREMENTS_IMAGE_CLASS = """gradio>=5.0.0
transformers>=4.40.0
torch>=2.0.0
Pillow>=10.0.0
"""

REQUIREMENTS_EMBEDDING = """gradio>=5.0.0
sentence-transformers>=2.2.0
numpy>=1.24.0
"""


# ============================================================================
# README TEMPLATES
# ============================================================================

README_TEMPLATE_INFERENCE = """---
title: {title}
emoji: {emoji}
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: {short_description}
---

# {title}

{description}

## Model

This Space uses [{model_id}](https://huggingface.co/{model_id}).

## Usage

{usage}
"""

README_TEMPLATE_ZEROGPU = """---
title: {title}
emoji: {emoji}
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: {short_description}
suggested_hardware: zero-a10g
---

# {title}

{description}

## Model

This Space uses [{model_id}](https://huggingface.co/{model_id}).

## How It Works

This Space uses **ZeroGPU** - a free GPU allocation system:
- The app runs on CPU by default (free)
- When you send a message, a GPU is allocated on-demand
- After generation completes, the GPU is released
- You get a daily quota of free GPU time

## Usage

{usage}
"""


def deploy_model(
    model_id: str,
    model_type: str,
    space_name: str | None = None,
    hardware: str | None = None,
    private: bool = False,
    organization: str | None = None,
    force_zerogpu: bool = False,
    base_model: str | None = None,
) -> str:
    """Deploy a model to a new Space."""

    api = HfApi()
    user = api.whoami()
    username = organization or user["name"]

    # Generate space name from model if not provided
    if not space_name:
        space_name = model_id.split("/")[-1].lower().replace("_", "-") + "-demo"

    repo_id = f"{username}/{space_name}"

    # Detect model type (LoRA adapter vs full model)
    model_info_result = detect_model_type(model_id)
    is_lora_adapter = model_info_result["is_adapter"]
    detected_base_model = model_info_result.get("base_model")

    # Use provided base model or detected one
    if is_lora_adapter:
        base_model = base_model or detected_base_model
        if not base_model:
            raise ValueError(
                f"Model {model_id} appears to be a LoRA adapter but no base model was found.\n"
                f"Please provide --base-model argument with the base model ID.\n"
                f"Check adapter_config.json for 'base_model_name_or_path' field."
            )

    # Determine deployment strategy
    use_inference_api = has_inference_api(model_id) and not force_zerogpu and not is_lora_adapter

    if model_type == "chat":
        if is_lora_adapter:
            app_template = CHAT_APP_LORA
            requirements = REQUIREMENTS_LORA
            readme_template = README_TEMPLATE_ZEROGPU
            default_hardware = "zero-a10g"
            strategy = f"LoRA Adapter + ZeroGPU (base: {base_model})"
        elif use_inference_api:
            app_template = CHAT_APP_INFERENCE
            requirements = REQUIREMENTS_INFERENCE
            readme_template = README_TEMPLATE_INFERENCE
            default_hardware = "cpu-basic"
            strategy = "Inference API"
        else:
            app_template = CHAT_APP_ZEROGPU
            requirements = REQUIREMENTS_ZEROGPU
            readme_template = README_TEMPLATE_ZEROGPU
            default_hardware = "zero-a10g"
            strategy = "ZeroGPU"
    elif model_type == "image-classification":
        app_template = IMAGE_CLASSIFICATION_APP
        requirements = REQUIREMENTS_IMAGE_CLASS
        readme_template = README_TEMPLATE_INFERENCE
        default_hardware = "cpu-upgrade"
        strategy = "Local transformers"
    elif model_type == "text-to-image":
        app_template = TEXT_TO_IMAGE_APP
        requirements = REQUIREMENTS_INFERENCE
        readme_template = README_TEMPLATE_INFERENCE
        default_hardware = "cpu-basic"
        strategy = "Inference API"
    elif model_type == "text-generation":
        app_template = TEXT_GENERATION_APP
        requirements = REQUIREMENTS_INFERENCE
        readme_template = README_TEMPLATE_INFERENCE
        default_hardware = "cpu-basic"
        strategy = "Inference API"
    elif model_type == "embedding":
        app_template = EMBEDDING_APP
        requirements = REQUIREMENTS_EMBEDDING
        readme_template = README_TEMPLATE_INFERENCE
        default_hardware = "cpu-upgrade"
        strategy = "Local sentence-transformers"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    hardware = hardware or default_hardware

    print(f"Deploying {model_id}")
    print(f"  Type: {model_type}")
    print(f"  Strategy: {strategy}")
    print(f"  Space: {repo_id}")
    print(f"  Hardware: {hardware}")

    # Create Space
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=private,
        exist_ok=False,
    )
    print("✓ Created Space repository")

    # Generate title
    title = model_id.split("/")[-1].replace("-", " ").replace("_", " ").title()

    # Get app content
    if is_lora_adapter:
        app_content = app_template.format(
            model_id=model_id,
            title=title,
            base_model=base_model,
        )
    else:
        app_content = app_template.format(
            model_id=model_id,
            title=title,
        )

    # Upload app.py
    upload_file(
        path_or_fileobj=app_content.encode(),
        path_in_repo="app.py",
        repo_id=repo_id,
        repo_type="space",
    )
    print("✓ Uploaded app.py")

    # Upload requirements.txt
    upload_file(
        path_or_fileobj=requirements.encode(),
        path_in_repo="requirements.txt",
        repo_id=repo_id,
        repo_type="space",
    )
    print("✓ Uploaded requirements.txt")

    # Generate and upload README
    emoji_map = {
        "chat": "💬",
        "image-classification": "🖼️",
        "text-to-image": "🎨",
        "text-generation": "📝",
        "embedding": "🔢",
    }

    usage_map = {
        "chat": "Type a message to start chatting with the model.",
        "image-classification": "Upload an image to classify it.",
        "text-to-image": "Enter a prompt to generate an image.",
        "text-generation": "Enter a prompt to generate text.",
        "embedding": "Enter text (one item per line) to generate embeddings.",
    }

    readme = readme_template.format(
        title=title,
        emoji=emoji_map.get(model_type, "🤖"),
        short_description=f"{model_type.replace('-', ' ').title()} demo",
        description=f"Interactive demo of {model_id} using Gradio.",
        model_id=model_id,
        usage=usage_map.get(model_type, "Use the interface to interact with the model."),
    )

    upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
    )
    print("✓ Uploaded README.md")

    space_url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\n✅ Model deployed successfully!")
    print(f"   URL: {space_url}")
    print(f"   Strategy: {strategy}")
    if hardware == "zero-a10g":
        print("   Note: ZeroGPU provides free GPU access with daily quota")

    return space_url


def main():
    parser = argparse.ArgumentParser(
        description="Deploy a Hugging Face model to a Space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Types:
    chat               - Chat/instruct models (auto-detects strategy)
    text-generation    - Text completion models (uses Inference API)
    image-classification - Image classifiers (uses transformers pipeline)
    text-to-image      - Diffusion models (uses Inference API)
    embedding          - Sentence embedding models (uses sentence-transformers)

Deployment Strategy (auto-detected):
    - LoRA adapters: Uses peft + ZeroGPU (detects adapter_config.json)
    - Popular providers (meta-llama, mistralai): Uses Inference API
    - Other models: Uses ZeroGPU (free with daily quota)

Examples:
    # Popular model with Inference API
    python deploy_model.py meta-llama/Llama-3-8B-Instruct --type chat

    # Fine-tuned full model (auto-detects ZeroGPU)
    python deploy_model.py GhostScientist/my-finetuned-model --type chat

    # LoRA adapter (auto-detects base model from adapter_config.json)
    python deploy_model.py GhostScientist/my-lora-adapter --type chat

    # LoRA adapter with explicit base model
    python deploy_model.py GhostScientist/my-lora --type chat --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct
        """,
    )

    parser.add_argument("model_id", help="Model ID on Hugging Face Hub")
    parser.add_argument(
        "--type",
        dest="model_type",
        required=True,
        choices=["chat", "image-classification", "text-to-image", "text-generation", "embedding"],
        help="Type of model/interface",
    )
    parser.add_argument(
        "--name",
        dest="space_name",
        help="Custom Space name (default: derived from model)",
    )
    parser.add_argument(
        "--hardware",
        choices=[
            "cpu-basic",
            "cpu-upgrade",
            "zero-a10g",
            "t4-small",
            "t4-medium",
            "a10g-small",
            "a10g-large",
        ],
        help="Hardware tier (default: auto-selected based on strategy)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make Space private",
    )
    parser.add_argument(
        "--org",
        dest="organization",
        help="Organization to create Space under",
    )
    parser.add_argument(
        "--force-zerogpu",
        action="store_true",
        help="Force ZeroGPU even if model supports Inference API",
    )
    parser.add_argument(
        "--base-model",
        dest="base_model",
        help="Base model ID for LoRA adapters (auto-detected from adapter_config.json if not provided)",
    )

    args = parser.parse_args()

    try:
        deploy_model(
            model_id=args.model_id,
            model_type=args.model_type,
            space_name=args.space_name,
            hardware=args.hardware,
            private=args.private,
            organization=args.organization,
            force_zerogpu=args.force_zerogpu,
            base_model=args.base_model,
        )
    except Exception as e:
        print(f"Error deploying model: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
