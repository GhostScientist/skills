#!/usr/bin/env python3
"""
Deploy a Hugging Face model to a Space with auto-generated UI.

Usage:
    python deploy_model.py meta-llama/Llama-3-8B-Instruct --type chat
    python deploy_model.py google/vit-base-patch16-224 --type image-classification
    python deploy_model.py stabilityai/stable-diffusion-xl-base-1.0 --type text-to-image
"""

import argparse
import os
from huggingface_hub import HfApi, create_repo, upload_file, model_info


# App templates for different model types
CHAT_APP = '''"""Chat interface for {model_id}"""
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
)

if __name__ == "__main__":
    demo.launch()
'''

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
        output.append(f"Text {i+1}: {{text[:50]}}...")
        output.append(f"Embedding shape: {{emb.shape}}")
        output.append(f"First 5 values: {{emb[:5].tolist()}}")
        output.append("")

    # Compute similarity if multiple texts
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

APPS = {
    "chat": CHAT_APP,
    "image-classification": IMAGE_CLASSIFICATION_APP,
    "text-to-image": TEXT_TO_IMAGE_APP,
    "text-generation": TEXT_GENERATION_APP,
    "embedding": EMBEDDING_APP,
}

REQUIREMENTS = {
    "chat": "gradio>=4.0.0\nhuggingface_hub>=0.26.0\n",
    "image-classification": "gradio>=4.0.0\ntransformers>=4.40.0\ntorch>=2.0.0\nPillow>=10.0.0\n",
    "text-to-image": "gradio>=4.0.0\nhuggingface_hub>=0.26.0\n",
    "text-generation": "gradio>=4.0.0\nhuggingface_hub>=0.26.0\n",
    "embedding": "gradio>=4.0.0\nsentence-transformers>=2.2.0\nnumpy>=1.24.0\n",
}

HARDWARE_RECOMMENDATIONS = {
    "chat": "cpu-basic",  # Uses InferenceClient (serverless)
    "image-classification": "cpu-upgrade",  # Small models run on CPU
    "text-to-image": "cpu-basic",  # Uses InferenceClient (serverless)
    "text-generation": "cpu-basic",  # Uses InferenceClient (serverless)
    "embedding": "cpu-upgrade",  # Sentence transformers are small
}


README_TEMPLATE = """---
title: {title}
emoji: {emoji}
colorFrom: {color_from}
colorTo: {color_to}
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: {short_description}
---

# {title}

{description}

## Model

This Space uses [{model_id}](https://huggingface.co/{model_id}).

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
) -> str:
    """Deploy a model to a new Space."""

    api = HfApi()
    user = api.whoami()
    username = organization or user["name"]

    # Generate space name from model if not provided
    if not space_name:
        space_name = model_id.split("/")[-1].lower().replace("_", "-") + "-demo"

    repo_id = f"{username}/{space_name}"

    # Get hardware recommendation
    if not hardware:
        hardware = HARDWARE_RECOMMENDATIONS.get(model_type, "cpu-basic")

    print(f"Deploying {model_id}")
    print(f"  Type: {model_type}")
    print(f"  Space: {repo_id}")
    print(f"  Hardware: {hardware}")

    # Create Space
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        space_hardware=hardware if hardware != "cpu-basic" else None,
        private=private,
        exist_ok=False,
    )
    print("✓ Created Space repository")

    # Generate title
    title = model_id.split("/")[-1].replace("-", " ").replace("_", " ").title()

    # Get app template
    app_content = APPS[model_type].format(
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
    requirements = REQUIREMENTS[model_type]
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

    readme = README_TEMPLATE.format(
        title=title,
        emoji=emoji_map.get(model_type, "🤖"),
        color_from="blue",
        color_to="purple",
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

    return space_url


def main():
    parser = argparse.ArgumentParser(
        description="Deploy a Hugging Face model to a Space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Types:
    chat               - Chat/instruct models (uses InferenceClient)
    text-generation    - Text completion models (uses InferenceClient)
    image-classification - Image classifiers (uses transformers pipeline)
    text-to-image      - Diffusion models (uses InferenceClient)
    embedding          - Sentence embedding models (uses sentence-transformers)

Examples:
    python deploy_model.py meta-llama/Llama-3-8B-Instruct --type chat
    python deploy_model.py google/vit-base-patch16-224 --type image-classification
    python deploy_model.py stabilityai/stable-diffusion-xl-base-1.0 --type text-to-image
    python deploy_model.py sentence-transformers/all-MiniLM-L6-v2 --type embedding
        """,
    )

    parser.add_argument("model_id", help="Model ID on Hugging Face Hub")
    parser.add_argument(
        "--type",
        dest="model_type",
        required=True,
        choices=list(APPS.keys()),
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
            "t4-small",
            "t4-medium",
            "a10g-small",
            "a10g-large",
        ],
        help="Hardware tier (default: auto-selected based on type)",
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

    args = parser.parse_args()

    try:
        deploy_model(
            model_id=args.model_id,
            model_type=args.model_type,
            space_name=args.space_name,
            hardware=args.hardware,
            private=args.private,
            organization=args.organization,
        )
    except Exception as e:
        print(f"Error deploying model: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
