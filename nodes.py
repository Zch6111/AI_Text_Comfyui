import os
import json
import base64
import urllib.request
import urllib.error
from PIL import Image
import io
import numpy as np
import torch

class GeminiImageToPrompt:
    CATEGORY = "flux/prompt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "main_image": ("IMAGE", {"label": "Main Subject Image"}),
                "background_image": ("IMAGE", {"label": "Background Scene Image"}),
                "model": (["gpt-4o", "gpt-4", "gpt-3.5-turbo"],)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate_prompt"
    OUTPUT_NODE = False
    DESCRIPTION = "Extract subject + background + style + language from 2 images via OpenAI vision model."

    def encode_image(self, image_tensor):
        if isinstance(image_tensor, torch.Tensor):
            image_tensor = image_tensor.cpu().numpy()

        if image_tensor.ndim == 4:
            image_tensor = image_tensor[0]

        if image_tensor.ndim == 3 and image_tensor.shape[0] in (1, 3):
            image_tensor = np.transpose(image_tensor, (1, 2, 0))

        image_array = (image_tensor * 255).clip(0, 255).astype(np.uint8)

        if image_array.ndim == 3 and image_array.shape[2] == 1:
            image_array = image_array[:, :, 0]

        img = Image.fromarray(image_array)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def call_openai_vision(self, api_key, image_base64, model, role):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Describe the {role} in this image for AI image generation. Include:\n- Description:\n- Style:\n- Language:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.5
        }

        try:
            request = urllib.request.Request(
                url="https://api.openai.com/v1/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(request) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result['choices'][0]['message']['content']
        except Exception as e:
            return f"[OpenAI Error] {str(e)}"

    def extract_info(self, raw):
        info = {"description": "", "style": "", "language": ""}
        for line in raw.strip().split("\n"):
            if line.lower().startswith("style:"):
                info["style"] = line.split(":", 1)[-1].strip()
            elif line.lower().startswith("language:"):
                info["language"] = line.split(":", 1)[-1].strip()
            elif line.lower().startswith("description:"):
                info["description"] = line.split(":", 1)[-1].strip()
            else:
                info["description"] += " " + line.strip()
        return info

    def generate_prompt(self, api_key, main_image, background_image, model):
        main_b64 = self.encode_image(main_image)
        bg_b64 = self.encode_image(background_image)

        main_raw = self.call_openai_vision(api_key, main_b64, model, "main subject")
        bg_raw = self.call_openai_vision(api_key, bg_b64, model, "background scene")

        main_info = self.extract_info(main_raw)
        bg_info = self.extract_info(bg_raw)

        prompt = (
            f"{main_info['description'].strip()} placed within {bg_info['description'].strip()}. "
            f"This image is rendered in {main_info['style'] or bg_info['style'] or 'cinematic'} style, "
            f"using {main_info['language'] or bg_info['language'] or 'natural'} language. "
            f"Focus on mood, lighting, and artistic detail."
        )

        return (prompt,)


class SmartAutoPromptNode:
    CATEGORY = "flux/prompt"

    def __init__(self):
        self.state = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": "sk-xxx"}),
                "model": (["gpt-4o", "gpt-4", "gpt-3.5-turbo"],),
                "num_prompts": ("INT", {"default": 5, "min": 1, "max": 100}),
                "prompt_input": ("STRING", {"multiline": True, "default": ""})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate_prompt"
    OUTPUT_NODE = False
    DESCRIPTION = "Rewrites a prompt to generate similar prompts with different compositions or angles."

    def generate_prompt(self, api_key, model, num_prompts, prompt_input):
        system = (
            "You are an expert prompt writer. Rewrite the given prompt into different versions while keeping the subject, character, object, and style the same."
            " Only vary composition, angle, camera view, or minor contextual differences. Output {num_prompts} line-separated versions."
        )

        user = f"Prompt to rewrite:\n{prompt_input}\n\nPlease generate {num_prompts} alternative prompts."

        payload = {
            "model": model,
            "temperature": 0.7,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        try:
            request = urllib.request.Request(
                "https://openai-api.codejoyai.com:8003/openai/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method='POST'
            )
            with urllib.request.urlopen(request) as response:
                result = json.loads(response.read().decode("utf-8"))
                lines = [line.strip() for line in result['choices'][0]['message']['content'].split("\n") if line.strip()]
                return ("\n".join(lines[:num_prompts]),)
        except Exception as e:
            return (f"[Rewrite Error] {str(e)}",)


NODE_CLASS_MAPPINGS = {
    "SmartAutoPromptNode": SmartAutoPromptNode,
    "GeminiImageToPrompt": GeminiImageToPrompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartAutoPromptNode": "Prompt Generator (Smart Input)",
    "GeminiImageToPrompt": "Gemini Image 2 Prompt"
}
