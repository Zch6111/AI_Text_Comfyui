import os
import json
import urllib.request
import urllib.error

class PromptGeneratorCore:
    def __init__(self, api_key, model, system_prompt, num_prompts, subject, obj, lora_trigger, setting, interaction, style):
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.api_url = "https://openai-api.codejoyai.com:8003/openai/v1/chat/completions"
        self.temperature = 0.88

        self.user_prompt = (
            f"Generate {num_prompts} image generation prompts based on the following core elements:\n"
            f"1. Model: Flux (implying detailed, high-quality output desired)\n"
            f"2. Subject: {subject}\n"
            f"3. Object: {obj}\n"
            f"4. Lora Trigger: Must include '{lora_trigger}'\n"
            f"5. Setting: {setting}\n"
            f"6. Interaction: {interaction}\n"
            f"7. Style: {style}\n"
            "Provide the output should start with the LoRA trigger. Response should be in line separated plain text without any irrelevant details."
        )

    def generate(self):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt}
            ]
        }

        try:
            request = urllib.request.Request(
                self.api_url,
                data=json.dumps(payload).encode('utf-8'),
                headers=headers,
                method='POST'
            )
            with urllib.request.urlopen(request) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result['choices'][0]['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"


class AutoPromptNode:
    CATEGORY = "flux/prompt"

    def __init__(self):
        self.state = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": "sk-xxx"}),
                "model": (["gpt-4o", "gpt-4", "gpt-3.5-turbo"],),
                "num_prompts": ("INT", {"default": 25, "min": 1, "max": 100}),
                "subject": ("STRING", {"default": "A futuristic city at night"}),
                "obj": ("STRING", {"default": "A flying car"}),
                "lora_trigger": ("STRING", {"default": "cinematic lighting, concept art"}),
                "setting": ("STRING", {"default": "Urban skyline, neon-lit, rainy"}),
                "interaction": ("STRING", {"default": "The object is hovering near buildings, glowing"}),
                "style": ("STRING", {"default": "high detail, 4k, digital painting"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate_auto_prompt"
    OUTPUT_NODE = False
    DESCRIPTION = "Auto-rotating smart prompt generator. Each run outputs a new line with dynamic prompt structure."

    def generate_auto_prompt(self, api_key, model, num_prompts, subject, obj, lora_trigger, setting, interaction, style):
        key = f"{api_key}_{model}_{num_prompts}_{subject}_{obj}"
        if key not in self.state:
            self.state[key] = {"index": 0, "lines": []}

        if not self.state[key]["lines"]:
            # ⬇️ 自动构建 system_prompt 和 user_prompt
            system_prompt = (
                "You are a creative prompt engineer for Flux. Generate diverse prompts for an AI image model. "
                "Each prompt must include the LoRA keyword and combine all elements clearly and imaginatively. "
                "Output line-separated plain text only."
            )
            user_prompt = (
                f"Generate {num_prompts} image generation prompts using the following core elements:\n"
                f"1. Subject: {subject}\n"
                f"2. Object: {obj}\n"
                f"3. LoRA Trigger: '{lora_trigger}'\n"
                f"4. Setting: {setting}\n"
                f"5. Interaction: {interaction}\n"
                f"6. Style: {style}\n"
                f"All prompts should begin with the LoRA trigger. Output line-separated plain text only, no extra commentary."
            )

            # 调用 Chat API
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "temperature": 0.88,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }

            try:
                request = urllib.request.Request(
                    "https://openai-api.codejoyai.com:8003/openai/v1/chat/completions",
                    data=json.dumps(payload).encode('utf-8'),
                    headers=headers,
                    method='POST'
                )
                with urllib.request.urlopen(request) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    text = result['choices'][0]['message']['content']
                    self.state[key]["lines"] = [line.strip() for line in text.split("\\n") if line.strip()]
            except Exception as e:
                return (f"Error: {str(e)}",)

        # 自动轮询
        lines = self.state[key]["lines"]
        idx = self.state[key]["index"] % len(lines)
        prompt = lines[idx]
        self.state[key]["index"] += 1

        return (prompt,)



NODE_CLASS_MAPPINGS = {
    "AutoPromptGeneratorNode": AutoPromptNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoPromptGeneratorNode": "Prompt Generator (Auto Step)"
}
