
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

        except urllib.error.HTTPError as e:
            return f"HTTP Error: {e.code} - {e.read().decode('utf-8')}"
        except Exception as e:
            return f"Error: {str(e)}"


class Node:
    CATEGORY = "flux/prompt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": "sk-xxx"}),
                "model": (["gpt-4o", "gpt-4", "gpt-3.5-turbo"],),
                "system_prompt": ("STRING", {"multiline": True, "default":
                    "You are an expert prompt engineer for AI image generation models, specifically for the 'Flux' model. "
                    "Your task is to generate creative and diverse prompts for 'Flux' model. Each prompt must include the Lora trigger: "
                    "'brita water filter with blue lid, product photography'. The scene should feature a cute animal "
                    "in a living room, curiously observing this water filter. Ensure variety in animal type, "
                    "living room style, lighting, and the animal's specific curious action. "
                    "Output exactly Flux prompts directly and based on best practice. Format the output as line separated plain text without irrelevant details."}),
                "num_prompts": ("INT", {"default": 25, "min": 1, "max": 100}),
                "subject": ("STRING", {"multiline": False, "default": "A cute animal (e.g., kitten, puppy, bunny, hamster, small fox, baby owl)"}),
                "obj": ("STRING", {"multiline": False, "default": "A Brita water filter pitcher"}),
                "lora_trigger": ("STRING", {"multiline": False, "default": "brita water filter with blue lid, product photography"}),
                "setting": ("STRING", {"multiline": False, "default": "A living room (e.g., modern, cozy, minimalist, bohemian)"}),
                "interaction": ("STRING", {"multiline": False, "default": "The animal is curious about the water filter (e.g., sniffing, pawing, tilting head, peering into it)"}),
                "style": ("STRING", {"multiline": False, "default": "Photorealistic or beautifully illustrative, with good lighting"}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("prompts",)

    FUNCTION = "generate_prompts"
    OUTPUT_NODE = False
    DESCRIPTION = "Flexible AI image prompt generator with detailed configurable inputs, returns prompts as a list."

    def generate_prompts(self, api_key, model, system_prompt, num_prompts, subject, obj, lora_trigger, setting, interaction, style):
        generator = PromptGeneratorCore(api_key, model, system_prompt, num_prompts, subject, obj, lora_trigger, setting, interaction, style)
        output = generator.generate()
        prompt_list = [line.strip() for line in output.strip().split("\n") if line.strip()]
        return (prompt_list,)


NODE_CLASS_MAPPINGS = {
    "PromptGeneratorNode": Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptGeneratorNode": "Prompt Generator"
}
