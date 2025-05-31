
import os
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv

# Load variables from .env if it exists
load_dotenv()

class BritaPromptGenerator:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable.")
        self.api_url = "https://openai-api.codejoyai.com:8003/openai/v1/chat/completions"
        self.model = "chatgpt-4o-latest"
        self.temperature = 0.88

        self.system_prompt = (
            "You are an expert prompt engineer for AI image generation models, specifically for the 'Flux' model. "
            "Your task is to generate creative and diverse prompts for 'Flux' model. Each prompt must include the Lora trigger: "
            "'brita water filter with blue lid, product photography'. The scene should feature a cute animal "
            "in a living room, curiously observing this water filter. Ensure variety in animal type, "
            "living room style, lighting, and the animal's specific curious action. "
            "Output exactly Flux prompts directly and based on best practice. Format the output as line separated plain text without irrelevant details."
        )

        self.user_prompt = (
            "Generate 25 image generation prompts based on the following core elements:\n"
            "1. Model: Flux (implying detailed, high-quality output desired)\n"
            "2. Subject: A cute animal (e.g., kitten, puppy, bunny, hamster, small fox, baby owl).\n"
            "3. Object: A Brita water filter pitcher.\n"
            "4. Lora Trigger: Must include 'brita water filter with blue lid, product photography'.\n"
            "5. Setting: A living room (e.g., modern, cozy, minimalist, bohemian).\n"
            "6. Interaction: The animal is curious about the water filter (e.g., sniffing, pawing, tilting head, peering into it).\n"
            "7. Style: Aim for photorealistic or beautifully illustrative, with good lighting.\n"
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

    def __init__(self):
        self.generator = BritaPromptGenerator()

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompts",)

    FUNCTION = "generate_prompts"
    OUTPUT_NODE = False
    DESCRIPTION = "Generate 25 Flux prompts with Brita water filter and cute animal interaction."

    def generate_prompts(self):
        output = self.generator.generate()
        return (output,)


NODE_CLASS_MAPPINGS = {
    "BritaPromptGeneratorNode": Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BritaPromptGeneratorNode": "Brita Prompt Generator"
}
