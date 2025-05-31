
import os
import json
import urllib.request
import urllib.error

class PromptGeneratorCore:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable.")
        self.api_url = "https://openai-api.codejoyai.com:8003/openai/v1/chat/completions"
        self.model = "chatgpt-4o-latest"
        self.temperature = 0.88

        self.system_prompt = "You are a helpful assistant that creates prompts for AI applications."
        self.user_prompt = "Generate 10 high-quality prompts for testing purposes."

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
        self.generator = PromptGeneratorCore()

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompts",)

    FUNCTION = "generate_prompts"
    OUTPUT_NODE = False
    DESCRIPTION = "Generate AI prompts using OpenAI Chat API."

    def generate_prompts(self):
        output = self.generator.generate()
        return (output,)


NODE_CLASS_MAPPINGS = {
    "PromptGeneratorNode": Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptGeneratorNode": "Prompt Generator"
}
