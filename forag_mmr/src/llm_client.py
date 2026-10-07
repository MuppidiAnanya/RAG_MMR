from __future__ import annotations

import os
import time


class LLMClient:
    def __init__(self, provider: str, model: str, temperature: float, max_tokens: int):
        self.provider, self.model = provider.lower(), model
        self.temperature, self.max_tokens = temperature, max_tokens

    def complete(self, prompt: str, retries: int = 3) -> str:
        if self.provider not in {"openai", "groq"}:
            raise ValueError("LLM_PROVIDER must be 'openai' or 'groq'.")
        api_key_name = "GROQ_API_KEY" if self.provider == "groq" else "OPENAI_API_KEY"
        if not os.getenv(api_key_name):
            raise RuntimeError(f"{api_key_name} is required for generation/evaluation.")
        from openai import OpenAI
        # Groq implements the OpenAI chat-completions interface.
        client = OpenAI(api_key=os.environ[api_key_name], base_url="https://api.groq.com/openai/v1" if self.provider == "groq" else None)
        error: Exception | None = None
        for attempt in range(retries):
            try:
                response = client.chat.completions.create(model=self.model,
                    messages=[{"role": "user", "content": prompt}], temperature=self.temperature,
                    max_tokens=self.max_tokens)
                return response.choices[0].message.content or ""
            except Exception as exc:
                error = exc
                if attempt + 1 < retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM request failed after {retries} attempts: {error}")
