import openai
import asyncio
import logging
from typing import Optional

class LLMClient:
    def __init__(self, api_key: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.logger = logging.getLogger("llm_client")
    
    async def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            raise