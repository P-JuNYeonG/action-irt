"""OpenRouter client wrapper for LLM-assisted preprocessing.

The API key is read from the OPENROUTER_API_KEY environment variable by
default. Do not hard-code API keys in this repository.
"""

from __future__ import annotations

import os
import re

from openai import OpenAI


class OpenRouterPipeline:
    """Small wrapper around the OpenRouter chat completion API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "anthropic/claude-sonnet-4.5",
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise ValueError("Set OPENROUTER_API_KEY or pass api_key explicitly.")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=resolved_key,
        )
        self.model = model

    @staticmethod
    def extract_markdown_table(markdown_text: str) -> str:
        table_pattern = r"\|[^|]*\|(?:\n\|[^|]*\|)*"
        tables = re.findall(table_pattern, markdown_text)
        if not tables:
            return markdown_text
        return max(tables, key=lambda table: table.count("\n")).replace("\\n", "\n")

    def run(
        self,
        system_prompt: str,
        input_data: str,
        temperature: float = 0.2,
        max_tokens: int = 20000,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"---\nThe input data follows the structure below.\n\n{input_data}",
                },
            ],
        )

        result = ""
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                result += content
        return result
