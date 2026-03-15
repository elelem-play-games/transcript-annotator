"""
Generic OpenAI-compatible LLM client.
Works with OpenAI, or any OpenAI-compatible endpoint (Azure, local Ollama, etc.).
Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL / OPENAI_MODEL) in your environment.
"""

import json
import os
from typing import Dict, List, Optional


class LLMClient:
    """
    Thin wrapper around the openai SDK.

    Environment variables
    ---------------------
    OPENAI_API_KEY   – required
    OPENAI_BASE_URL  – optional, override for Azure / local endpoints
    OPENAI_MODEL     – optional default model (default: gpt-4o-mini)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        from openai import OpenAI

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY or pass api_key=..."
            )

        kwargs: Dict = {"api_key": self.api_key}
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if resolved_base_url:
            kwargs["base_url"] = resolved_base_url

        self.client = OpenAI(**kwargs)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> str:
        """
        Send a chat request and return the assistant message content.

        Parameters
        ----------
        messages : list of {"role": ..., "content": ...} dicts
        model    : override the instance-level default model
        """
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def embed(self, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
        """
        Return embeddings for a list of texts.

        Parameters
        ----------
        texts : list of strings to embed
        model : embedding model name
        """
        response = self.client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]

    # ------------------------------------------------------------------
    # Higher-level helpers used by the pipeline
    # ------------------------------------------------------------------

    def analyze_section(
        self,
        section_title: str,
        text: str,
        max_tokens: int = 1000,
    ) -> Dict:
        """
        Analyse a markdown section and return structured metadata as a dict.
        Used by pipeline/chunk_documents.py.
        """
        if len(text) < 100:
            return {
                "topic": section_title,
                "summary": text[:100],
                "entities": [],
                "key_concepts": [],
                "should_split": False,
                "split_suggestion": None,
            }

        prompt = f"""Analyse this section from a technical document.

Section Title: {section_title}

Content:
{text}

Please provide:
1. **Topic**: A concise topic/theme (5-10 words)
2. **Summary**: One sentence summary of the main point
3. **Entities**: List of technical terms, systems, tools, metrics mentioned (max 10)
4. **Key Concepts**: Main concepts/ideas discussed (max 5)
5. **Should Split**: Should this section be split into smaller chunks? (true/false)
6. **Split Suggestion**: If yes, briefly explain where/how to split (or null if no split needed)

Return ONLY valid JSON in this exact format (no markdown, no extra text):
{{
  "topic": "...",
  "summary": "...",
  "entities": ["entity1", "entity2"],
  "key_concepts": ["concept1", "concept2"],
  "should_split": false,
  "split_suggestion": null
}}"""

        try:
            response_text = self.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a technical documentation analyst. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
            )

            # Strip markdown code fences if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    json_lines.append(line)
                response_text = "\n".join(json_lines).strip()

            result = json.loads(response_text)

            # Ensure all required fields exist
            defaults = {
                "topic": section_title,
                "summary": text[:100],
                "entities": [],
                "key_concepts": [],
                "should_split": False,
                "split_suggestion": None,
            }
            for key, default in defaults.items():
                result.setdefault(key, default)

            return result

        except json.JSONDecodeError as exc:
            print(f"  ⚠️  JSON parsing failed: {exc}")
            return {
                "topic": section_title,
                "summary": text[:100],
                "entities": [],
                "key_concepts": [],
                "should_split": False,
                "split_suggestion": None,
            }
        except Exception as exc:
            print(f"  ⚠️  LLM analysis failed: {exc}")
            return {
                "topic": section_title,
                "summary": text[:100],
                "entities": [],
                "key_concepts": [],
                "should_split": False,
                "split_suggestion": None,
            }
