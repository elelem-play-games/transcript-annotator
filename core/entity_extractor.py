"""
Entity extractor for transcripts.
Extracts technical terms, systems, tools, and metrics using an LLM.
"""

import json
import re
from typing import List, Dict
import argparse

from llm.llm import LLMClient


class EntityExtractor:
    """Extract entities from transcript text using LLM."""

    def __init__(self, model: str = None):
        """
        Initialize entity extractor.

        Args:
            model: Model name override (uses OPENAI_MODEL env var or default if None)
        """
        print(f"🔍 Initializing Entity Extractor")

        self.llm = LLMClient(model=model)

        print(f"   Model: {self.llm.model}")
        print(f"  ✅ Extractor ready!\n")

    def extract(self, transcript: str, max_entities: int = 20) -> List[str]:
        """
        Extract entities from transcript.

        Args:
            transcript: Transcript text
            max_entities: Maximum number of entities to extract

        Returns:
            List of entity strings
        """
        print(f"🔍 Extracting entities from transcript...")
        print(f"   Transcript length: {len(transcript)} chars")

        prompt = self._build_extraction_prompt(transcript, max_entities)

        try:
            response_text = self.llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at identifying proper nouns and names in "
                            "technical transcripts. Focus on extracting names of systems, "
                            "tools, software, people, projects, and codenames. "
                            "Return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
            )

            entities = self._parse_entities(response_text)
            print(f"  ✅ Extracted {len(entities)} entities")
            return entities

        except Exception as e:
            print(f"  ⚠️  Entity extraction failed: {e}")
            return self._fallback_extraction(transcript)

    def extract_with_context(self, transcript: str, max_entities: int = 20) -> List[Dict]:
        """
        Extract entities with their context (sentence where they appear).

        Args:
            transcript: Transcript text
            max_entities: Maximum number of entities to extract

        Returns:
            List of dicts with 'entity', 'contexts', 'context_count'
        """
        print(f"🔍 Extracting entities with context...")

        entities = self.extract(transcript, max_entities)
        sentences = self._split_into_sentences(transcript)

        entities_with_context = []
        for entity in entities:
            contexts = [s.strip() for s in sentences if entity.lower() in s.lower()]
            if contexts:
                entities_with_context.append(
                    {
                        "entity": entity,
                        "contexts": contexts,
                        "context_count": len(contexts),
                    }
                )

        print(f"  ✅ Found context for {len(entities_with_context)} entities")
        return entities_with_context

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_extraction_prompt(self, transcript: str, max_entities: int) -> str:
        return f"""Extract ALL proper nouns and names from this technical transcript.

**Extract these types of names:**
- System names (e.g., Borgmon, Bigtable, Kubernetes, "board mon", "big table")
- Tool/software names (e.g., Blaze, Docker, Jenkins, "blase")
- Product names and services
- Project codenames
- People's names (e.g., Sarah, John)
- Technical acronyms (e.g., DNS, SLO, SLA)

**IMPORTANT:**
- Extract names even if they look misspelled (e.g., "board mon", "big table", "blase")
- Extract multi-word names (e.g., "on call", "error budget", "hermetic builds")
- Include names that appear multiple times
- Be aggressive - when in doubt, include it

**Do NOT extract:**
- Common verbs/adjectives (running, slow, better)
- Generic nouns (system, process, issue, team, cluster)
- Time references (yesterday, last night, 2 AM)

Transcript:
{transcript}

Return ONLY a JSON array of names (max {max_entities}):
["entity1", "entity2", "entity3"]

Remember: Extract ALL names, including potential misspellings!
"""

    def _parse_entities(self, response_text: str) -> List[str]:
        # Strip markdown code fences
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()

        try:
            entities = json.loads(response_text)
            if not isinstance(entities, list):
                return []
            return [e.strip() for e in entities if isinstance(e, str) and e.strip()]
        except json.JSONDecodeError:
            match = re.search(r"\[(.*?)\]", response_text, re.DOTALL)
            if match:
                return [e.strip() for e in re.findall(r'"([^"]+)"', match.group(1))]
            return []

    def _fallback_extraction(self, transcript: str) -> List[str]:
        print(f"  Using fallback extraction...")
        words = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b", transcript)
        technical = re.findall(r"\b[a-z]+[A-Z][a-zA-Z]*\b", transcript)
        return list(set(words + technical))[:20]

    def _split_into_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
