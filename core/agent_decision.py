"""
Multi-Signal Agent Decision — treats fuzzy, RAG, and IPA as equal sources of truth.
"""

import json
from typing import Dict, List, Optional

from llm.llm import LLMClient


class MultiSignalAgent:
    """Agent that considers fuzzy, RAG, and IPA as equal sources."""

    def __init__(self, model: str = None):
        """
        Initialize agent.

        Args:
            model: LLM model override (uses OPENAI_MODEL env var or default if None)
        """
        print(f"🤖 Initializing Multi-Signal Agent")
        self.llm = LLMClient(model=model)
        print(f"   Model: {self.llm.model}")
        print(f"  ✅ Agent ready!\n")

    def decide(
        self,
        original_entity: str,
        transcript_context: str,
        fuzzy_matches: List[Dict],
        rag_chunks: List[Dict],
        ipa_matches: List[Dict],
        rag_concepts: str = "",
    ) -> Dict:
        """
        Make a correction decision based on all three signals.

        Args:
            original_entity: Entity from transcript (e.g., "board mon")
            transcript_context: Context where entity appears
            fuzzy_matches: List of fuzzy matches with scores
            rag_chunks: List of RAG chunks with concepts and entities
            ipa_matches: List of IPA phonetic matches with distances
            rag_concepts: Extracted concepts from transcript

        Returns:
            Dict with keys: action, corrected_entity, confidence, reasoning, signal_agreement
        """
        print(f"\n🤖 Multi-Signal Agent Decision")
        print(f"   Original: '{original_entity}'")
        print(f"   Context: '{transcript_context[:60]}...'")

        if not fuzzy_matches and not rag_chunks and not ipa_matches:
            return {
                "action": "skip",
                "corrected_entity": None,
                "confidence": 0,
                "reasoning": "No matches found from any signal (fuzzy, RAG, IPA)",
                "signal_agreement": "none",
            }

        prompt = self._build_decision_prompt(
            original_entity=original_entity,
            transcript_context=transcript_context,
            fuzzy_matches=fuzzy_matches,
            rag_chunks=rag_chunks,
            ipa_matches=ipa_matches,
            rag_concepts=rag_concepts,
        )

        try:
            response_text = self.llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at correcting technical terminology in "
                            "transcripts. You analyse multiple signals (fuzzy matching, "
                            "contextual relevance, phonetic similarity) to make informed "
                            "decisions."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=800,
            )

            decision = self._parse_decision(response_text, original_entity)

            print(f"   Action: {decision['action']}")
            print(f"   Entity: {decision.get('corrected_entity', 'N/A')}")
            print(f"   Confidence: {decision['confidence']}")
            print(f"   Reasoning: {decision['reasoning'][:100]}...")

            return decision

        except Exception as e:
            print(f"  ⚠️  Agent decision failed: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
            return self._fallback_decision(fuzzy_matches, rag_chunks, ipa_matches)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_decision_prompt(
        self,
        original_entity: str,
        transcript_context: str,
        fuzzy_matches: List[Dict],
        rag_chunks: List[Dict],
        ipa_matches: List[Dict],
        rag_concepts: str,
    ) -> str:
        # Fuzzy section
        if fuzzy_matches:
            fuzzy_str = "**Fuzzy Matches** (string similarity):\n"
            for i, m in enumerate(fuzzy_matches[:5], 1):
                fuzzy_str += f"  {i}. {m['name']:30s} (score: {m['score']})\n"
        else:
            fuzzy_str = "**Fuzzy Matches:** None found\n"

        # RAG section
        if rag_chunks:
            rag_str = f'**RAG Chunks** (contextually relevant to: "{rag_concepts}"):\n'
            for i, chunk in enumerate(rag_chunks[:5], 1):
                rag_str += (
                    f"\nChunk {i} (Rank {chunk.get('rank', i)}, "
                    f"Distance: {chunk.get('distance', 0):.2f}):\n"
                )
                concepts = chunk.get("concepts", [])
                rag_str += f"  Concepts: {', '.join(concepts[:5]) if concepts else '(none)'}\n"
                entities = chunk.get("entities", [])
                rag_str += f"  Entities: {', '.join(entities[:8]) if entities else '(none)'}\n"
        else:
            rag_str = "**RAG Chunks:** None found\n"

        # IPA section
        if ipa_matches:
            ipa_str = "**IPA Phonetic Matches** (pronunciation similarity):\n"
            for i, m in enumerate(ipa_matches[:5], 1):
                ipa_str += (
                    f"  {i}. {m['entity']:30s} "
                    f"(distance: {m['distance']}, score: {m['score']})\n"
                )
        else:
            ipa_str = "**IPA Phonetic Matches:** None found\n"

        return f"""You are correcting a transcript where technical terms may be misspelled or misheard by a speech-to-text engine.

**Transcript context:**
"{transcript_context}"

**Word to evaluate:** "{original_entity}"

You are given three sources of truth:

{fuzzy_str}
{rag_str}
{ipa_str}

**Your job:**
Look at the word "{original_entity}" and use your judgment:
- **How it looks** (fuzzy): does it look like a mangled version of any of the fuzzy candidates?
- **How it sounds** (IPA): could it be a speech-to-text mishearing of any of the IPA candidates?
- **What the context implies** (RAG): given what's being discussed in the transcript, do any of the RAG candidates make sense as the intended word?

If a candidate fits on at least two of these dimensions, it's likely the correct word.
If nothing fits convincingly, leave it alone.

**Respond with ONLY valid JSON:**
{{
  "action": "auto_correct" | "ask_user" | "skip",
  "corrected_entity": "EntityName" | null,
  "confidence": 0-100,
  "reasoning": "Brief explanation (1-2 sentences)",
  "signal_agreement": "all" | "partial" | "none"
}}

- **auto_correct**: you're confident this is a mishearing/misspelling and you know what it should be
- **ask_user**: something looks off but you're not sure enough to correct automatically
- **skip**: the word looks fine as-is, or nothing from the three sources fits convincingly
"""

    def _parse_decision(self, response_text: str, original_entity: str) -> Dict:
        # Strip markdown fences
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()

        defaults = {
            "action": "ask_user",
            "corrected_entity": None,
            "confidence": 50,
            "reasoning": "No reasoning provided",
            "signal_agreement": "unknown",
        }

        try:
            decision = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find a JSON object anywhere in the response
            try:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start != -1 and end > start:
                    decision = json.loads(response_text[start:end])
                else:
                    raise ValueError("No JSON object found")
            except Exception:
                response_lower = response_text.lower()
                if "auto_correct" in response_lower or "auto correct" in response_lower:
                    action = "auto_correct"
                elif "skip" in response_lower:
                    action = "skip"
                else:
                    action = "ask_user"
                return {
                    **defaults,
                    "action": action,
                    "reasoning": f"Could not parse agent response: {response_text[:100]}",
                }

        for key, default in defaults.items():
            decision.setdefault(key, default)

        if decision["action"] not in ("auto_correct", "ask_user", "skip"):
            decision["action"] = "ask_user"

        # If corrected entity is same as original, treat as skip
        if decision.get("corrected_entity") and original_entity:
            if decision["corrected_entity"].lower() == original_entity.lower():
                decision["action"] = "skip"
                decision["reasoning"] = "No correction needed - entity is already correct"

        return decision

    def _fallback_decision(
        self,
        fuzzy_matches: List[Dict],
        rag_chunks: List[Dict],
        ipa_matches: List[Dict],
    ) -> Dict:
        """Rule-based fallback if LLM call fails."""
        candidates: Dict[str, float] = {}

        for m in fuzzy_matches[:3]:
            candidates[m["name"]] = candidates.get(m["name"], 0) + m["score"] / 100

        for chunk in rag_chunks[:5]:
            for entity in chunk.get("entities", [])[:5]:
                candidates[entity] = candidates.get(entity, 0) + 0.5

        for m in ipa_matches[:3]:
            score = max(0.0, 1 - m["distance"] / 10)
            candidates[m["entity"]] = candidates.get(m["entity"], 0) + score

        if not candidates:
            return {
                "action": "skip",
                "corrected_entity": None,
                "confidence": 0,
                "reasoning": "Fallback: No candidates from any signal",
                "signal_agreement": "none",
            }

        best_entity, best_score = max(candidates.items(), key=lambda x: x[1])
        confidence = int(min(100, best_score * 50))

        if confidence >= 80:
            action = "auto_correct"
        elif confidence >= 50:
            action = "ask_user"
        else:
            action = "skip"

        return {
            "action": action,
            "corrected_entity": best_entity,
            "confidence": confidence,
            "reasoning": f"Fallback decision based on signal agreement score: {confidence}",
            "signal_agreement": "partial",
        }
