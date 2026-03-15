"""
IPA Phonetic Matcher — matches words based on pronunciation similarity.
Uses Levenshtein distance on IPA representations.
"""

import json
from pathlib import Path
from typing import List, Dict

from Levenshtein import distance as levenshtein_distance

from core.espeak_tts import ipa_batch


class IPAPhoneticMatcher:
    """Phonetic matcher using IPA and Levenshtein distance."""

    def __init__(self, entity_store_path: Path):
        """
        Initialize matcher with entity store containing IPA data.

        Args:
            entity_store_path: Path to entity_store.json (must have IPA fields)
        """
        print(f"📖 Loading entity store with IPAs from: {entity_store_path}")

        with open(entity_store_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.entities = data["entities"]

        has_ipa = any("ipa" in entity for entity in self.entities.values())
        if not has_ipa:
            raise ValueError(
                "Entity store missing IPA data. Run pipeline/add_ipa.py first!"
            )

        print(f"  ✅ Loaded {len(self.entities)} entities with IPAs")

    def match(
        self,
        word: str,
        top_k: int = 10,
        max_distance: int = 4,
    ) -> List[Dict]:
        """
        Match word against entity store using phonetic similarity.

        Args:
            word: Word to match (e.g., "board mon")
            top_k: Number of top matches to return
            max_distance: Maximum Levenshtein distance to consider

        Returns:
            List of match dicts: entity, entity_ipa, distance, score, frequency
        """
        print(f"\n🔊 Phonetic matching: '{word}'")

        try:
            word_ipa = ipa_batch([word])[0]
            print(f"  IPA: {word_ipa}")
        except Exception as e:
            print(f"  ⚠️  Failed to get IPA: {e}")
            return []

        matches = []
        for entity_name, entity_data in self.entities.items():
            entity_ipa = entity_data.get("ipa", "")
            if not entity_ipa:
                continue

            dist = levenshtein_distance(word_ipa, entity_ipa)
            if dist <= max_distance:
                max_len = max(len(word_ipa), len(entity_ipa))
                score = int(100 * (1 - dist / max_len)) if max_len > 0 else 0
                matches.append(
                    {
                        "entity": entity_name,
                        "entity_ipa": entity_ipa,
                        "distance": dist,
                        "score": score,
                        "frequency": entity_data.get("frequency", 0),
                    }
                )

        matches.sort(key=lambda x: (x["distance"], -x["frequency"]))
        top_matches = matches[:top_k]

        print(f"  Found {len(matches)} matches within distance {max_distance}")
        for m in top_matches:
            print(f"    • {m['entity']:30s} (distance: {m['distance']}, score: {m['score']})")
            print(f"      IPA: {m['entity_ipa']}")

        return top_matches

    def batch_match(
        self,
        words: List[str],
        top_k: int = 10,
        max_distance: int = 4,
    ) -> Dict[str, List[Dict]]:
        """Match multiple words at once (batches the IPA call for speed)."""
        print(f"\n🔊 Batch phonetic matching for {len(words)} words")

        try:
            word_ipas = ipa_batch(words)
        except Exception as e:
            print(f"  ⚠️  Failed to get IPAs: {e}")
            return {}

        results: Dict[str, List[Dict]] = {}
        for word, word_ipa in zip(words, word_ipas):
            print(f"\n  '{word}' → {word_ipa}")
            matches = []
            for entity_name, entity_data in self.entities.items():
                entity_ipa = entity_data.get("ipa", "")
                if not entity_ipa:
                    continue
                dist = levenshtein_distance(word_ipa, entity_ipa)
                if dist <= max_distance:
                    max_len = max(len(word_ipa), len(entity_ipa))
                    score = int(100 * (1 - dist / max_len)) if max_len > 0 else 0
                    matches.append(
                        {
                            "entity": entity_name,
                            "entity_ipa": entity_ipa,
                            "distance": dist,
                            "score": score,
                            "frequency": entity_data.get("frequency", 0),
                        }
                    )
            matches.sort(key=lambda x: (x["distance"], -x["frequency"]))
            results[word] = matches[:top_k]
            print(
                f"    Top matches: "
                f"{', '.join(m['entity'] for m in results[word][:3])}"
            )

        return results
