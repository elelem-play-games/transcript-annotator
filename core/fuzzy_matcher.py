"""
Fuzzy matcher for entity correction.
Matches misspelled entities against the entity store.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from fuzzywuzzy import fuzz
import argparse


class FuzzyMatcher:
    """Fuzzy matcher for entity correction."""

    def __init__(self, entity_store_path: Path):
        """
        Initialize fuzzy matcher with entity store.

        Args:
            entity_store_path: Path to entity_store.json
        """
        print(f"📖 Loading entity store from: {entity_store_path}")

        with open(entity_store_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.entities = data["entities"]
        self.entity_names = list(self.entities.keys())

        # Build variant lookup for exact matches on known misspellings
        self.variant_to_canonical: Dict[str, str] = {}
        for canonical_name, entity_data in self.entities.items():
            self.variant_to_canonical[canonical_name.lower()] = canonical_name
            for variant in entity_data.get("variants", []):
                self.variant_to_canonical[variant.lower()] = canonical_name

        print(f"  ✅ Loaded {len(self.entities)} entities")
        print(
            f"  ✅ Loaded {len(self.variant_to_canonical)} variants "
            f"(including canonical names)"
        )

    def exact_match(self, query: str) -> Optional[str]:
        """Return canonical name for an exact (case-insensitive) match, or None."""
        return self.variant_to_canonical.get(query.lower())

    def fuzzy_match(
        self,
        query: str,
        top_k: int = 5,
        threshold: int = 60,
    ) -> List[Tuple[str, int, Dict]]:
        """
        Fuzzy match query against entity store.

        Returns list of (entity_name, score, entity_data) sorted by score desc.
        """
        matches = []
        for entity_name in self.entity_names:
            ratio = fuzz.ratio(query.lower(), entity_name.lower())
            partial_ratio = fuzz.partial_ratio(query.lower(), entity_name.lower())
            token_sort_ratio = fuzz.token_sort_ratio(query.lower(), entity_name.lower())
            score = int(0.3 * ratio + 0.4 * partial_ratio + 0.3 * token_sort_ratio)
            if score >= threshold:
                matches.append((entity_name, score, self.entities[entity_name]))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]

    def match(self, query: str, top_k: int = 5, threshold: int = 60) -> Dict:
        """
        Match query: exact match first, then fuzzy.

        Returns a result dict with keys:
            query, match_type, canonical_name, confidence, entity_data, fuzzy_matches
        """
        exact = self.exact_match(query)
        if exact:
            return {
                "query": query,
                "match_type": "exact",
                "canonical_name": exact,
                "confidence": 100,
                "entity_data": self.entities[exact],
                "fuzzy_matches": [
                    {
                        "name": exact,
                        "score": 100,
                        "frequency": self.entities[exact]["frequency"],
                    }
                ],
            }

        fuzzy_matches = self.fuzzy_match(query, top_k, threshold)
        if fuzzy_matches:
            best_match, best_score, best_data = fuzzy_matches[0]
            return {
                "query": query,
                "match_type": "fuzzy",
                "canonical_name": best_match,
                "confidence": best_score,
                "entity_data": best_data,
                "fuzzy_matches": [
                    {"name": name, "score": score, "frequency": data["frequency"]}
                    for name, score, data in fuzzy_matches
                ],
            }

        return {
            "query": query,
            "match_type": "no_match",
            "canonical_name": None,
            "confidence": 0,
            "entity_data": None,
            "fuzzy_matches": [],
        }
