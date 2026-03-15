"""
Transcript Corrector — Multi-signal approach (fuzzy + RAG + IPA).
Treats all three signals as equal sources of truth.
"""

import json
import re
from pathlib import Path
from typing import List, Dict
import argparse

from core.entity_extractor import EntityExtractor
from core.fuzzy_matcher import FuzzyMatcher
from core.concept_rag_validator import ConceptBasedRAGValidator
from core.ipa_phonetic_matcher import IPAPhoneticMatcher
from core.agent_decision import MultiSignalAgent


class TranscriptCorrector:
    """Multi-signal transcript corrector."""

    def __init__(
        self,
        entity_store_path: Path = Path("artifacts/entity_store.json"),
        chroma_db_path: str = "./artifacts/chroma_db",
        collection_name: str = "text_chunks",
        model: str = None,
        use_ipa: bool = True,
    ):
        """
        Initialize the corrector.

        Args:
            entity_store_path: Path to entity_store.json
            chroma_db_path: Path to ChromaDB directory
            collection_name: ChromaDB collection name
            model: LLM model override (uses OPENAI_MODEL env var or default if None)
            use_ipa: Whether to enable IPA phonetic matching
        """
        print(f"🚀 Initializing Transcript Corrector (Multi-Signal)")
        print(f"{'='*70}\n")

        self.entity_extractor = EntityExtractor(model=model)
        self.fuzzy_matcher = FuzzyMatcher(entity_store_path)
        self.rag_validator = ConceptBasedRAGValidator(chroma_db_path, collection_name, model)
        self.agent = MultiSignalAgent(model=model)

        self.use_ipa = use_ipa
        self.ipa_matcher = None
        if use_ipa:
            try:
                self.ipa_matcher = IPAPhoneticMatcher(entity_store_path)
                print(f"  ✅ IPA phonetic matching enabled")
            except Exception as e:
                print(f"  ⚠️  IPA matcher initialization failed: {e}")
                print(f"  ⚠️  Continuing without IPA matching")
                self.use_ipa = False
        else:
            print(f"  ⚠️  IPA phonetic matching disabled")

        print(f"\n✅ Pipeline fully initialized!\n")

    def correct(
        self,
        transcript: str,
        fuzzy_threshold: int = 60,
        auto_correct_only: bool = False,
    ) -> Dict:
        """
        Correct entities in transcript using multi-signal approach.

        Args:
            transcript: Raw transcript text
            fuzzy_threshold: Minimum fuzzy match score (0-100)
            auto_correct_only: If True, only apply auto-corrections (skip ask_user)

        Returns:
            Dict with keys: original, corrected, corrections, stats
        """
        print(f"\n{'='*70}")
        print(f"🔧 Processing Transcript (Multi-Signal)")
        print(f"{'='*70}")
        print(f"\nOriginal transcript:\n{transcript}\n")

        # Step 1: Extract entities
        print(f"\n{'='*70}")
        print(f"Step 1: Entity Extraction")
        print(f"{'='*70}")

        entities_with_context = self.entity_extractor.extract_with_context(transcript)

        if not entities_with_context:
            print(f"  ⚠️  No entities extracted")
            return {
                "original": transcript,
                "corrected": transcript,
                "corrections": [],
                "stats": {
                    "entities_found": 0,
                    "corrections_made": 0,
                    "auto_corrected": 0,
                    "asked_user": 0,
                    "skipped": 0,
                    "no_match": 0,
                },
            }

        print(f"\n  Found {len(entities_with_context)} entities:")
        for e in entities_with_context:
            print(f"    • {e['entity']} (appears {e['context_count']} times)")

        # Step 2: Process each entity through all three signals
        corrections = []
        stats = {
            "entities_found": len(entities_with_context),
            "corrections_made": 0,
            "auto_corrected": 0,
            "asked_user": 0,
            "skipped": 0,
            "no_match": 0,
        }

        for entity_data in entities_with_context:
            entity = entity_data["entity"]
            contexts = entity_data["contexts"]
            context = contexts[0] if contexts else transcript

            print(f"\n{'='*70}")
            print(f"Processing entity: '{entity}'")
            print(f"{'='*70}")

            # Signal 1: Fuzzy matching
            print(f"\n📍 Signal 1: Fuzzy Matching")
            fuzzy_result = self.fuzzy_matcher.match(entity, top_k=5, threshold=fuzzy_threshold)

            if fuzzy_result["match_type"] == "no_match":
                print(f"  ⚠️  No fuzzy matches found - skipping")
                stats["no_match"] += 1
                continue

            fuzzy_matches = fuzzy_result.get("fuzzy_matches", [])
            print(f"  Found {len(fuzzy_matches)} matches")
            for m in fuzzy_matches[:3]:
                print(f"    • {m['name']:30s} (score: {m['score']})")

            # Signal 2: RAG context
            print(f"\n📍 Signal 2: RAG Context")
            rag_result = self.rag_validator.validate(
                transcript_context=context,
                candidate_entity=fuzzy_result["canonical_name"],
                fuzzy_score=fuzzy_result["confidence"],
                top_k=10,
            )

            rag_chunks = rag_result.get("matched_chunks", [])
            rag_concepts = rag_result.get("extracted_concepts", "")
            print(f"  Concepts: {rag_concepts}")
            print(f"  Found {len(rag_chunks)} chunks")
            if rag_chunks:
                print(f"  Top chunk: {rag_chunks[0]['chapter']} - {rag_chunks[0]['section']}")

            # Signal 3: IPA phonetic matching
            ipa_matches = []
            if self.use_ipa and self.ipa_matcher:
                print(f"\n📍 Signal 3: IPA Phonetic Matching")
                try:
                    ipa_matches = self.ipa_matcher.match(word=entity, top_k=10, max_distance=6)
                    if ipa_matches:
                        print(f"  Found {len(ipa_matches)} matches")
                        for m in ipa_matches[:3]:
                            print(f"    • {m['entity']:30s} (distance: {m['distance']})")
                    else:
                        print(f"  No phonetic matches found")
                except Exception as e:
                    print(f"  ⚠️  IPA matching failed: {e}")
                    ipa_matches = []

            # Agent decision
            print(f"\n📍 Step 4: Multi-Signal Agent Decision")
            decision = self.agent.decide(
                original_entity=entity,
                transcript_context=context,
                fuzzy_matches=fuzzy_matches,
                rag_chunks=rag_chunks,
                ipa_matches=ipa_matches,
                rag_concepts=rag_concepts,
            )

            print(f"  Action: {decision['action']}")
            print(f"  Entity: {decision.get('corrected_entity', 'N/A')}")
            print(f"  Confidence: {decision['confidence']}")
            print(f"  Signal agreement: {decision.get('signal_agreement', 'N/A')}")

            correction = {
                "original": entity,
                "corrected": decision.get("corrected_entity"),
                "action": decision["action"],
                "confidence": decision["confidence"],
                "reasoning": decision["reasoning"],
                "signal_agreement": decision.get("signal_agreement", "unknown"),
                "contexts": contexts,
                "fuzzy_matches": fuzzy_matches,
                "rag_chunks": rag_chunks[:5],
                "rag_concepts": rag_concepts,
                "ipa_matches": ipa_matches[:5] if ipa_matches else [],
            }

            corrections.append(correction)

            if decision["action"] == "auto_correct":
                stats["auto_corrected"] += 1
                stats["corrections_made"] += 1
            elif decision["action"] == "ask_user":
                stats["asked_user"] += 1
                if not auto_correct_only:
                    stats["corrections_made"] += 1
            elif decision["action"] == "skip":
                stats["skipped"] += 1

        # Step 3: Apply corrections
        corrected_transcript = self._apply_corrections(
            transcript, corrections, auto_correct_only=auto_correct_only
        )

        result = {
            "original": transcript,
            "corrected": corrected_transcript,
            "corrections": corrections,
            "stats": stats,
        }

        self._print_summary(result)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_corrections(
        self,
        transcript: str,
        corrections: List[Dict],
        auto_correct_only: bool = False,
    ) -> str:
        corrected = transcript
        for correction in corrections:
            if correction["action"] == "auto_correct":
                apply = True
            elif correction["action"] == "ask_user" and not auto_correct_only:
                apply = True
            else:
                apply = False

            if apply and correction.get("corrected"):
                pattern = re.compile(re.escape(correction["original"]), re.IGNORECASE)
                corrected = pattern.sub(correction["corrected"], corrected)

        return corrected

    def _print_summary(self, result: Dict):
        print(f"\n{'='*70}")
        print(f"📊 Correction Summary")
        print(f"{'='*70}")

        stats = result["stats"]
        print(f"\nEntities found: {stats['entities_found']}")
        print(f"Corrections made: {stats['corrections_made']}")
        print(f"  • Auto-corrected: {stats['auto_corrected']}")
        print(f"  • Asked user: {stats['asked_user']}")
        print(f"  • Skipped: {stats['skipped']}")
        print(f"  • No match: {stats['no_match']}")

        if result["corrections"]:
            print(f"\n📝 Corrections:")
            for i, corr in enumerate(result["corrections"], 1):
                action_emoji = {
                    "auto_correct": "✅",
                    "ask_user": "❓",
                    "skip": "⏭️",
                }.get(corr["action"], "•")
                print(
                    f"\n  {i}. {action_emoji} '{corr['original']}' "
                    f"→ '{corr.get('corrected', 'N/A')}'"
                )
                print(f"     Action: {corr['action']}")
                print(f"     Confidence: {corr['confidence']}")
                print(f"     Signal agreement: {corr.get('signal_agreement', 'N/A')}")
                print(f"     Reasoning: {corr['reasoning'][:80]}...")

        print(f"\n{'='*70}")
        print(f"Corrected transcript:")
        print(f"{'='*70}")
        print(f"\n{result['corrected']}\n")


def main():
    parser = argparse.ArgumentParser(description="Transcript corrector — multi-signal")
    parser.add_argument(
        "--entity-store",
        type=Path,
        default=Path("artifacts/entity_store.json"),
    )
    parser.add_argument("--chroma-db", default="./artifacts/chroma_db")
    parser.add_argument("--collection", default="text_chunks")
    parser.add_argument("--model", default=None, help="LLM model override")
    parser.add_argument("--transcript", type=str)
    parser.add_argument("--auto-correct-only", action="store_true")
    parser.add_argument("--no-ipa", action="store_true")

    args = parser.parse_args()

    if not args.entity_store.exists():
        print(f"❌ Entity store not found: {args.entity_store}")
        return

    if not Path(args.chroma_db).exists():
        print(f"❌ ChromaDB not found: {args.chroma_db}")
        return

    corrector = TranscriptCorrector(
        entity_store_path=args.entity_store,
        chroma_db_path=args.chroma_db,
        collection_name=args.collection,
        model=args.model,
        use_ipa=not args.no_ipa,
    )

    if args.transcript:
        corrector.correct(args.transcript, auto_correct_only=args.auto_correct_only)
    else:
        test_transcripts = [
            "We had a latency spike yesterday. Board mon showed 500ms p99 latency. "
            "The team investigated board mon logs and found the issue was related to "
            "the big table cluster running out of storage space.",
            "Our build system using blase is taking too long to compile. We need to "
            "look into hermetic builds and make sure the build environment is consistent. "
            "The DNS resolution has also been slow lately.",
            "During last night's incident, the on call engineer took 45 minutes to "
            "identify the root cause. We need to reduce toil by automating more of our "
            "deployment process and improving our runbooks.",
        ]

        for i, transcript in enumerate(test_transcripts, 1):
            print(f"\n{'#'*70}")
            print(f"# Test Transcript {i}")
            print(f"{'#'*70}")

            result = corrector.correct(
                transcript, auto_correct_only=args.auto_correct_only
            )

            output_file = Path(f"correction_result_{i}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Saved result to: {output_file}")

    print(f"\n{'='*70}")
    print(f"✅ Pipeline complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
