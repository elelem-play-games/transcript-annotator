"""
Build entity store from artifacts/chunks.json.

Extracts all unique entities and their contexts for fuzzy matching,
and writes artifacts/entity_store.json.

Usage
-----
    python -m pipeline.build_entity_store
    python -m pipeline.build_entity_store --input artifacts/chunks.json --output artifacts/entity_store.json
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict


def build_entity_store(chunks_file: Path) -> Dict:
    """
    Extract all unique entities from chunks.json and build entity store.

    Returns a dict mapping entity name → metadata.
    """
    print(f"📖 Loading chunks from: {chunks_file}")

    with open(chunks_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data["chunks"]
    print(f"  Found {len(chunks)} chunks")

    entity_data = defaultdict(
        lambda: {
            "canonical_name": "",
            "appears_in": [],
            "contexts": set(),
            "summaries": [],
            "variants": [],
        }
    )

    for chunk in chunks:
        entities = chunk.get("entities", [])
        chunk_id = chunk["chunk_id"]
        chapter = chunk["chapter_name"]
        section = chunk["section_title"]
        summary = chunk.get("summary", "")
        topic = chunk.get("topic", "")

        for entity in entities:
            if not entity or not entity.strip():
                continue

            name = entity.strip()

            if not entity_data[name]["canonical_name"]:
                entity_data[name]["canonical_name"] = name

            entity_data[name]["appears_in"].append(
                {"chunk_id": chunk_id, "chapter": chapter, "section": section}
            )

            if topic:
                entity_data[name]["contexts"].add(topic)
            if summary:
                entity_data[name]["summaries"].append(summary)

    entity_store = {}
    for name, data in entity_data.items():
        entity_store[name] = {
            "canonical_name": data["canonical_name"],
            "appears_in": data["appears_in"],
            "contexts": list(data["contexts"]),
            "summaries": data["summaries"][:3],
            "variants": data["variants"],
            "frequency": len(data["appears_in"]),
        }

    print(f"\n✅ Extracted {len(entity_store)} unique entities")
    return entity_store


def print_entity_stats(entity_store: Dict):
    print(f"\n{'='*60}")
    print(f"📊 Entity Store Statistics")
    print(f"{'='*60}")

    total = len(entity_store)
    print(f"\nTotal unique entities: {total}")

    frequencies = [e["frequency"] for e in entity_store.values()]
    if frequencies:
        avg_freq = sum(frequencies) / len(frequencies)
        max_freq = max(frequencies)
        print(f"Average appearances per entity: {avg_freq:.1f}")
        print(f"Max appearances: {max_freq}")

    sorted_entities = sorted(
        entity_store.items(), key=lambda x: x[1]["frequency"], reverse=True
    )

    print(f"\n🔝 Top 10 most frequent entities:")
    for i, (name, data) in enumerate(sorted_entities[:10], 1):
        print(f"  {i:2d}. {name:30s} (appears {data['frequency']} times)")

    print(f"\n🔍 Sample entities with contexts:")
    for i, (name, data) in enumerate(sorted_entities[:5], 1):
        print(f"\n  {i}. {name}")
        print(f"     Frequency: {data['frequency']}")
        print(f"     Contexts: {', '.join(data['contexts'][:3])}")
        if data["summaries"]:
            print(f"     Example: {data['summaries'][0][:80]}...")


def save_entity_store(entity_store: Dict, output_file: Path):
    print(f"\n💾 Saving entity store to: {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metadata": {
            "total_entities": len(entity_store),
            "source": "chunks.json",
            "description": "Entity store for transcript correction",
        },
        "entities": entity_store,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(entity_store)} entities")


def main():
    parser = argparse.ArgumentParser(
        description="Build entity store from chunks.json"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/chunks.json"),
        help="Input chunks.json (default: artifacts/chunks.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/entity_store.json"),
        help="Output entity store JSON (default: artifacts/entity_store.json)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"❌ Input file not found: {args.input}")
        print("Run pipeline/chunk_documents.py first!")
        return

    print(f"🚀 Building entity store")
    print(f"{'='*60}\n")

    entity_store = build_entity_store(args.input)
    print_entity_stats(entity_store)
    save_entity_store(entity_store, args.output)

    print(f"\n{'='*60}")
    print(f"✅ Entity store built successfully!")
    print(f"📁 Output: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
