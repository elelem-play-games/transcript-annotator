"""
Add IPA (International Phonetic Alphabet) pronunciations to entity store.

Uses espeak-ng via WSL for phonetic transcription.
Reads artifacts/entity_store.json and writes IPA fields back in-place
(or to a separate output path).

Prerequisites
-------------
    wsl sudo apt-get install -y espeak-ng

Usage
-----
    python -m pipeline.add_ipa
    python -m pipeline.add_ipa --input artifacts/entity_store.json
"""

import json
import argparse
from pathlib import Path

from core.espeak_tts import ipa_batch


def add_ipa_to_entity_store(
    entity_store_path: Path,
    output_path: Path = None,
) -> bool:
    """
    Add IPA pronunciations to all entities in the store.

    Args:
        entity_store_path: Path to entity_store.json
        output_path: Where to save the updated store (default: overwrite original)

    Returns:
        True on success, False on failure
    """
    print(f"📖 Loading entity store from: {entity_store_path}")

    with open(entity_store_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entities = data["entities"]
    entity_names = list(entities.keys())
    print(f"  Found {len(entity_names)} entities")

    # Filter out problematic entities (too long or contain colons)
    filtered_names = []
    skipped = []
    for name in entity_names:
        if len(name) > 50 or ":" in name:
            skipped.append(name)
        else:
            filtered_names.append(name)

    if skipped:
        print(
            f"  ⚠️  Skipping {len(skipped)} problematic entities "
            f"(too long or contain ':')"
        )
        for name in skipped[:5]:
            print(f"    • {name[:60]}...")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")

    entity_names = filtered_names
    print(f"  Processing {len(entity_names)} entities")

    print(f"\n🔊 Generating IPA pronunciations...")
    print(f"  (This may take a moment for {len(entity_names)} entities)")

    try:
        ipas = ipa_batch(entity_names)

        if len(ipas) != len(entity_names):
            print(
                f"  ❌ ERROR: Got {len(ipas)} IPAs for {len(entity_names)} entities"
            )
            print(f"  espeak output is misaligned!")
            return False

        print(f"\n  Adding IPAs to entities...")
        for entity_name, ipa in zip(entity_names, ipas):
            entities[entity_name]["ipa"] = ipa

        print(f"  ✅ Generated {len(ipas)} IPA pronunciations")

    except Exception as e:
        print(f"  ❌ Error generating IPAs: {e}")
        print(f"  Make sure espeak-ng is installed in WSL:")
        print(f"    wsl sudo apt-get install -y espeak-ng")
        return False

    if output_path is None:
        output_path = entity_store_path

    print(f"\n💾 Saving updated entity store to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Saved!")

    print(f"\n📊 Sample IPAs:")
    for name in list(entities.keys())[:10]:
        ipa = entities[name].get("ipa", "N/A")
        print(f"  {name:30s} → {ipa}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Add IPA pronunciations to entity store"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/entity_store.json"),
        help="Input entity store JSON (default: artifacts/entity_store.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: overwrite input)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"❌ Entity store not found: {args.input}")
        print("Run pipeline/build_entity_store.py first!")
        return

    print(f"🚀 Adding IPA pronunciations to entity store")
    print(f"{'='*70}\n")

    success = add_ipa_to_entity_store(args.input, args.output)

    print(f"\n{'='*70}")
    if success:
        print(f"✅ IPA pronunciations added successfully!")
    else:
        print(f"❌ Failed to add IPA pronunciations")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
