"""
Context-aware document chunking using an LLM for semantic analysis.

Scans all markdown files in data/documents/, parses them into sections,
enriches each section with LLM-generated metadata, and writes
artifacts/chunks.json.

Usage
-----
    # With LLM analysis (requires OPENAI_API_KEY):
    python -m pipeline.chunk_documents

    # Skip LLM analysis (just parse sections):
    python -m pipeline.chunk_documents --no-llm

    # Custom paths:
    python -m pipeline.chunk_documents --input-dir data/documents --output artifacts/chunks.json
"""

import json
import os
import re
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional

from llm.llm import LLMClient


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """Represents a semantic chunk with metadata."""
    chunk_id: str
    chapter_name: str
    chapter_file: str
    parent_section: Optional[str]
    section_title: str
    section_level: int
    text: str
    char_count: int
    # LLM-generated fields
    topic: Optional[str] = None
    summary: Optional[str] = None
    entities: Optional[List[str]] = None
    key_concepts: Optional[List[str]] = None
    should_split: Optional[bool] = None
    split_suggestion: Optional[str] = None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_markdown_sections(content: str, file_path: Path) -> List[Dict]:
    """
    Parse a markdown file into sections by scanning header boundaries.

    Tracks parent-child relationships between # and ## headers.
    Each chunk spans from one header to the next.
    """
    lines = content.split("\n")

    headers = []
    for i, line in enumerate(lines):
        match = re.match(r"^(#{1,2})\s+(.+)$", line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            headers.append({"line_num": i, "level": level, "title": title})

    if not headers:
        return []

    chapter_name = (
        headers[0]["title"] if headers[0]["level"] == 1 else "Unknown Chapter"
    )

    current_parent = None
    for header in headers:
        if header["level"] == 1:
            current_parent = header["title"]
            header["parent"] = None
        else:
            header["parent"] = current_parent

    sections = []
    for i, current in enumerate(headers):
        next_header = headers[i + 1] if i + 1 < len(headers) else None
        start_line = current["line_num"] + 1
        end_line = next_header["line_num"] if next_header else len(lines)
        text = "\n".join(lines[start_line:end_line]).strip()

        if text:
            sections.append(
                {
                    "chapter_name": chapter_name,
                    "parent_section": current.get("parent"),
                    "chapter_file": file_path.name,
                    "section_title": current["title"],
                    "section_level": current["level"],
                    "text": text,
                    "char_count": len(text),
                }
            )

    return sections


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_file(
    file_path: Path,
    llm: Optional[LLMClient],
    use_llm: bool = True,
) -> List[Chunk]:
    """Process a single markdown file into semantic chunks."""
    print(f"\n📄 Processing: {file_path.name}")

    content = file_path.read_text(encoding="utf-8")
    sections = parse_markdown_sections(content, file_path)
    print(f"  Found {len(sections)} sections")

    chunks = []
    for i, section in enumerate(sections):
        chunk_id = f"{file_path.stem}-sec{i+1:02d}"
        print(
            f"  Analysing section {i+1}/{len(sections)}: "
            f"{section['section_title'][:50]}..."
        )

        if use_llm and llm is not None:
            llm_data = llm.analyze_section(
                section_title=section["section_title"],
                text=section["text"],
            )
        else:
            llm_data = {
                "topic": section["section_title"],
                "summary": section["text"][:100],
                "entities": [],
                "key_concepts": [],
                "should_split": False,
                "split_suggestion": None,
            }

        chunk = Chunk(
            chunk_id=chunk_id,
            chapter_name=section["chapter_name"],
            chapter_file=section["chapter_file"],
            parent_section=section.get("parent_section"),
            section_title=section["section_title"],
            section_level=section["section_level"],
            text=section["text"],
            char_count=section["char_count"],
            topic=llm_data.get("topic"),
            summary=llm_data.get("summary"),
            entities=llm_data.get("entities", []),
            key_concepts=llm_data.get("key_concepts", []),
            should_split=llm_data.get("should_split", False),
            split_suggestion=llm_data.get("split_suggestion"),
        )
        chunks.append(chunk)

        if chunk.should_split:
            print(f"    💡 Split suggestion: {chunk.split_suggestion}")

    return chunks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Context-aware chunking with LLM semantic analysis"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/documents"),
        help="Directory containing markdown files (default: data/documents)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/chunks.json"),
        help="Output JSON file (default: artifacts/chunks.json)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model override (default: OPENAI_MODEL env var or gpt-4o-mini)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM analysis — just parse sections",
    )

    args = parser.parse_args()

    if not args.no_llm and not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Use --no-llm to skip LLM analysis.")
        return

    print(f"🔍 Context-aware chunking")
    print(f"📁 Input directory: {args.input_dir}")
    print(
        f"🤖 Model: "
        f"{args.model or os.getenv('OPENAI_MODEL', 'gpt-4o-mini') if not args.no_llm else 'None (LLM disabled)'}"
    )

    llm = LLMClient(model=args.model) if not args.no_llm else None

    # Find all markdown files (skip files starting with _)
    md_files = sorted(
        f for f in args.input_dir.glob("*.md") if not f.name.startswith("_")
    )

    if not md_files:
        print(f"❌ No markdown files found in {args.input_dir}")
        print(f"   Drop your .md files into {args.input_dir}/ and re-run.")
        return

    print(f"📚 Found {len(md_files)} document(s)\n")

    all_chunks: List[Chunk] = []
    for md_file in md_files:
        chunks = process_file(md_file, llm=llm, use_llm=not args.no_llm)
        all_chunks.extend(chunks)

    total_chars = sum(c.char_count for c in all_chunks)
    avg_chunk_size = total_chars // len(all_chunks) if all_chunks else 0

    all_entities: set = set()
    for chunk in all_chunks:
        if chunk.entities:
            all_entities.update(chunk.entities)

    output_data = {
        "metadata": {
            "total_chunks": len(all_chunks),
            "total_documents": len(md_files),
            "total_chars": total_chars,
            "avg_chunk_size": avg_chunk_size,
            "unique_entities": len(all_entities),
            "model_used": (args.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
            if not args.no_llm
            else None,
        },
        "chunks": [asdict(chunk) for chunk in all_chunks],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"✅ Chunking complete!")
    print(f"📊 Statistics:")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Avg chunk size: {avg_chunk_size} chars")
    print(f"  Unique entities: {len(all_entities)}")
    print(f"\n💾 Saved to: {args.output}")

    print(f"\n🔍 Sample chunks:")
    for chunk in all_chunks[:3]:
        print(f"\n  📌 {chunk.chunk_id}")
        print(f"     Document: {chunk.chapter_file}")
        print(f"     Chapter: {chunk.chapter_name}")
        print(f"     Section: {chunk.section_title}")
        print(f"     Topic: {chunk.topic}")
        print(f"     Summary: {chunk.summary}")
        print(
            f"     Entities: "
            f"{', '.join(chunk.entities[:5]) if chunk.entities else 'None'}"
        )
        print(f"     Size: {chunk.char_count} chars")


if __name__ == "__main__":
    main()
