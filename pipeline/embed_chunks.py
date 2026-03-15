"""
Split chunks into token-limited overlapping pieces and store in ChromaDB.

Reads artifacts/chunks.json, splits large sections into ≤500-token sub-chunks
with 50-token overlap, generates OpenAI embeddings, and persists everything
to artifacts/chroma_db/.

Usage
-----
    python -m pipeline.embed_chunks
    python -m pipeline.embed_chunks --max-tokens 300 --overlap-tokens 30
"""

import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import chromadb
import tiktoken

from llm.llm import LLMClient


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SubChunk:
    id: str
    text: str
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def split_text_by_tokens(
    text: str,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
    encoding_name: str = "cl100k_base",
) -> List[str]:
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunks.append(enc.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = end - overlap_tokens
    return chunks


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_chunks(
    chunks_file: Path,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> List[SubChunk]:
    """Load chunks.json and split into token-limited sub-chunks."""
    print(f"📖 Loading chunks from: {chunks_file}")

    with open(chunks_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data["chunks"]
    print(f"  Found {len(chunks)} sections")

    sub_chunks: List[SubChunk] = []

    for chunk in chunks:
        text = chunk["text"]
        token_count = count_tokens(text)

        base_metadata = {
            "chunk_id": chunk["chunk_id"],
            "chapter_name": chunk["chapter_name"],
            "chapter_file": chunk["chapter_file"],
            "parent_section": chunk.get("parent_section") or "",
            "section_title": chunk["section_title"],
            "section_level": chunk["section_level"],
            "section_summary": chunk.get("summary") or "",
            "section_topic": chunk.get("topic") or "",
            "section_entities": json.dumps(chunk.get("entities", [])),
            "section_concepts": json.dumps(chunk.get("key_concepts", [])),
        }

        if token_count <= max_tokens:
            sub_chunks.append(
                SubChunk(
                    id=f"{chunk['chunk_id']}-00",
                    text=text,
                    metadata={
                        **base_metadata,
                        "sub_chunk_index": 0,
                        "total_sub_chunks": 1,
                        "token_count": token_count,
                    },
                )
            )
        else:
            pieces = split_text_by_tokens(text, max_tokens, overlap_tokens)
            for i, piece in enumerate(pieces):
                sub_chunks.append(
                    SubChunk(
                        id=f"{chunk['chunk_id']}-{i:02d}",
                        text=piece,
                        metadata={
                            **base_metadata,
                            "sub_chunk_index": i,
                            "total_sub_chunks": len(pieces),
                            "token_count": count_tokens(piece),
                        },
                    )
                )

    print(f"  Created {len(sub_chunks)} sub-chunks")
    return sub_chunks


def create_chromadb_collection(
    sub_chunks: List[SubChunk],
    llm: LLMClient,
    collection_name: str = "text_chunks",
    persist_directory: str = "./artifacts/chroma_db",
    embedding_model: str = "text-embedding-3-small",
    batch_size: int = 100,
) -> chromadb.Collection:
    """Embed sub-chunks and store in a ChromaDB collection."""
    print(f"\n🔧 Creating ChromaDB collection: {collection_name}")
    print(f"📁 Persist directory: {persist_directory}")
    print(f"🤖 Embedding model: {embedding_model}")

    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=persist_directory)

    try:
        chroma_client.delete_collection(collection_name)
        print(f"  🗑️  Deleted existing collection")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"description": "Document text chunks for RAG"},
    )

    ids = [sc.id for sc in sub_chunks]
    documents = [sc.text for sc in sub_chunks]
    metadatas = [sc.metadata for sc in sub_chunks]

    print(f"\n🔢 Generating embeddings for {len(documents)} chunks...")
    all_embeddings: List[List[float]] = []

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(documents) - 1) // batch_size + 1
        print(f"  Processing batch {batch_num}/{total_batches}...")
        all_embeddings.extend(llm.embed(batch, model=embedding_model))
        print(f"  Processed {min(i + batch_size, len(documents))}/{len(documents)}")

    print(f"\n💾 Adding to ChromaDB...")
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=all_embeddings,
        metadatas=metadatas,
    )

    print(f"✅ Added {len(sub_chunks)} chunks to collection")
    return collection


def test_retrieval(
    collection: chromadb.Collection,
    llm: LLMClient,
    query: str = "monitoring latency",
    embedding_model: str = "text-embedding-3-small",
):
    """Test retrieval with a sample query."""
    print(f"\n🔍 Testing retrieval with query: '{query}'")

    query_embedding = llm.embed([query], model=embedding_model)[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    print(f"\n📊 Top 3 results:")
    for i, (doc, metadata, distance) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        1,
    ):
        print(f"\n{i}. Score: {1 - distance:.4f}")
        print(f"   Chapter: {metadata['chapter_name']}")
        print(f"   Section: {metadata['section_title']}")
        print(f"   Topic: {metadata.get('section_topic', 'N/A')}")
        print(f"   Entities: {metadata.get('section_entities', '[]')}")
        print(f"   Text preview: {doc[:150]}...")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Split chunks and store embeddings in ChromaDB"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/chunks.json"),
        help="Input chunks.json (default: artifacts/chunks.json)",
    )
    parser.add_argument(
        "--collection-name",
        default="text_chunks",
        help="ChromaDB collection name (default: text_chunks)",
    )
    parser.add_argument(
        "--persist-dir",
        default="./artifacts/chroma_db",
        help="ChromaDB persist directory (default: ./artifacts/chroma_db)",
    )
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--overlap-tokens", type=int, default=50)
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model override (unused here, kept for consistency)",
    )
    parser.add_argument("--test-query", default="monitoring latency")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"❌ Input file not found: {args.input}")
        print("Run pipeline/chunk_documents.py first!")
        return

    print(f"🚀 Starting embedding pipeline")
    print(f"{'='*60}\n")

    llm = LLMClient(model=args.model)

    sub_chunks = process_chunks(
        args.input,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )

    collection = create_chromadb_collection(
        sub_chunks,
        llm=llm,
        collection_name=args.collection_name,
        persist_directory=args.persist_dir,
        embedding_model=args.embedding_model,
    )

    test_retrieval(
        collection,
        llm=llm,
        query=args.test_query,
        embedding_model=args.embedding_model,
    )

    print(f"\n{'='*60}")
    print(f"✅ Pipeline complete!")
    print(f"📁 Database saved to: {args.persist_dir}")
    print(f"🔍 Collection: {args.collection_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
