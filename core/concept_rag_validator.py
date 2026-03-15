"""
Concept-based RAG Validator.

Queries ChromaDB using concepts (not entity names), then checks which entities
appear in the retrieved chunks.
"""

import json
from typing import Dict, List

import chromadb

from llm.llm import LLMClient


class ConceptBasedRAGValidator:
    """RAG validator that queries by concepts, not entity names."""

    def __init__(
        self,
        chroma_db_path: str = "./artifacts/chroma_db",
        collection_name: str = "text_chunks",
        model: str = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        Initialize validator.

        Args:
            chroma_db_path: Path to the persisted ChromaDB directory
            collection_name: Name of the ChromaDB collection
            model: LLM model override
            embedding_model: OpenAI embedding model to use for queries
        """
        print(f"📖 Loading ChromaDB from: {chroma_db_path}")
        print(f"   Collection: {collection_name}")

        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.chroma_client.get_collection(collection_name)
        self.llm = LLMClient(model=model)
        self.embedding_model = embedding_model

        print(f"  ✅ Loaded collection with {self.collection.count()} chunks")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        transcript_context: str,
        candidate_entity: str,
        fuzzy_score: int,
        top_k: int = 10,
    ) -> Dict:
        """
        Validate entity using concept-based RAG.

        Args:
            transcript_context: The transcript snippet (may contain misspelled entity)
            candidate_entity: The entity we're validating (correct spelling)
            fuzzy_score: The fuzzy match score
            top_k: Number of chunks to retrieve

        Returns:
            Dict with validation results
        """
        print(f"\n🔍 Validating: '{candidate_entity}'")
        print(f"   Context: '{transcript_context[:80]}...'")

        # Step 1: Extract concepts (no entity name)
        concepts = self._extract_concepts(transcript_context)
        print(f"   📝 Concepts: '{concepts}'")

        # Step 2: Query ChromaDB with concept embedding
        query_embedding = self.llm.embed([concepts], model=self.embedding_model)[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        # Step 3: Analyse retrieved chunks
        entity_mentions = 0
        matched_chunks = []
        all_entities_found: set = set()

        for i, (doc, metadata) in enumerate(
            zip(results["documents"][0], results["metadatas"][0])
        ):
            chunk_entities = json.loads(metadata.get("section_entities", "[]"))
            chunk_concepts = json.loads(metadata.get("section_concepts", "[]"))
            all_entities_found.update(chunk_entities)

            entity_found = any(
                candidate_entity.lower() in e.lower() for e in chunk_entities
            )
            if entity_found:
                entity_mentions += 1

            matched_chunks.append(
                {
                    "rank": i + 1,
                    "chapter": metadata.get("chapter_name", ""),
                    "section": metadata.get("section_title", ""),
                    "topic": metadata.get("section_topic", ""),
                    "concepts": chunk_concepts,
                    "entities": chunk_entities,
                    "distance": results["distances"][0][i],
                    "text_preview": doc[:200],
                    "has_candidate": entity_found,
                }
            )

        # Step 4: Score
        validation_score = self._calculate_validation_score(
            entity_mentions=entity_mentions,
            total_retrieved=top_k,
            fuzzy_score=fuzzy_score,
            matched_chunks=matched_chunks,
        )

        result = {
            "candidate_entity": candidate_entity,
            "transcript_context": transcript_context,
            "extracted_concepts": concepts,
            "fuzzy_score": fuzzy_score,
            "validation_score": validation_score,
            "context_match": entity_mentions > 0,
            "entity_mentions": entity_mentions,
            "total_retrieved": top_k,
            "matched_chunks": matched_chunks,
            "all_entities_found": sorted(list(all_entities_found))[:20],
        }

        print(f"   Entity mentions: {entity_mentions}/{top_k}")
        print(f"   Validation score: {validation_score}")
        print(f"   Context match: {result['context_match']}")

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_concepts(self, transcript_context: str) -> str:
        """Extract technical concepts from transcript context (no entity names)."""
        prompt = f"""Extract the key technical concepts and actions from this transcript sentence.

**Transcript:** "{transcript_context}"

What is being discussed? What operation, system, or topic?

Examples:
- "Board mon showed 500ms latency" → "monitoring system performance metrics latency"
- "The big table cluster is out of storage" → "database cluster storage capacity management"
- "Our build system using blase is slow" → "build system compilation performance"
- "Check DNS resolution" → "DNS name resolution troubleshooting"

Respond with ONLY the key concepts (as a short phrase), nothing else. Do NOT include any specific tool/product names."""

        try:
            return self.llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a technical concept extractor.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=100,
            ).strip("\"'")
        except Exception as e:
            print(f"  ⚠️  Concept extraction failed: {e}")
            return transcript_context

    def _calculate_validation_score(
        self,
        entity_mentions: int,
        total_retrieved: int,
        fuzzy_score: int,
        matched_chunks: List[Dict],
    ) -> int:
        mention_rate = entity_mentions / total_retrieved
        mention_score = int(mention_rate * 50)

        if matched_chunks:
            entity_chunks = [c for c in matched_chunks if c["has_candidate"]]
            if entity_chunks:
                avg_distance = sum(c["distance"] for c in entity_chunks) / len(entity_chunks)
            else:
                avg_distance = (
                    sum(c["distance"] for c in matched_chunks[:3]) / min(3, len(matched_chunks))
                )
            similarity_score = int(max(0, (1 - avg_distance) * 30))
        else:
            similarity_score = 0

        fuzzy_contribution = (
            int((fuzzy_score - 60) / 40 * 20) if fuzzy_score >= 60 else 0
        )

        return min(100, mention_score + similarity_score + fuzzy_contribution)
