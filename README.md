# Transcript Annotator

## The Problem

Collaboration platforms like Microsoft Teams and Zoom ship with built-in transcription that works well for everyday English — but falls apart the moment a meeting turns technical. Internal project codenames, product acronyms, tool names, and people's names are routinely mangled.

The deeper issue is that there is **no practical way to fix this**. Vendors don't expose fine-tuning hooks for their ASR models, and standing up a custom speech pipeline is expensive and operationally heavy.

This project experiments with a different approach: **leave the ASR model alone and fix the transcript afterwards**. This is done by mining your own documents for domain-specific entities and combining three independent correction signals — fuzzy string matching, retrieval-augmented context, and phonetic similarity.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ONE-TIME SETUP PIPELINE                  │
│                  (run pipeline/ scripts once)               │
└─────────────────────────────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
  │ data/         │  │ data/         │  │ data/         │
  │ documents/    │  │ documents/    │  │ documents/    │
  │ doc_1.md      │  │ doc_2.md      │  │ ...           │
  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
          └──────────────────┼──────────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  chunk_documents.py │  Parse markdown → sections
                  │  LLM: topic,        │  LLM enriches each section
                  │  summary, entities  │  with metadata
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  artifacts/         │
                  │  chunks.json        │
                  └──────────┬──────────┘
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
  ┌────────────────────┐       ┌────────────────────┐
  │  embed_chunks.py   │       │ build_entity_      │
  │  Split → 500-token │       │ store.py           │
  │  sub-chunks        │       │ Collect all unique │
  │  OpenAI embeddings │       │ entities + contexts│
  └─────────┬──────────┘       └──────────┬─────────┘
            │                             │
            ▼                             ▼
  ┌──────────────────┐        ┌──────────────────────┐
  │ artifacts/       │        │ artifacts/           │
  │ chroma_db/       │        │ entity_store.json    │
  │ (vector store)   │        │                      │
  └──────────────────┘        └──────────┬───────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │  add_ipa.py          │
                              │  espeak-ng →         │
                              │  IPA per entity      │
                              └──────────┬───────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │ artifacts/            │
                              │ entity_store.json     │
                              │ (+ ipa fields)        │
                              └──────────────────────┘


┌─────────────────────────────────────────────────────────────┐
│                    CORRECTION RUNTIME                       │
│                     (app/app.py)                            │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Raw Transcript    │
                  │  (pasted by user)   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  EntityExtractor    │  LLM identifies candidate
                  │  (LLM)              │  tokens to verify
                  └──────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │   Signal 1   │  │   Signal 2   │  │   Signal 3   │
  │    Fuzzy     │  │     RAG      │  │     IPA      │
  │   Matcher    │  │  Validator   │  │   Matcher    │
  │              │  │              │  │              │
  │ fuzzywuzzy   │  │ Concept →    │  │ espeak-ng →  │
  │ string sim   │  │ ChromaDB     │  │ Levenshtein  │
  │ vs entity    │  │ query →      │  │ on phonetic  │
  │ store        │  │ chunk lookup │  │ strings      │
  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  MultiSignalAgent   │  LLM weighs all three
                │  (LLM)              │  signals holistically
                └──────────┬──────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌────────────┐ ┌──────────┐ ┌────────┐
       │auto_correct│ │ ask_user │ │  skip  │
       └─────┬──────┘ └────┬─────┘ └───┬────┘
             │             │           │
             │             ▼           │
             │    ┌─────────────────┐  │
             │    │  Streamlit UI   │  │
             │    │  Accept/Reject  │  │
             │    └────────┬────────┘  │
             │             │           │
             └─────────────┼───────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  Corrected          │
                │  Transcript         │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  MLflow Logging     │  Per-correction run:
                │                     │  • signal scores
                │                     │  • agent reasoning
                │                     │  • action taken
                │                     │  • user decision
                │                     │  → offline evaluation
                │                     │    & threshold tuning
                └─────────────────────┘
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:

```bash
# .env or shell
export OPENAI_API_KEY=sk-...
```

For IPA phonetic matching, install `espeak-ng` inside WSL:

```bash
wsl sudo apt-get install -y espeak-ng
```

### 2. Add your documents

Drop any number of Markdown files into `data/documents/`:

```
data/documents/
    internal_glossary.md
    project_handbook.md
    team_runbook.md
```

### 3. Run the setup pipeline (once)

```bash
# Step 1 — chunk & enrich with LLM metadata
python -m pipeline.chunk_documents

# Step 2 — embed chunks into ChromaDB
python -m pipeline.embed_chunks

# Step 3 — build entity store
python -m pipeline.build_entity_store

# Step 4 — add IPA pronunciations (requires espeak-ng in WSL)
python -m pipeline.add_ipa
```

All outputs land in `artifacts/` (gitignored).

### 4. Launch the app

```bash
streamlit run app/app.py
```
