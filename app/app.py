"""
Streamlit UI for Transcript Corrector.
Multi-signal approach: Fuzzy + RAG + IPA.

Run from the project root:
    streamlit run app/app.py
"""

import json
import re
import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path when launched via `streamlit run app/app.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.transcript_corrector import TranscriptCorrector

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Transcript Annotator", layout="wide")


# ---------------------------------------------------------------------------
# Cached resource
# ---------------------------------------------------------------------------

@st.cache_resource
def load_corrector():
    try:
        return TranscriptCorrector(
            entity_store_path=ROOT / "artifacts" / "entity_store.json",
            chroma_db_path=str(ROOT / "artifacts" / "chroma_db"),
            collection_name="text_chunks",
            use_ipa=True,
        )
    except Exception as e:
        st.error(f"Failed to initialise corrector: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def highlight_text(text: str, corrections: list, show_original: bool = False) -> str:
    """Apply colour highlights to text based on corrections."""
    replacements = {}
    for corr in corrections:
        original = corr["original"]
        corrected = corr.get("corrected")
        action = corr["action"]

        if not corrected:
            continue
        if original.lower() == corrected.lower():
            continue

        if action == "auto_correct":
            color = "#90EE90"
            display = f"{original} → {corrected}" if show_original else corrected
        elif action == "ask_user":
            color = "#FFFF99"
            display = f"{original} → {corrected}?"
        else:
            color = "#E0E0E0"
            display = original

        highlighted = (
            f'<span style="background-color: {color}; padding: 2px 4px; '
            f'border-radius: 3px;">{display}</span>'
        )
        replacements[original] = highlighted

    result = text
    for original, highlighted in replacements.items():
        pattern = re.compile(re.escape(original), re.IGNORECASE)
        result = pattern.sub(highlighted, result, count=1)

    return result


def show_debug_panel(corrections: list):
    """Show debug information for each correction with all signals."""
    st.subheader("🔍 Debug Panel — Multi-Signal Analysis")

    for i, corr in enumerate(corrections, 1):
        action = corr["action"]
        emoji = {"auto_correct": "🟢", "ask_user": "🟡"}.get(action, "⚪")
        title = (
            f"{emoji} {i}. \"{corr['original']}\" → "
            f"\"{corr.get('corrected', 'N/A')}\" ({action})"
        )

        with st.expander(title, expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", corr["confidence"])
            with col2:
                st.metric(
                    "Signal Agreement",
                    corr.get("signal_agreement", "N/A").upper(),
                )
            with col3:
                label = {
                    "auto_correct": "AUTO CORRECT",
                    "ask_user": "ASK USER",
                    "skip": "SKIP",
                }.get(action, action)
                st.metric("Action", label)

            st.divider()
            st.write("**💭 Agent Reasoning:**")
            st.info(corr["reasoning"])
            st.divider()

            # Signal 1: Fuzzy
            st.write("**🔤 Signal 1: Fuzzy Matches** (string similarity)")
            if corr.get("fuzzy_matches"):
                st.dataframe(
                    [
                        {
                            "Rank": idx,
                            "Entity": m["name"],
                            "Score": m["score"],
                            "Frequency": m["frequency"],
                        }
                        for idx, m in enumerate(corr["fuzzy_matches"][:5], 1)
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No fuzzy matches found")

            st.divider()

            # Signal 2: RAG
            st.write("**📚 Signal 2: RAG Chunks** (contextually relevant)")
            if corr.get("rag_concepts"):
                st.caption(f"**Extracted concepts:** {corr['rag_concepts']}")

            if corr.get("rag_chunks"):
                for chunk in corr["rag_chunks"]:
                    with st.container():
                        st.caption(
                            f"**Chunk {chunk['rank']}** "
                            f"(Distance: {chunk['distance']:.2f})"
                        )
                        if chunk.get("concepts"):
                            concepts_html = " ".join(
                                f'<span style="background-color: #FFF3E0; padding: 2px 6px; '
                                f'margin: 2px; border-radius: 3px; display: inline-block; '
                                f'font-size: 0.85em;">{c}</span>'
                                for c in chunk["concepts"][:5]
                            )
                            st.markdown(
                                f"Concepts: {concepts_html}", unsafe_allow_html=True
                            )
                        if chunk.get("entities"):
                            entities_html = " ".join(
                                f'<span style="background-color: #E3F2FD; padding: 2px 6px; '
                                f'margin: 2px; border-radius: 3px; display: inline-block; '
                                f'font-size: 0.85em;">{e}</span>'
                                for e in chunk["entities"][:8]
                            )
                            st.markdown(
                                f"Entities: {entities_html}", unsafe_allow_html=True
                            )
                        st.caption(
                            f"From: {chunk['chapter']} — {chunk['section']}"
                        )
                        st.write("")
            else:
                st.caption("No RAG chunks found")

            st.divider()

            # Signal 3: IPA
            st.write("**🔊 Signal 3: IPA Phonetic Matches** (pronunciation similarity)")
            if corr.get("ipa_matches"):
                st.dataframe(
                    [
                        {
                            "Rank": idx,
                            "Entity": m["entity"],
                            "IPA": m["entity_ipa"],
                            "Distance": m["distance"],
                            "Score": m["score"],
                        }
                        for idx, m in enumerate(corr["ipa_matches"][:5], 1)
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No IPA matches found")

            st.divider()
            st.write("**📝 Contexts:**")
            for ctx in corr["contexts"][:3]:
                st.caption(f"• {ctx[:150]}...")


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("🎙️ Transcript Annotator")
st.caption("Multi-signal correction: Fuzzy + RAG + IPA")

# Pre-warm the corrector on startup
with st.spinner("Loading models…"):
    load_corrector()

transcript = st.text_area(
    "Paste transcript here:",
    height=200,
    placeholder="Enter your transcript with potential misspellings…",
)

if st.button("✨ Correct Transcript", type="primary"):
    if not transcript:
        st.warning("Please enter a transcript first!")
    else:
        with st.spinner("Processing with multi-signal analysis…"):
            try:
                corrector = load_corrector()
                if corrector is None:
                    st.error("Corrector failed to initialise. Check logs above.")
                else:
                    result = corrector.correct(transcript, auto_correct_only=False)
                    st.session_state["result"] = result
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if "result" in st.session_state:
    result = st.session_state["result"]

    st.success(f"✅ Processed {result['stats']['entities_found']} entities")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Auto-corrected", result["stats"]["auto_corrected"])
    with col2:
        st.metric("Need confirmation", result["stats"]["asked_user"])
    with col3:
        st.metric("Skipped", result["stats"]["skipped"])
    with col4:
        st.metric("No match", result["stats"]["no_match"])

    # Review section
    ask_user_corrections = [
        c for c in result["corrections"] if c["action"] == "ask_user"
    ]

    if ask_user_corrections:
        st.subheader("⚠️ Review Corrections Needing Confirmation")
        st.write(f"{len(ask_user_corrections)} correction(s) need your review:")

        if "decisions" not in st.session_state:
            st.session_state["decisions"] = {}

        for i, corr in enumerate(ask_user_corrections):
            with st.container():
                col1, col2, col3 = st.columns([5, 1, 1])

                with col1:
                    st.markdown(
                        f"**\"{corr['original']}\"** → **\"{corr.get('corrected', 'N/A')}\"**"
                    )
                    st.caption(f"Context: {corr['contexts'][0][:120]}...")

                    fuzzy_top = (
                        corr["fuzzy_matches"][0]["name"]
                        if corr.get("fuzzy_matches")
                        else "N/A"
                    )
                    rag_has = "✗"
                    if corr.get("rag_chunks") and corr.get("corrected"):
                        for chunk in corr["rag_chunks"]:
                            if any(
                                corr["corrected"].lower() in e.lower()
                                for e in chunk.get("entities", [])
                            ):
                                rag_has = "✓"
                                break
                    ipa_top = (
                        corr["ipa_matches"][0]["entity"]
                        if corr.get("ipa_matches")
                        else "N/A"
                    )

                    st.caption(
                        f"Confidence: {corr['confidence']} | "
                        f"Signal agreement: {corr.get('signal_agreement', 'N/A')} | "
                        f"Fuzzy: {fuzzy_top} | RAG: {rag_has} | IPA: {ipa_top}"
                    )

                with col2:
                    if st.button("✓ Accept", key=f"accept_{i}", use_container_width=True):
                        st.session_state["decisions"][corr["original"]] = "accept"
                        st.rerun()

                with col3:
                    if st.button("✗ Reject", key=f"reject_{i}", use_container_width=True):
                        st.session_state["decisions"][corr["original"]] = "reject"
                        st.rerun()

                if corr["original"] in st.session_state["decisions"]:
                    decision = st.session_state["decisions"][corr["original"]]
                    if decision == "accept":
                        st.success("✓ Accepted — will be corrected")
                    else:
                        st.info("✗ Rejected — will keep original")

                st.divider()

        if st.session_state["decisions"]:
            if st.button("🔄 Apply Decisions", type="primary"):
                for corr in result["corrections"]:
                    if corr["original"] in st.session_state["decisions"]:
                        decision = st.session_state["decisions"][corr["original"]]
                        corr["action"] = "auto_correct" if decision == "accept" else "skip"

                corrected_text = result["original"]
                for corr in result["corrections"]:
                    if corr["action"] == "auto_correct" and corr.get("corrected"):
                        pattern = re.compile(re.escape(corr["original"]), re.IGNORECASE)
                        corrected_text = pattern.sub(corr["corrected"], corrected_text)

                result["corrected"] = corrected_text
                st.session_state["result"] = result
                st.session_state["decisions"] = {}
                st.success("✅ Decisions applied!")
                st.rerun()

    # Side-by-side comparison
    st.subheader("📝 Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original:**")
        st.markdown(
            highlight_text(result["original"], result["corrections"], show_original=True),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("**Corrected:**")
        st.markdown(
            highlight_text(result["corrected"], result["corrections"], show_original=False),
            unsafe_allow_html=True,
        )

    if result["corrections"]:
        show_debug_panel(result["corrections"])

    st.download_button(
        label="📥 Download Results (JSON)",
        data=json.dumps(result, indent=2),
        file_name="correction_results.json",
        mime="application/json",
    )
