"""Lens OS — Streamlit web inspector.

A local dev tool for testing the pipeline without the CLI.
Upload an image, enter optional GPS coordinates, and inspect every
agent's output alongside the final response card.

Run with:
    streamlit run src/inspector.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Async bridge — Streamlit is sync; the pipeline is async
# ---------------------------------------------------------------------------


def _run(coro):
    """Execute an async coroutine in a dedicated event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_pipeline_sync(image_bytes: bytes, lat: float | None, lng: float | None, user_id: str):
    """Save image to a temp file and run the full pipeline."""
    from src import db
    from src.cache import init_cache
    from src.contracts import LensInput
    from src.orchestrator import run_pipeline

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(image_bytes)
        tmp_path = f.name

    try:
        db.init_db()
        init_cache(db.DB_PATH)
        inp = LensInput(
            image_path=tmp_path,
            lat=lat,
            lng=lng,
            user_id=user_id,
        )
        return _run(run_pipeline(inp))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _confidence_badge(level: str) -> str:
    colours = {
        "certain": "🟢",
        "fairly_sure": "🟡",
        "uncertain": "🟠",
        "guessing": "🔴",
    }
    return f"{colours.get(level, '⚪')} {level}"


def _show_vision(vision) -> None:
    if vision is None:
        st.warning("Vision agent returned nothing.")
        return
    st.markdown(f"**{vision.entity_name}** — `{vision.entity_type}`")
    st.markdown(f"Confidence: {_confidence_badge(vision.confidence_level)}")
    if vision.needs_fallback:
        st.error("Needs fallback")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Evidence**")
        for e in vision.evidence:
            st.markdown(f"- {e}")
    with col2:
        st.markdown("**Failure modes checked**")
        for f in vision.failure_modes_checked:
            st.markdown(f"- {f}")

    if vision.alternatives:
        st.markdown("**Alternatives considered**")
        for a in vision.alternatives:
            st.markdown(f"- {a}")


def _show_memory(memory) -> None:
    if memory is None:
        st.info("Memory agent returned nothing.")
        return
    st.markdown(f"User: `{memory.user_id}`")

    # Layer 1 — in-context recent window
    st.markdown("#### Layer 1 — In-context recent scans")
    if memory.recent_context:
        st.info(memory.recent_context)
    else:
        st.caption("No recent scans in this session (L1 resets on server restart).")

    # Layer 2 — vector hits
    st.markdown("#### Layer 2 — Hybrid retrieval hits (BM25 + semantic)")
    if memory.user_interests_snapshot:
        st.markdown("**Interest snapshot**")
        for k, v in sorted(memory.user_interests_snapshot.items(), key=lambda x: -x[1])[:5]:
            st.markdown(f"- `{k}`: {v:.2f}")
    if memory.hits:
        st.markdown(f"**{len(memory.hits)} hit(s) above 0.75 similarity**")
        for h in memory.hits:
            st.markdown(
                f"- **{h.subject_name}** (semantic {h.similarity_score:.2f}): {h.summary[:80]}…"
            )
    else:
        st.markdown("No prior interactions found for this user.")

    # Layer 3 — persisted entity facts
    st.markdown("#### Layer 3 — Persisted entity facts")
    if memory.entity_facts:
        st.markdown(f"**{len(memory.entity_facts)} stored fact(s)**")
        for f in memory.entity_facts[:8]:
            tag = f.get("as_of", "")
            source = f.get("source", "")
            label = f"*{f['fact_key']}*" if f.get("fact_key") else ""
            meta = " · ".join(filter(None, [source, f"as of {tag}" if tag else ""]))
            st.markdown(f"- {label} {f['fact_value']} *(  {meta})*")
    else:
        st.caption("No stored facts for this entity yet — will populate after first full scan.")


def _show_search(search) -> None:
    if search is None:
        st.info("Search was skipped (confidence gate) or failed.")
        return
    st.markdown(f"**Plan:** {search.research_plan}")
    if search.tool_calls:
        st.markdown(f"**Tool calls ({len(search.tool_calls)}/3 budget)**")
        for tc in search.tool_calls:
            with st.expander(f"`{tc.tool}`"):
                st.json(tc.input)
                st.markdown(f"*{tc.justification}*")
                st.markdown(f"→ {tc.observation}")
    if search.historical_facts:
        st.markdown("**Historical facts**")
        for f in search.historical_facts:
            st.markdown(f"- {f.fact} *(source: {f.source})*")
    if search.live_facts:
        st.markdown("**Live facts**")
        for f in search.live_facts:
            st.markdown(f"- {f.fact} *(as of {f.as_of}, source: {f.source})*")
    elif search.live_facts_skipped_reason:
        st.markdown(f"*Live facts skipped: {search.live_facts_skipped_reason}*")
    if search.identification_concerns:
        st.warning("Identification concerns: " + "; ".join(search.identification_concerns))
    if search.nearby_context:
        st.markdown(f"**Nearby:** {search.nearby_context}")


def _show_card(card, session_id: str | None = None, from_cache: bool = False) -> None:
    card_dict = card.model_dump()
    if from_cache:
        st.caption("⚡ Served from cache — agents skipped.")
    if card_dict.get("card_type") == "fallback":
        st.error(f"**{card.headline}**")
        if card.observation:
            st.markdown(card.observation)
        st.info(f"Suggestion: {card.suggestion}")
    else:
        st.success(f"**{card.headline}**")
        st.markdown(card.body)
        if card.personalized_hooks:
            st.markdown("**Personalized picks**")
            for h in card.personalized_hooks:
                st.markdown(f"- {h.fact}")
        if card.citations:
            st.markdown(
                "**Sources:** "
                + " · ".join(
                    f"[{c.source_name}]({c.url})" if c.url else c.source_name
                    for c in card.citations
                )
            )
        st.markdown(
            f"Confidence: `{card.confidence_displayed}` · "
            f"Vision: {'✓' if card.source_mix.used_vision else '–'} · "
            f"Memory: {'✓' if card.source_mix.used_memory else '–'} · "
            f"Search: {'✓' if card.source_mix.used_search else '–'}"
        )

    st.markdown(f"**Cost:** ${card.cost_usd_total:.5f} · **Latency:** {card.latency_ms} ms")
    if session_id:
        st.code(session_id, language=None)
        st.caption("↑ Session ID — pass as `scan_session_id` to VoiceOS for voice follow-up")


def _show_cost_log(cost_log) -> None:
    if not cost_log:
        st.info("No cost entries recorded.")
        return
    rows = [
        {
            "Agent": e.agent,
            "Model": e.model,
            "In tokens": e.input_tokens,
            "Out tokens": e.output_tokens,
            "Cost USD": f"${e.cost_usd:.6f}",
        }
        for e in cost_log
    ]
    st.table(rows)
    total = sum(e.cost_usd for e in cost_log)
    st.markdown(f"**Total: ${total:.5f}**")


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Lens OS Inspector", page_icon="🔍", layout="wide")
    st.title("🔍 Lens OS — Pipeline Inspector")
    st.caption("Phase 0 dev tool — upload an image and inspect every agent's output.")

    if not os.environ.get("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY not set. Add it to your .env file and restart.")
        st.stop()

    # Sidebar — inputs
    with st.sidebar:
        st.header("Input")
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])
        lat = st.number_input("Latitude", value=12.9716, format="%.6f")
        lng = st.number_input("Longitude", value=77.5946, format="%.6f")
        use_location = st.checkbox("Include GPS in request", value=True)
        user_id = st.text_input("User ID", value="inspector-user")
        run_btn = st.button("▶ Run pipeline", type="primary", disabled=uploaded is None)

    if uploaded is None:
        st.info("Upload an image in the sidebar to get started.")
        return

    col_img, col_card = st.columns([1, 2])
    with col_img:
        st.image(uploaded, caption=uploaded.name, use_container_width=True)

    if not run_btn:
        return

    image_bytes = uploaded.read()
    lat_val = lat if use_location else None
    lng_val = lng if use_location else None

    with st.spinner("Running pipeline…"):
        try:
            state = run_pipeline_sync(image_bytes, lat_val, lng_val, user_id)
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            return

    card = state.get("response_card")
    vision = state.get("vision_result")
    memory = state.get("memory_result")
    search = state.get("search_result")
    reasoning = state.get("reasoning_trace")
    cost_log = state.get("cost_log", [])
    errors = state.get("errors", [])

    # Detect cache hit: vision ran iff vision_result has real evidence (not the cache placeholder)
    from_cache = vision is not None and vision.evidence == ["(restored from cache)"]

    # Generate a session ID (mirrors server._store_session logic)
    session_id: str | None = None
    if vision is not None and card is not None:
        from src.contracts import NormalCard as _NCard
        from src.session_store import create_session

        card_body = card.body if isinstance(card, _NCard) else card.observation
        ctx = create_session(
            entity_name=vision.entity_name,
            entity_type=vision.entity_type,
            confidence_level=vision.confidence_level,
            card_headline=card.headline,
            card_body=card_body,
            historical_facts=search.historical_facts if search else [],
            live_facts=search.live_facts if search else [],
            nearby_context=search.nearby_context if search else "",
            user_id=user_id,
            image_b64=state.get("image_b64", ""),
        )
        session_id = ctx.session_id

    # Card
    with col_card:
        st.subheader("Response card")
        if card:
            _show_card(card, session_id=session_id, from_cache=from_cache)
        else:
            st.error("No card produced.")
        if errors:
            with st.expander("Pipeline errors"):
                for e in errors:
                    st.markdown(f"- `{e}`")

    st.divider()

    # Agent outputs
    tab_vision, tab_memory, tab_reasoning, tab_search, tab_cost, tab_raw = st.tabs(
        ["👁 Vision", "🧠 Memory", "🤔 Reasoning", "🔎 Search", "💰 Cost", "📄 Raw JSON"]
    )

    with tab_vision:
        _show_vision(vision)

    with tab_memory:
        _show_memory(memory)

    with tab_reasoning:
        if reasoning is None:
            st.info("Reasoning step was skipped (cache hit or Vision failed).")
        else:
            st.markdown(f"**What we know:** {reasoning.what_we_know}")
            if reasoning.memory_context:
                st.markdown(f"**Memory context:** {reasoning.memory_context}")
            if reasoning.key_unknowns:
                st.markdown("**Key unknowns**")
                for u in reasoning.key_unknowns:
                    st.markdown(f"- {u}")
            st.markdown(f"**Research brief:** {reasoning.research_brief}")
            if reasoning.suggested_tool_priority:
                st.markdown(
                    "**Suggested tool order:** "
                    + " → ".join(f"`{t}`" for t in reasoning.suggested_tool_priority)
                )

    with tab_search:
        _show_search(search)

    with tab_cost:
        _show_cost_log(cost_log)

    with tab_raw:
        st.json(
            {
                "session_id": session_id,
                "from_cache": from_cache,
                "card": card.model_dump() if card else None,
                "vision": vision.model_dump() if vision else None,
                "memory": memory.model_dump(mode="json") if memory else None,
                "reasoning": reasoning.model_dump() if reasoning else None,
                "search": search.model_dump() if search else None,
            }
        )


if __name__ == "__main__":
    main()
