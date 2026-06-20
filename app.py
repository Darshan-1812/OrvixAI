import streamlit as st
import time
import re
from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
    claim_extractor_chain,
    query_refiner_chain,
)
from agents import build_verifier_agent
from tools import parse_critic_score

# ── Config ────────────────────────────────────────────────────────────────────
MIN_SCORE      = 7.0
MAX_ITERATIONS = 3
MAX_CLAIMS     = 5

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchMind · AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #e8e4dc;
}
.stApp {
    background: #0a0a0f;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(255,140,50,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(255,80,30,0.08) 0%, transparent 55%);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

.hero { text-align: center; padding: 3.5rem 0 2.5rem; }
.hero-eyebrow {
    font-family: 'DM Mono', monospace; font-size: 0.7rem; font-weight: 500;
    letter-spacing: 0.25em; text-transform: uppercase; color: #ff8c32;
    margin-bottom: 1rem; opacity: 0.9;
}
.hero h1 {
    font-family: 'Syne', sans-serif; font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 800; line-height: 1.0; letter-spacing: -0.03em;
    color: #f0ebe0; margin: 0 0 1rem;
}
.hero h1 span { color: #ff8c32; }
.hero-sub {
    font-size: 1.05rem; font-weight: 300; color: #a09890;
    max-width: 520px; margin: 0 auto; line-height: 1.65;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,140,50,0.3), transparent);
    margin: 2rem 0;
}
.input-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,140,50,0.15);
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 2rem;
    backdrop-filter: blur(8px);
}
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,140,50,0.25) !important;
    border-radius: 10px !important; color: #f0ebe0 !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 1rem !important;
    padding: 0.75rem 1rem !important; transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #ff8c32 !important;
    box-shadow: 0 0 0 3px rgba(255,140,50,0.12) !important;
}
.stTextInput > label {
    font-family: 'DM Mono', monospace !important; font-size: 0.72rem !important;
    letter-spacing: 0.15em !important; text-transform: uppercase !important;
    color: #ff8c32 !important; font-weight: 500 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #ff8c32 0%, #ff5a1a 100%) !important;
    color: #0a0a0f !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 0.95rem !important;
    letter-spacing: 0.04em !important; border: none !important;
    border-radius: 10px !important; padding: 0.7rem 2.2rem !important;
    cursor: pointer !important; transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s !important;
    box-shadow: 0 4px 20px rgba(255,140,50,0.3) !important; width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(255,140,50,0.4) !important; opacity: 0.95 !important;
}

/* ── Step cards ── */
.step-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 1.5rem 1.8rem; margin-bottom: 1.2rem;
    position: relative; overflow: hidden; transition: border-color 0.3s;
}
.step-card.active  { border-color: rgba(255,140,50,0.4); background: rgba(255,140,50,0.04); }
.step-card.done    { border-color: rgba(80,200,120,0.3);  background: rgba(80,200,120,0.03); }
.step-card.healing { border-color: rgba(180,100,255,0.4); background: rgba(180,100,255,0.04); }
.step-card::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    border-radius: 14px 0 0 14px; background: rgba(255,255,255,0.05); transition: background 0.3s;
}
.step-card.active::before  { background: #ff8c32; }
.step-card.done::before    { background: #50c878; }
.step-card.healing::before { background: #b464ff; }

.step-header { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.3rem; }
.step-num    { font-family: 'DM Mono', monospace; font-size: 0.68rem; font-weight: 500; letter-spacing: 0.15em; color: #ff8c32; opacity: 0.7; }
.step-title  { font-family: 'Syne', sans-serif; font-size: 0.95rem; font-weight: 700; color: #f0ebe0; }
.step-status { margin-left: auto; font-family: 'DM Mono', monospace; font-size: 0.68rem; letter-spacing: 0.1em; }
.status-waiting  { color: #555; }
.status-running  { color: #ff8c32; }
.status-done     { color: #50c878; }
.status-healing  { color: #b464ff; }
.status-skipped  { color: #444; }

/* ── Healing badge ── */
.heal-badge {
    display: inline-block;
    background: rgba(180,100,255,0.15);
    border: 1px solid rgba(180,100,255,0.3);
    border-radius: 6px;
    padding: 0.2rem 0.65rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    color: #b464ff;
    margin-left: 0.5rem;
}

/* ── Score pill ── */
.score-pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 999px; padding: 0.3rem 0.9rem;
    font-family: 'DM Mono', monospace; font-size: 0.8rem;
}
.score-pill.low  { border-color: rgba(255,80,30,0.4);  color: #ff5a1a; }
.score-pill.mid  { border-color: rgba(255,200,0,0.4);  color: #ffc800; }
.score-pill.high { border-color: rgba(80,200,120,0.4); color: #50c878; }

/* ── Iteration log ── */
.iter-log {
    background: rgba(180,100,255,0.05);
    border: 1px solid rgba(180,100,255,0.15);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-top: 0.8rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #b4a0c8;
    line-height: 1.8;
}
.iter-log-title {
    color: #b464ff; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; font-size: 0.68rem; margin-bottom: 0.5rem;
}

/* ── Claim verification table ── */
.claim-table { width: 100%; border-collapse: collapse; margin-top: 0.8rem; }
.claim-table th {
    font-family: 'DM Mono', monospace; font-size: 0.65rem; letter-spacing: 0.15em;
    text-transform: uppercase; color: #605850; padding: 0.5rem 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.07); text-align: left;
}
.claim-table td {
    font-size: 0.85rem; color: #cdc8bf; padding: 0.75rem 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.04); vertical-align: top;
    line-height: 1.5;
}
.claim-table tr:last-child td { border-bottom: none; }
.verdict-verified    { color: #50c878; font-family: 'DM Mono', monospace; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; }
.verdict-unverified  { color: #ffc800; font-family: 'DM Mono', monospace; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; }
.verdict-contradicted{ color: #ff5a1a; font-family: 'DM Mono', monospace; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; }

/* ── Summary bar ── */
.summary-bar {
    display: flex; gap: 1rem; flex-wrap: wrap;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1.5rem;
}
.summary-item { display: flex; flex-direction: column; gap: 0.2rem; }
.summary-label { font-family: 'DM Mono', monospace; font-size: 0.62rem; letter-spacing: 0.15em; text-transform: uppercase; color: #555; }
.summary-value { font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 700; }
.summary-sep { width: 1px; background: rgba(255,255,255,0.07); margin: 0 0.5rem; }

/* Result panels */
.result-panel {
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 1.8rem 2rem; margin-top: 1rem; margin-bottom: 1.5rem;
}
.result-panel-title {
    font-family: 'DM Mono', monospace; font-size: 0.7rem; font-weight: 500;
    letter-spacing: 0.2em; text-transform: uppercase; color: #ff8c32;
    margin-bottom: 1rem; padding-bottom: 0.7rem;
    border-bottom: 1px solid rgba(255,140,50,0.15);
}
.result-content {
    font-size: 0.92rem; line-height: 1.8; color: #cdc8bf;
    white-space: pre-wrap; font-family: 'DM Sans', sans-serif;
}
.report-panel {
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,140,50,0.2);
    border-radius: 16px; padding: 2rem 2.5rem; margin-top: 1rem;
}
.feedback-panel {
    background: rgba(255,255,255,0.025); border: 1px solid rgba(80,200,120,0.2);
    border-radius: 16px; padding: 2rem 2.5rem; margin-top: 1rem;
}
.verify-panel {
    background: rgba(255,255,255,0.025); border: 1px solid rgba(100,160,255,0.2);
    border-radius: 16px; padding: 2rem 2.5rem; margin-top: 1rem;
}
.panel-label {
    font-family: 'DM Mono', monospace; font-size: 0.7rem; letter-spacing: 0.2em;
    text-transform: uppercase; margin-bottom: 1.2rem; padding-bottom: 0.7rem;
}
.panel-label.orange { color: #ff8c32; border-bottom: 1px solid rgba(255,140,50,0.15); }
.panel-label.green  { color: #50c878; border-bottom: 1px solid rgba(80,200,120,0.15); }
.panel-label.blue   { color: #64a0ff; border-bottom: 1px solid rgba(100,160,255,0.15); }

.section-heading {
    font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 700;
    color: #f0ebe0; margin: 2rem 0 1rem;
}
.notice {
    font-family: 'DM Mono', monospace; font-size: 0.72rem;
    color: #605850; text-align: center; margin-top: 3rem; letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def step_card(num: str, title: str, state: str, desc: str = "", badge: str = ""):
    status_map = {
        "waiting": ("WAITING",    "status-waiting"),
        "running": ("● RUNNING",  "status-running"),
        "done":    ("✓ DONE",     "status-done"),
        "healing": ("↺ HEALING",  "status-healing"),
        "skipped": ("— SKIPPED",  "status-skipped"),
    }
    label, cls   = status_map.get(state, ("", ""))
    card_cls_map = {"running": "active", "done": "done", "healing": "healing"}
    card_cls     = card_cls_map.get(state, "")
    badge_html   = f'<span class="heal-badge">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="step-card {card_cls}">
        <div class="step-header">
            <span class="step-num">{num}</span>
            <span class="step-title">{title}</span>{badge_html}
            <span class="step-status {cls}">{label}</span>
        </div>
        {"<div style='font-size:0.82rem;color:#706860;margin-top:0.3rem;'>"+desc+"</div>" if desc else ""}
    </div>
    """, unsafe_allow_html=True)


def score_pill(score: float) -> str:
    cls = "high" if score >= MIN_SCORE else ("mid" if score >= 5 else "low")
    return f'<span class="score-pill {cls}">Score {score:.1f}/10</span>'


def parse_claims_from_text(raw: str) -> list[str]:
    lines = raw.strip().splitlines()
    claims = []
    for line in lines:
        m = re.match(r"^\d+\.\s+(.+)$", line.strip())
        if m:
            claims.append(m.group(1).strip())
    return claims[:MAX_CLAIMS]


# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "results":    {},
    "running":    False,
    "done":       False,
    "iter_log":   [],       # list of {iteration, score, query} dicts
    "final_score": 0.0,
    "iterations_taken": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Multi-Agent AI System</div>
    <h1>Research<span>Mind</span></h1>
    <p class="hero-sub">
        Autonomous agents search, scrape, write, self-heal, and fact-check —
        delivering a verified research report on any topic.
    </p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ── Layout ────────────────────────────────────────────────────────────────────
col_input, col_spacer, col_pipeline = st.columns([5, 0.5, 4])

with col_input:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    topic = st.text_input(
        "Research Topic",
        placeholder="e.g. Quantum computing breakthroughs in 2025",
        key="topic_input",
    )
    run_btn = st.button("⚡  Run Research Pipeline", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1.5rem;align-items:center;">
        <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#605850;letter-spacing:0.1em;">TRY →</span>
    """, unsafe_allow_html=True)
    for ex in ["LLM agents 2025", "CRISPR gene editing", "Fusion energy progress"]:
        st.markdown(f"""
        <span style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
            border-radius:6px;padding:0.25rem 0.7rem;font-size:0.75rem;color:#a09890;
            font-family:'DM Sans',sans-serif;">{ex}</span>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_pipeline:
    st.markdown('<div class="section-heading">Pipeline</div>', unsafe_allow_html=True)

    r    = st.session_state.results
    ilog = st.session_state.iter_log

    def step_state(key):
        if not r:
            return "waiting"
        if key in r:
            return "done"
        if st.session_state.running:
            order = ["search", "reader", "writer", "critic", "healing", "verify"]
            for k in order:
                if k not in r:
                    return "running" if k == key else "waiting"
        return "waiting"

    healing_count = len(ilog)
    heal_badge    = f"×{healing_count} healed" if healing_count > 0 else ""

    step_card("01", "Search Agent",    step_state("search"),  "Gathers recent web information")
    step_card("02", "Reader Agent",    step_state("reader"),  "Scrapes & extracts deep content")
    step_card("03", "Writer + Critic", step_state("critic"),  "Drafts report & scores quality")
    step_card("04", "Self-Heal Loop",  step_state("healing"), "Re-searches if score < 7/10", badge=heal_badge)
    step_card("05", "Claim Verifier",  step_state("verify"),  "Fact-checks key claims")

    # Show iteration log if healing happened
    if ilog:
        log_lines = "".join(
            f'<div>↺ Iter {entry["iteration"]} — Score {entry["score"]:.1f}/10 '
            f'— Query: <em>{entry["query"][:60]}…</em></div>'
            for entry in ilog
        )
        st.markdown(f"""
        <div class="iter-log">
            <div class="iter-log-title">Self-Heal Log</div>
            {log_lines}
        </div>
        """, unsafe_allow_html=True)


# ── Trigger ───────────────────────────────────────────────────────────────────
if run_btn:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        st.session_state.results          = {}
        st.session_state.running          = True
        st.session_state.done             = False
        st.session_state.iter_log         = []
        st.session_state.final_score      = 0.0
        st.session_state.iterations_taken = 0
        st.rerun()


# ── Pipeline execution ────────────────────────────────────────────────────────
if st.session_state.running and not st.session_state.done:
    results   = {}
    topic_val = st.session_state.topic_input
    iter_log  = []

    def _extract_text(content) -> str:
        """Agent message .content can be a str or a list of content blocks."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block["text"])
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)
        return str(content)

    def run_search_and_scrape(query: str) -> dict:
        sa = build_search_agent()
        sr = sa.invoke({"messages": [("user", f"Find recent, reliable and detailed information about: {query}")]})
        search_text = _extract_text(sr["messages"][-1].content)

        ra = build_reader_agent()
        rr = ra.invoke({"messages": [(
            "user",
            f"Based on the following search results about '{topic_val}', "
            f"pick the most relevant URL and scrape it.\n\nSearch Results:\n{search_text[:800]}"
        )]})
        return {"search": search_text, "reader": _extract_text(rr["messages"][-1].content)}

    def run_writer_critic(search_text: str, reader_text: str) -> dict:
        combined = f"SEARCH RESULTS:\n{search_text}\n\nDETAILED SCRAPED CONTENT:\n{reader_text}"
        report   = writer_chain.invoke({"topic": topic_val, "research": combined})
        feedback = critic_chain.invoke({"report": report})
        score    = parse_critic_score(feedback)
        return {"writer": report, "critic": feedback, "score": score}

    # ── Steps 1 & 2: Search + Reader ──
    with st.spinner("🔍  Search & Reader agents gathering research…"):
        data = run_search_and_scrape(topic_val)
        results["search"] = data["search"]
        results["reader"] = data["reader"]
        st.session_state.results = dict(results)

    # ── Steps 3: Writer + Critic ──
    with st.spinner("✍️  Writer drafting | Critic evaluating…"):
        wr = run_writer_critic(results["search"], results["reader"])
        results.update(wr)
        st.session_state.results = dict(results)

    # ── Step 4: Self-Healing Loop ──
    iteration = 1
    while results["score"] < MIN_SCORE and iteration < MAX_ITERATIONS:
        iteration += 1
        with st.spinner(f"↺  Self-Healing — iteration {iteration}/{MAX_ITERATIONS} (score was {results['score']:.1f}/10)…"):
            refined_query = query_refiner_chain.invoke({
                "topic":    topic_val,
                "feedback": results["critic"],
            }).strip()

            iter_log.append({
                "iteration": iteration,
                "score":     results["score"],
                "query":     refined_query,
            })
            st.session_state.iter_log = list(iter_log)

            new_data = run_search_and_scrape(refined_query)

            # Merge new research into existing
            merged_search = results["search"] + "\n\n--- REFINED SEARCH ---\n" + new_data["search"]
            merged_reader = results["reader"] + "\n\n--- REFINED SCRAPE ---\n"  + new_data["reader"]

            wr = run_writer_critic(merged_search, merged_reader)
            results["search"] = merged_search
            results["reader"] = merged_reader
            results.update(wr)
            st.session_state.results = dict(results)

    results["healing"] = True   # marks step as done in the pipeline sidebar
    st.session_state.results          = dict(results)
    st.session_state.final_score      = results["score"]
    st.session_state.iterations_taken = iteration

    # ── Step 5: Claim Verification ──
    with st.spinner("🔎  Claim Verifier fact-checking the report…"):
        raw_claims    = claim_extractor_chain.invoke({"report": results["writer"]})
        claims        = parse_claims_from_text(raw_claims)
        verifier      = build_verifier_agent()
        verified_list = []

        for claim in claims:
            vr = verifier.invoke({"messages": [(
                "user",
                f"Use the verify_claim tool to check this claim, then give a final verdict.\n\n"
                f"Claim: {claim}\n\n"
                f"After seeing the evidence, respond in this exact format:\n"
                f"VERDICT: <VERIFIED|UNVERIFIED|CONTRADICTED>\n"
                f"REASON: <one sentence explaining why>"
            )]})
            out = _extract_text(vr["messages"][-1].content)
            vm  = re.search(r"VERDICT:\s*(VERIFIED|UNVERIFIED|CONTRADICTED)", out, re.IGNORECASE)
            rm  = re.search(r"REASON:\s*(.+)", out)
            verified_list.append({
                "claim":   claim,
                "verdict": vm.group(1).upper() if vm else "UNVERIFIED",
                "reason":  rm.group(1).strip()  if rm else "Could not determine.",
            })

        results["verify"] = verified_list
        st.session_state.results = dict(results)

    st.session_state.running = False
    st.session_state.done    = True
    st.rerun()


# ── Results display ───────────────────────────────────────────────────────────
r = st.session_state.results

if r and st.session_state.done:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Results</div>', unsafe_allow_html=True)

    # ── Summary bar ──
    score     = st.session_state.final_score
    iters     = st.session_state.iterations_taken
    v_list    = r.get("verify", [])
    n_verified = sum(1 for c in v_list if c["verdict"] == "VERIFIED")
    n_total    = len(v_list)
    score_cls  = "high" if score >= MIN_SCORE else ("mid" if score >= 5 else "low")

    st.markdown(f"""
    <div class="summary-bar">
        <div class="summary-item">
            <span class="summary-label">Report Quality</span>
            <span class="summary-value" style="color:{'#50c878' if score >= MIN_SCORE else '#ffc800'};">{score:.1f}/10</span>
        </div>
        <div class="summary-sep"></div>
        <div class="summary-item">
            <span class="summary-label">Iterations</span>
            <span class="summary-value" style="color:#b464ff;">{iters}</span>
        </div>
        <div class="summary-sep"></div>
        <div class="summary-item">
            <span class="summary-label">Claims Verified</span>
            <span class="summary-value" style="color:#64a0ff;">{n_verified}/{n_total}</span>
        </div>
        <div class="summary-sep"></div>
        <div class="summary-item">
            <span class="summary-label">Self-Healed</span>
            <span class="summary-value" style="color:#b464ff;">{"Yes" if iters > 1 else "No"}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Raw outputs ──
    if "search" in r:
        with st.expander("🔍 Search Results (raw)", expanded=False):
            st.markdown(f'<div class="result-panel"><div class="result-panel-title">Search Agent Output</div>'
                        f'<div class="result-content">{r["search"]}</div></div>', unsafe_allow_html=True)
    if "reader" in r:
        with st.expander("📄 Scraped Content (raw)", expanded=False):
            st.markdown(f'<div class="result-panel"><div class="result-panel-title">Reader Agent Output</div>'
                        f'<div class="result-content">{r["reader"]}</div></div>', unsafe_allow_html=True)

    # ── Self-Heal log ──
    if st.session_state.iter_log:
        with st.expander("↺ Self-Heal Iterations", expanded=False):
            for entry in st.session_state.iter_log:
                st.markdown(f"""
                <div class="iter-log">
                    <div class="iter-log-title">Iteration {entry['iteration']}</div>
                    <div>Previous score: {entry['score']:.1f}/10</div>
                    <div>Refined query: {entry['query']}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Final report ──
    if "writer" in r:
        st.markdown('<div class="report-panel"><div class="panel-label orange">📝 Final Research Report</div>', unsafe_allow_html=True)
        st.markdown(r["writer"])
        st.markdown("</div>", unsafe_allow_html=True)
        st.download_button(
            label="⬇  Download Report (.md)",
            data=r["writer"],
            file_name=f"research_report_{int(time.time())}.md",
            mime="text/markdown",
        )

    # ── Critic feedback ──
    if "critic" in r:
        st.markdown('<div class="feedback-panel"><div class="panel-label green">🧐 Critic Feedback</div>', unsafe_allow_html=True)
        st.markdown(r["critic"])
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Claim verification table ──
    if v_list:
        verdict_css = {
            "VERIFIED":     "verdict-verified",
            "UNVERIFIED":   "verdict-unverified",
            "CONTRADICTED": "verdict-contradicted",
        }
        verdict_icon = {"VERIFIED": "✓", "UNVERIFIED": "?", "CONTRADICTED": "✗"}
        rows = "".join(
            f"""<tr>
                <td style="width:44%;">{c['claim']}</td>
                <td style="width:14%;"><span class="{verdict_css.get(c['verdict'], '')}">{verdict_icon.get(c['verdict'], '')} {c['verdict']}</span></td>
                <td>{c['reason']}</td>
            </tr>"""
            for c in v_list
        )
        st.markdown(f"""
        <div class="verify-panel">
            <div class="panel-label blue">🔎 Claim Verification</div>
            <table class="claim-table">
                <thead><tr><th>Claim</th><th>Verdict</th><th>Reason</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="notice">
    ResearchMind · Self-Healing Multi-Agent Pipeline · Built with LangChain + Streamlit
</div>
""", unsafe_allow_html=True)