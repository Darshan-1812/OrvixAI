# ResearchMind

A self-healing multi-agent research pipeline built with LangChain, LangGraph, and Streamlit. Given a topic, the system searches the web, scrapes sources, drafts a structured report, critiques its own output, automatically retries if quality is insufficient, and then fact-checks every major claim before delivery.

---

## Architecture

```
Stage 1          Stage 2          Stage 3             Stage 4              Stage 5
-----------      -----------      ---------------     -----------------    ---------------
Search Agent --> Reader Agent --> Writer + Critic --> Self-Heal Loop   --> Claim Verifier
web_search       scrape_url       writer_chain        query_refiner        claim_extractor
                                  critic_chain        web_search           verify_claim
                                                      |
                                                      | score < 7.0
                                                      | (max 3 iterations)
                                                      v
                                                 back to Stage 1
                                                 with refined query

Output: Research Report | Critic Feedback | Heal Log | Claim Verdicts
```

### Stage 1 — Search Agent

A LangGraph ReAct agent equipped with the `web_search` tool (Tavily). Receives the user topic and returns titles, URLs, and snippets from the top 5 results.

### Stage 2 — Reader Agent

A second ReAct agent equipped with `scrape_url`. Reads the Stage 1 output, selects the most relevant URL, fetches the page, strips navigation and scripts, and returns up to 3000 characters of clean text.

### Stage 3 — Writer and Critic Chains

Two sequential LLM chains (no tools needed):

`writer_chain` receives the merged search and scraped content and produces a structured Markdown report: Introduction, Key Findings (at least 3 points), Conclusion, and Sources.

`critic_chain` receives that report and returns a numeric score out of 10, a strengths list, an areas-to-improve list, and a one-line verdict. The score is parsed by `parse_critic_score()` and used to gate Stage 4.

### Stage 4 — Self-Healing Loop

If the critic score is below 7.0, the `query_refiner_chain` reads the feedback and the original topic and generates a sharper search query targeting the identified gaps. The pipeline re-runs Stages 1 and 2 with the refined query, merges the new research into the existing context, and re-runs Stage 3. This repeats until the score reaches 7.0 or 3 iterations are exhausted, whichever comes first.

### Stage 5 — Claim Verifier

`claim_extractor_chain` pulls the 5 most specific, verifiable factual claims from the final report. A ReAct verifier agent then calls the `verify_claim` tool for each claim, which runs a fresh Tavily search and returns evidence snippets. The agent evaluates the evidence and returns one of three verdicts: VERIFIED, UNVERIFIED, or CONTRADICTED, with a one-sentence reason.

---

## Project Structure

```
.
├── agents.py          LLM agents, writer/critic/extractor/refiner chains
├── app.py             Streamlit UI
├── pipeline.py        CLI entry point with Rich terminal output
├── tools.py           LangChain tools and parse_critic_score utility
├── requirements.txt   Python dependencies
└── .env               API keys (not committed)
```

---

## Setup

### Prerequisites

- Python 3.10 or later
- A Mistral API key
- A Tavily API key

### Installation

```bash
git clone https://github.com/your-username/researchmind.git
cd researchmind
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
MISTRAL_API_KEY=your_mistral_key_here
TAVILY_API_KEY=your_tavily_key_here
```

### Running the UI

```bash
streamlit run app.py
```

### Running the CLI

```bash
python pipeline.py
```

---

## Requirements

```
langchain>=0.2.0
langchain-core>=0.2.0
langchain-community>=0.2.0
langchain-mistralai>=0.1.0
langgraph>=0.1.0
tavily-python>=0.3.0
beautifulsoup4>=4.12.0
requests>=2.31.0
python-dotenv>=1.0.0
rich>=13.7.0
streamlit>=1.28.0
pydantic>=2.5.0
```

---

## Configuration

The following constants in `pipeline.py` and `app.py` control pipeline behavior:

| Constant | Default | Description |
|---|---|---|
| `MIN_SCORE` | `7.0` | Critic score required to skip self-healing |
| `MAX_ITERATIONS` | `3` | Maximum self-healing retries |
| `MAX_CLAIMS` | `5` | Number of claims sent to the verifier |

---

## Key Design Decisions

**Why merge research across iterations rather than replace it?**
Each healing iteration appends new search results to the existing context rather than starting fresh. This gives the writer cumulative evidence across multiple search angles, producing more comprehensive reports than a single retry would.

**Why a separate claim extractor chain instead of prompting the writer?**
Separating extraction from writing keeps each component focused on one task. The writer optimises for prose quality; the extractor optimises for identifying specific, falsifiable claims. Mixing both into one prompt degrades both outputs.

**Why ReAct agents for search and verification but plain chains for writing and critiquing?**
Writing and critiquing are pure text transformation tasks that benefit from a single, focused prompt. Tool-calling adds latency and unpredictability with no benefit when the task does not require external information. The agents use tools only where external data is genuinely required.

**Why cap at 3 healing iterations?**
Beyond 3 iterations, the marginal quality gain from additional searches is typically small relative to the added latency and API cost. The cap also prevents runaway loops on topics where the critic is structurally hard to satisfy.

---

## Limitations

- Scraping is blocked on pages that require JavaScript rendering or authentication.
- Claim verification is limited by Tavily search quality. Obscure or very recent claims may return insufficient evidence and default to UNVERIFIED.
- The self-healing loop adds latency proportional to the number of iterations. A topic requiring 3 full iterations takes roughly 3 times as long as a topic that passes on the first attempt.
- `mistral-small-2506` is used for all chains and agents. Larger models will produce higher-quality reports and more accurate claim verdicts at higher cost.

---

## License

MIT