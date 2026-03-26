# RSS AI Agent

An agentic AI system built with **PydanticAI** that continuously discovers, verifies, and maintains a deduplicated database of AI/ML RSS feeds.

---

## Architecture

```
rss-agent/
├── agent.py        # PydanticAI agent with two MCP tools registered
├── mcp_tools.py    # MCP tool implementations (search + availability check)
├── mcp_server.py   # Standalone MCP server (stdio) for external MCP clients
├── models.py       # Pydantic data models (RSSLink, EvalCase, EvalStats …)
├── database.py     # Thread-safe JSON persistence for unique RSS links
├── eval.py         # Evaluation framework (EvalRunner, scoring, stats)
├── config.py       # All tuneable parameters, seed feeds, eval cases
├── run.py          # CLI entry point
└── requirements.txt
```

---

## MCP Tools

| Tool                      | Description                                                                                                     |
|---------------------------|-----------------------------------------------------------------------------------------------------------------|
| `search_rss_links`        | Web search (DuckDuckGo) for AI-related RSS feed URLs. Falls back to the seed list when offline or rate-limited. |
| `check_feed_availability` | HTTP GET + feedparser validation. Persists valid feeds to the JSON database.                                    |

Both tools are registered on the PydanticAI agent **and** exposed via the standalone `mcp_server.py` (stdio transport) so any MCP-compatible client (Claude Desktop, Cursor …) can call them.

---

## Evaluation System

Five built-in eval cases live in `config.EVAL_CASES`:

| Case                   | What it tests                                               |
|------------------------|-------------------------------------------------------------|
| `major_ai_labs`        | Finds RSS feeds from OpenAI, DeepMind, Google, Hugging Face |
| `ml_news_feeds`        | Finds VentureBeat, TechCrunch, MIT Technology Review feeds  |
| `arxiv_research_feeds` | Finds arXiv AI/ML/CL category feeds                         |
| `uniqueness_check`     | Verifies no duplicate URLs after a discovery run            |
| `availability_check`   | Agent verifies feeds are actually reachable                 |

**Scoring formula** (per case):
```
score = pattern_coverage * 0.6 + link_count_ratio * 0.3 + timing_score * 0.1
```
A case passes when `score >= 0.5`.  Results are persisted to `eval_results.json`.

---

## Quick Start

```bash
cd rss-agent

# 1. Install dependencies
pip install -r requirements.txt

# 2. Seed the DB with known AI feeds
python run.py --mode seed

# 3. Run the agent in discovery mode (uses Ollama mistral-nemo)
python run.py --mode discover

# 4. Run a single custom query
python run.py --mode discover --query "AI safety research blog RSS 2024"

# 5. Run the evaluation suite
python run.py --mode eval

# 6. Print current DB status
python run.py --mode status

# 7. Show historical eval stats
python run.py --mode history
```

### Standalone MCP Server (for Claude Desktop / Cursor)

```bash
python mcp_server.py
```

Add to your MCP client config:
```json
{
  "mcpServers": {
    "rss-ai-agent": {
      "command": "python",
      "args": ["P:/ollama-dev/rss-agent/mcp_server.py"]
    }
  }
}
```

---

## Configuration

Edit `config.py` to change:

| Setting                 | Default                  | Purpose                       |
|-------------------------|--------------------------|-------------------------------|
| `OLLAMA_MODEL`          | `mistral-nemo:latest`    | LLM used by the agent         |
| `OLLAMA_HOST`           | `http://127.0.0.1:11434` | Ollama server URL             |
| `REQUEST_TIMEOUT`       | `12`                     | HTTP timeout (seconds)        |
| `MAX_SEARCH_RESULTS`    | `25`                     | DuckDuckGo max results        |
| `SEED_RSS_FEEDS`        | 26 feeds                 | Pre-loaded AI RSS feeds       |
| `AI_RSS_SEARCH_QUERIES` | 5 queries                | Discovery-mode search queries |
| `EVAL_CASES`            | 5 cases                  | Evaluation test suite         |

---

## Dependencies

- **pydantic-ai** – agent framework with tool/MCP support
- **pydantic** – data validation
- **feedparser** – RSS/Atom feed parsing
- **duckduckgo-search** – free web search (no API key)
- **requests** – HTTP client
- **rich** – console output
- **mcp** *(optional)* – standalone MCP server

