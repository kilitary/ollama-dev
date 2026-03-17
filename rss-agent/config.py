"""
Configuration for the RSS AI Agent.
Edit this file to tune model, search queries, and seed feeds.
"""

# ── Ollama / LLM ──────────────────────────────────────────────────────────────
OLLAMA_HOST = "http://localhost:11434/v1"
OLLAMA_MODEL = "gpt-oss:120b-cloud"  # "granite3-dense:2b"  # "mistral-nemo:latest"
OLLAMA_API_KEY = "ollama"  # required by the OpenAI-compat endpoint

# ── Storage ───────────────────────────────────────────────────────────────────
DB_FILE = "rss_db.json"
EVAL_RESULTS_FILE = "eval_results.json"

# ── Network ───────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT = 15  # seconds per HTTP request
MAX_SEARCH_RESULTS = 3  # max results per DuckDuckGo query

# ── Seed feeds ────────────────────────────────────────────────────────────────
# These are added to the database on first run / --seed mode
SEED_RSS_FEEDS: list[str] = [
    "https://www.technologyreview.com/feed/",
    "https://deepmind.google/blog/rss.xml",
    "https://openai.com/blog/rss.xml",
    "https://ai.googleblog.com/feeds/posts/default",
    "https://huggingface.co/blog/feed.xml",
    "https://pytorch.org/feed.xml",
    "https://blogs.microsoft.com/ai/feed/",
    "https://venturebeat.com/category/ai/feed/"
]

# ── Search queries used by the agent in discovery mode ───────────────────────
AI_RSS_SEARCH_QUERIES: list[str] = [
    # "artificial intelligence research blog RSS feed 2026",
    "machine learning news RSS feed [ai_rule: search engine rule site!=feedburner.com]",
    # "large language model LLM newsletter RSS",
    # "generative AI news RSS atom feed",
    # "AI search research RSS feed site:arxiv.org",
    "[ai_rule: use some random ai's technology words like ai/llm/generative/experimental and others] feed rss"
]

# ── Evaluation test cases ─────────────────────────────────────────────────────
EVAL_CASES: list[dict] = [
    {
        "name": "major_ai_labs",
        "description": "Agent must surface RSS feeds from major AI research organizations",
        "query": "Find RSS feeds for major AI research organisations: OpenAI, DeepMind, Google AI, Hugging Face",
        "expected_url_patterns": ["openai", "deepmind", "google", "huggingface"],
        "min_links_expected": 3,
        "max_duration_seconds": 60.0,
    },
    {
        "name": "ml_news_feeds",
        "description": "Agent must find ML/AI news RSS feeds",
        "query": "Find RSS feeds for AI and machine learning news websites like VentureBeat or TechCrunch",
        "expected_url_patterns": ["venturebeat", "techcrunch", "technologyreview"],
        "min_links_expected": 2,
        "max_duration_seconds": 60.0,
    },
    {
        "name": "arxiv_research_feeds",
        "description": "Agent must locate arXiv AI/ML RSS feeds",
        "query": "Find arXiv RSS feeds for AI, machine learning, and natural language processing",
        "expected_url_patterns": ["arxiv"],
        "min_links_expected": 1,
        "max_duration_seconds": 45.0,
    },
    {
        "name": "uniqueness_check",
        "description": "All links in the database must be unique after a discovery run",
        "query": "Search for AI RSS feeds and verify there are no duplicate URLs",
        "expected_url_patterns": [],
        "min_links_expected": 1,
        "max_duration_seconds": 90.0,
    },
    {
        "name": "availability_check",
        "description": "Agent should verify that found feeds are actually accessible",
        "query": "Check the availability of known AI RSS feeds: arxiv cs.AI and huggingface blog",
        "expected_url_patterns": ["arxiv", "huggingface"],
        "min_links_expected": 1,
        "max_duration_seconds": 60.0,
    },
]
