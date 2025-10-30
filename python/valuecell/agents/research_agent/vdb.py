"""
Vector database configuration for Research Agent.

This module prefers explicit EMBEDDER_* environment variables to construct an
embedder (e.g., OpenAI embedding). When not provided, it falls back to the
centralized provider factory which can auto-select a provider that supports
embedding (e.g., SiliconFlow).

Configuration priority:
1) EMBEDDER_* env vars (model id / dimension / base url / api key)
2) Centralized provider config via valuecell.utils.model
"""

from agno.vectordb.lancedb import LanceDb
from agno.vectordb.search import SearchType

import os
import valuecell.utils.model as model_utils_mod
from valuecell.utils.db import resolve_lancedb_uri
from agno.knowledge.embedder.openai import OpenAIEmbedder

def _get_embedder() -> object:
    """Return an embedder instance.

    - If EMBEDDER_API_KEY is set, construct an OpenAIEmbedder using
      EMBEDDER_MODEL_ID / EMBEDDER_DIMENSION / EMBEDDER_BASE_URL.
    - Otherwise, fall back to the centralized provider factory which will pick
      a provider supporting embedding (e.g., SiliconFlow) based on available keys.
    """
    api_key = os.getenv("EMBEDDER_API_KEY")
    if api_key:
        model_id = os.getenv("EMBEDDER_MODEL_ID") or "text-embedding-3-large"
        base_url = os.getenv("EMBEDDER_BASE_URL") or "https://api.openai.com/v1"
        try:
            dim = int(os.getenv("EMBEDDER_DIMENSION", "1536"))
        except Exception:
            dim = 1536
        return OpenAIEmbedder(dimensions=dim, id=model_id, base_url=base_url, api_key=api_key)

    # Fallback to centralized factory (auto-pick a provider with embeddings)
    return model_utils_mod.get_embedder_for_agent("research_agent")


embedder = _get_embedder()

# Alternative usage examples:
# embedder = get_embedder()  # Use default env key
# embedder = get_embedder("EMBEDDER_MODEL_ID", dimensions=3072)  # Override dimensions
# embedder = get_embedder_for_agent("research_agent")  # Use agent-specific config

# Create vector database with the configured embedder
vector_db = LanceDb(
    table_name="research_agent_knowledge_base",
    uri=resolve_lancedb_uri(),
    embedder=embedder,
    # reranker=reranker,  # Optional: can be configured later, reranker config in modelprovider yaml file if needed
    search_type=SearchType.hybrid,
    use_tantivy=False,
)
