"""Agent Memory - Long-term memory system for coding agents."""

from agent_memory.models import (
    Trajectory,
    Fragment,
    State,
    Methodology,
    ErrorPattern,
)
from agent_memory.neo4j_store import Neo4jStore
from agent_memory.embeddings import EmbeddingClient, get_embedding_client

__version__ = "0.1.0"
__all__ = [
    "Trajectory",
    "Fragment",
    "State",
    "Methodology",
    "ErrorPattern",
    "Neo4jStore",
    "EmbeddingClient",
    "get_embedding_client",
]
