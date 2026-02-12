"""Memory Retriever - multi-dimensional search."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import re
import logging

from agent_memory.models import State, Methodology, Fragment, ErrorPattern

if TYPE_CHECKING:
    from agent_memory.neo4j_store import Neo4jStore
    from agent_memory.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from memory retrieval."""

    methodologies: List[Methodology] = field(default_factory=list)
    similar_fragments: List[Fragment] = field(default_factory=list)
    error_solutions: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return (
            not self.methodologies and
            not self.similar_fragments and
            not self.error_solutions
        )


class MemoryRetriever:
    """Retrieve relevant memories using multiple dimensions."""

    def __init__(
        self,
        store: Optional["Neo4jStore"],
        embedder: Optional["EmbeddingClient"],
    ):
        self.store = store
        self.embedder = embedder

    def retrieve(self, current_state: State, top_k: int = 5) -> RetrievalResult:
        """
        Retrieve relevant memories for current state.

        Uses four dimensions:
        1. Error-based: Match by error type and keywords
        2. Task-based: Match by task type and repo context
        3. State-based: Match by phase and tools
        4. Semantic: Match by embedding similarity
        """
        result = RetrievalResult()

        if not self.store:
            return result

        # 1. Error-based retrieval
        if current_state.current_error:
            error_results = self.by_error(current_state.current_error)
            result.methodologies.extend(error_results)

        # 2. Task-based retrieval
        task_results = self.by_task(
            task_description=current_state.task_description,
            repo_summary=current_state.repo_summary,
        )
        result.similar_fragments.extend(task_results)

        # 3. State-based retrieval
        state_results = self.by_state(current_state)
        result.methodologies.extend(state_results)

        # 4. Semantic retrieval (if embedder available)
        if self.embedder and current_state.embedding:
            semantic_results = self.by_semantic(current_state.embedding, top_k)
            result.similar_fragments.extend(semantic_results)

        # Fuse and rank results
        result.methodologies = self._dedupe_and_rank(result.methodologies, top_k)
        result.similar_fragments = self._dedupe_fragments(result.similar_fragments, top_k)

        # Add warnings for potential failure patterns
        result.warnings = self._get_warnings(current_state)

        return result

    def by_error(self, error_message: str) -> List[Methodology]:
        """Retrieve methodologies that resolved similar errors."""
        if not self.store:
            return []

        query = self._build_error_query(
            error_type=self._extract_error_type(error_message),
            error_message=error_message,
        )

        results = self.store.execute_query(query)
        return [self._dict_to_methodology(r["m"]) for r in results if "m" in r]

    def by_task(
        self,
        task_description: str,
        repo_summary: str,
    ) -> List[Fragment]:
        """Retrieve fragments from similar successful trajectories."""
        if not self.store:
            return []

        # Extract repo name from summary if available
        repo_filter = self._extract_repo_name(repo_summary)

        # Extract keywords from task description for matching
        task_keywords = self._extract_keywords(task_description)

        if repo_filter and task_keywords:
            # Filter by repo and match task keywords in trajectory summary
            query = """
            MATCH (t:Trajectory)-[:HAS_FRAGMENT]->(f:Fragment)
            WHERE t.success = true
              AND (t.repo CONTAINS $repo OR t.summary CONTAINS $repo)
              AND ANY(kw IN $keywords WHERE toLower(t.summary) CONTAINS toLower(kw))
            RETURN f, t.summary as traj_summary
            ORDER BY f.outcome DESC
            LIMIT 10
            """
            results = self.store.execute_query(query, {
                "repo": repo_filter,
                "keywords": task_keywords,
            })
        elif repo_filter:
            # Filter by repo only
            query = """
            MATCH (t:Trajectory)-[:HAS_FRAGMENT]->(f:Fragment)
            WHERE t.success = true
              AND (t.repo CONTAINS $repo OR t.summary CONTAINS $repo)
            RETURN f
            ORDER BY f.outcome DESC
            LIMIT 10
            """
            results = self.store.execute_query(query, {"repo": repo_filter})
        elif task_keywords:
            # Filter by task keywords only
            query = """
            MATCH (t:Trajectory)-[:HAS_FRAGMENT]->(f:Fragment)
            WHERE t.success = true
              AND ANY(kw IN $keywords WHERE toLower(t.summary) CONTAINS toLower(kw))
            RETURN f
            ORDER BY f.outcome DESC
            LIMIT 10
            """
            results = self.store.execute_query(query, {"keywords": task_keywords})
        else:
            # Fallback to generic query
            query = """
            MATCH (t:Trajectory)-[:HAS_FRAGMENT]->(f:Fragment)
            WHERE t.success = true
            RETURN f
            ORDER BY f.outcome DESC
            LIMIT 10
            """
            results = self.store.execute_query(query)

        return [self._dict_to_fragment(r["f"]) for r in results if "f" in r]

    def _extract_repo_name(self, repo_summary: str) -> Optional[str]:
        """Extract repository name from repo summary."""
        if not repo_summary:
            return None
        # Try to extract owner/repo format
        match = re.search(r'([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)', repo_summary)
        if match:
            return match.group(1)
        # Return first significant word as fallback
        words = repo_summary.split()
        return words[0] if words else None

    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract meaningful keywords from text."""
        if not text:
            return []
        # Remove common stop words and extract significant terms
        stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "is", "are", "was", "were", "and", "or", "but"}
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        # Return unique keywords
        seen = set()
        unique = []
        for w in keywords:
            if w not in seen:
                seen.add(w)
                unique.append(w)
                if len(unique) >= max_keywords:
                    break
        return unique

    def by_state(self, state: State) -> List[Methodology]:
        """Retrieve methodologies applicable to current state."""
        if not self.store:
            return []

        query = """
        MATCH (m:Methodology)
        WHERE m.situation CONTAINS $phase
        RETURN m
        ORDER BY m.confidence DESC
        LIMIT 5
        """

        results = self.store.execute_query(query, {"phase": state.phase})
        return [self._dict_to_methodology(r["m"]) for r in results if "m" in r]

    def by_semantic(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Fragment]:
        """Retrieve fragments by semantic similarity."""
        if not self.store:
            return []

        # Neo4j vector search (requires vector index)
        query = """
        MATCH (f:Fragment)
        WHERE f.embedding IS NOT NULL
        WITH f, gds.similarity.cosine(f.embedding, $embedding) AS similarity
        ORDER BY similarity DESC
        LIMIT $top_k
        RETURN f, similarity
        """

        try:
            results = self.store.execute_query(query, {
                "embedding": query_embedding,
                "top_k": top_k,
            })
            return [self._dict_to_fragment(r["f"]) for r in results if "f" in r]
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return []

    def _build_error_query(self, error_type: Optional[str], error_message: str) -> str:
        """Build Cypher query for error-based retrieval."""
        if error_type:
            return f"""
            MATCH (e:ErrorPattern {{error_type: '{error_type}'}})-[:RESOLVED_BY]->(m:Methodology)
            RETURN m
            ORDER BY m.confidence DESC
            LIMIT 5
            """
        else:
            return """
            MATCH (m:Methodology)
            WHERE m.situation CONTAINS 'error'
            RETURN m
            ORDER BY m.confidence DESC
            LIMIT 5
            """

    def _extract_error_type(self, error_message: str) -> Optional[str]:
        """Extract error type from message."""
        match = re.search(r'(\w+Error|\w+Exception)', error_message)
        return match.group(1) if match else None

    def _dedupe_and_rank(
        self,
        methodologies: List[Methodology],
        top_k: int,
    ) -> List[Methodology]:
        """Deduplicate and rank methodologies."""
        seen_ids = set()
        unique = []
        for m in methodologies:
            if m.id not in seen_ids:
                seen_ids.add(m.id)
                unique.append(m)

        # Sort by confidence
        unique.sort(key=lambda m: m.confidence, reverse=True)
        return unique[:top_k]

    def _dedupe_fragments(
        self,
        fragments: List[Fragment],
        top_k: int,
    ) -> List[Fragment]:
        """Deduplicate fragments."""
        seen_ids = set()
        unique = []
        for f in fragments:
            if f.id not in seen_ids:
                seen_ids.add(f.id)
                unique.append(f)
        return unique[:top_k]

    def _get_warnings(self, state: State) -> List[str]:
        """Get warnings for potential failure patterns."""
        warnings = []

        if state.current_error:
            error_type = self._extract_error_type(state.current_error)
            if error_type:
                warnings.append(f"Common mistake with {error_type}: Check import paths carefully")

        return warnings

    def _dict_to_methodology(self, d: Dict[str, Any]) -> Methodology:
        """Convert dict to Methodology."""
        return Methodology.from_dict(d)

    def _dict_to_fragment(self, d: Dict[str, Any]) -> Fragment:
        """Convert dict to Fragment."""
        return Fragment.from_dict(d)
