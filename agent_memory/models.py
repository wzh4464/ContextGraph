"""Core data models for Agent Memory."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime


@dataclass
class Trajectory:
    """Trajectory summary node - represents a complete agent run."""

    id: str
    instance_id: str          # SWE-bench instance ID
    repo: str                 # Repository name
    task_type: str            # bug_fix / feature / refactor
    success: bool             # Whether the task succeeded
    total_steps: int          # Total steps taken
    summary: str              # Natural language summary
    embedding: Optional[List[float]] = None  # Semantic embedding
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "instance_id": self.instance_id,
            "repo": self.repo,
            "task_type": self.task_type,
            "success": self.success,
            "total_steps": self.total_steps,
            "summary": self.summary,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trajectory":
        created_at = d.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        return cls(
            id=d["id"],
            instance_id=d["instance_id"],
            repo=d["repo"],
            task_type=d["task_type"],
            success=d["success"],
            total_steps=d["total_steps"],
            summary=d["summary"],
            embedding=d.get("embedding"),
            created_at=created_at or datetime.now(),
        )


# Placeholder classes to satisfy __init__.py imports
@dataclass
class Fragment:
    """Placeholder - will be implemented in Task 1.3."""
    id: str
    step_range: Tuple[int, int] = (0, 0)
    fragment_type: str = "exploration"
    description: str = ""
    action_sequence: List[str] = field(default_factory=list)
    outcome: str = ""
    embedding: Optional[List[float]] = None


@dataclass
class State:
    """Placeholder - will be implemented in Task 1.4."""
    tools: List[str] = field(default_factory=list)
    repo_summary: str = ""
    task_description: str = ""
    current_error: str = ""
    phase: str = "understanding"
    embedding: Optional[List[float]] = None


@dataclass
class Methodology:
    """Placeholder - will be implemented in Task 1.5."""
    id: str = ""
    situation: str = ""
    strategy: str = ""
    confidence: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    embedding: Optional[List[float]] = None


@dataclass
class ErrorPattern:
    """Placeholder - will be implemented in Task 1.6."""
    id: str = ""
    error_type: str = ""
    error_keywords: List[str] = field(default_factory=list)
    context: str = ""
    frequency: int = 0
