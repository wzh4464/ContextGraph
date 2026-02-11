# Evaluation Module Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the evaluation module to measure Agent Memory effectiveness by comparing SWE-agent performance with and without memory augmentation.

**Architecture:** Parse SWE-agent trajectory files, split 50/50 for train/test, build memory graph from training set, run evaluation comparing control (no memory) vs treatment (with memory) groups.

**Tech Stack:** Python 3.11, pytest, dataclasses, JSON parsing, existing agent_memory module

---

## Task 1: Trajectory Parser

**Files:**
- Create: `agent_memory/evaluation/__init__.py`
- Create: `agent_memory/evaluation/trajectory_parser.py`
- Test: `tests/evaluation/__init__.py`
- Test: `tests/evaluation/test_trajectory_parser.py`

**Step 1: Create evaluation package init files**

```python
# agent_memory/evaluation/__init__.py
"""Evaluation module for Agent Memory experiments."""

from agent_memory.evaluation.trajectory_parser import (
    parse_swe_agent_trajectory,
    SWEAgentStep,
)

__all__ = [
    "parse_swe_agent_trajectory",
    "SWEAgentStep",
]
```

```python
# tests/evaluation/__init__.py
"""Tests for evaluation module."""
```

**Step 2: Write the failing test for trajectory parser**

```python
# tests/evaluation/test_trajectory_parser.py
"""Tests for SWE-agent trajectory parser."""

import pytest
from agent_memory.evaluation.trajectory_parser import (
    parse_swe_agent_trajectory,
    SWEAgentStep,
)
from agent_memory.writer import RawTrajectory


class TestSWEAgentStep:
    def test_step_creation(self):
        step = SWEAgentStep(
            thought="Looking at the file",
            action="cat file.py",
            observation="def foo(): pass",
        )
        assert step.thought == "Looking at the file"
        assert step.action == "cat file.py"
        assert step.observation == "def foo(): pass"


class TestParseSWEAgentTrajectory:
    def test_parse_simple_trajectory(self, tmp_path):
        """Test parsing a simple SWE-agent trajectory file."""
        traj_file = tmp_path / "test__test-123.traj"
        traj_file.write_text('''{
            "environment": "swe_main",
            "trajectory": [
                {
                    "thought": "Let me explore the repo",
                    "action": "ls -la",
                    "observation": "file1.py file2.py",
                    "state": "{\\"working_dir\\": \\"/repo\\"}"
                },
                {
                    "thought": "Now edit the file",
                    "action": "edit file1.py",
                    "observation": "File edited successfully"
                }
            ],
            "info": {
                "exit_status": "submitted",
                "model_stats": {"instance_cost": 0.5}
            }
        }''')

        result = parse_swe_agent_trajectory(traj_file)

        assert isinstance(result, RawTrajectory)
        assert result.instance_id == "test__test-123"
        assert result.success is True
        assert len(result.steps) == 2
        assert result.steps[0]["action"] == "ls -la"
        assert result.steps[0]["observation"] == "file1.py file2.py"

    def test_parse_failed_trajectory(self, tmp_path):
        """Test parsing a failed trajectory (exit_status not submitted)."""
        traj_file = tmp_path / "django__django-12345.traj"
        traj_file.write_text('''{
            "environment": "swe_main",
            "trajectory": [
                {
                    "thought": "Try something",
                    "action": "edit",
                    "observation": "Error: SyntaxError"
                }
            ],
            "info": {
                "exit_status": "failed"
            }
        }''')

        result = parse_swe_agent_trajectory(traj_file)

        assert result.success is False
        assert result.repo == "django/django"
        assert result.instance_id == "django__django-12345"

    def test_extract_repo_from_instance_id(self, tmp_path):
        """Test repo extraction from various instance ID formats."""
        traj_file = tmp_path / "scikit-learn__scikit-learn-9876.traj"
        traj_file.write_text('''{
            "environment": "swe_main",
            "trajectory": [],
            "info": {"exit_status": "submitted"}
        }''')

        result = parse_swe_agent_trajectory(traj_file)

        assert result.repo == "scikit-learn/scikit-learn"
```

**Step 3: Run test to verify it fails**

Run: `cd /Volumes/Mac_Ext/link_cache/codes/ContextGraph/.worktrees/evaluation && python -m pytest tests/evaluation/test_trajectory_parser.py -v`
Expected: FAIL with "No module named 'agent_memory.evaluation'"

**Step 4: Write minimal implementation**

```python
# agent_memory/evaluation/trajectory_parser.py
"""Parse SWE-agent trajectory files into RawTrajectory format."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import re

from agent_memory.writer import RawTrajectory


@dataclass
class SWEAgentStep:
    """Parsed step from SWE-agent trajectory."""

    thought: str
    action: str
    observation: str
    state: Optional[str] = None


def parse_swe_agent_trajectory(path: Path) -> RawTrajectory:
    """
    Parse a SWE-agent trajectory file into RawTrajectory format.

    Args:
        path: Path to .traj file

    Returns:
        RawTrajectory ready for ingestion into Agent Memory
    """
    with open(path, "r") as f:
        data = json.load(f)

    # Extract instance_id from filename (e.g., "django__django-12345.traj")
    instance_id = path.stem

    # Extract repo from instance_id (e.g., "django__django-12345" -> "django/django")
    repo = _extract_repo(instance_id)

    # Check success from exit_status
    info = data.get("info", {})
    exit_status = info.get("exit_status", "")
    success = exit_status == "submitted"

    # Parse trajectory steps
    trajectory = data.get("trajectory", [])
    steps = []
    for step in trajectory:
        steps.append({
            "action": step.get("action", ""),
            "observation": step.get("observation", ""),
            "thought": step.get("thought", ""),
        })

    # Extract problem statement if available
    problem_statement = data.get("problem_statement", "")

    return RawTrajectory(
        instance_id=instance_id,
        repo=repo,
        success=success,
        steps=steps,
        problem_statement=problem_statement,
    )


def _extract_repo(instance_id: str) -> str:
    """
    Extract repository name from SWE-bench instance ID.

    Examples:
        "django__django-12345" -> "django/django"
        "scikit-learn__scikit-learn-9876" -> "scikit-learn/scikit-learn"
        "psf__requests-5432" -> "psf/requests"
    """
    # Pattern: owner__repo-issue_number
    match = re.match(r'^([^_]+(?:__[^_]+)?)__([^-]+(?:-[^-]+)*)-\d+$', instance_id)
    if match:
        owner_part = match.group(1)
        repo_part = match.group(2)

        # Handle double-underscore in owner (e.g., scikit-learn)
        owner = owner_part.replace("__", "-") if "__" in owner_part else owner_part

        return f"{owner}/{repo_part}"

    # Fallback: split on double underscore
    parts = instance_id.split("__")
    if len(parts) >= 2:
        owner = parts[0]
        # Remove issue number from repo
        repo = re.sub(r'-\d+$', '', parts[1])
        return f"{owner}/{repo}"

    return "unknown/unknown"
```

**Step 5: Run test to verify it passes**

Run: `cd /Volumes/Mac_Ext/link_cache/codes/ContextGraph/.worktrees/evaluation && python -m pytest tests/evaluation/test_trajectory_parser.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add agent_memory/evaluation/ tests/evaluation/
git commit -m "feat(eval): add SWE-agent trajectory parser"
```

---

## Task 2: Data Splitter

**Files:**
- Modify: `agent_memory/evaluation/__init__.py`
- Create: `agent_memory/evaluation/data_splitter.py`
- Test: `tests/evaluation/test_data_splitter.py`

**Step 1: Write the failing test**

```python
# tests/evaluation/test_data_splitter.py
"""Tests for data splitter."""

import pytest
from pathlib import Path
from agent_memory.evaluation.data_splitter import random_split, DataSplit


class TestRandomSplit:
    def test_split_ratio(self, tmp_path):
        """Test 50/50 split ratio."""
        # Create 10 dummy trajectory files
        for i in range(10):
            (tmp_path / f"test__test-{i}.traj").touch()

        files = list(tmp_path.glob("*.traj"))
        split = random_split(files, ratio=0.5, seed=42)

        assert len(split.train) == 5
        assert len(split.test) == 5
        assert set(split.train).isdisjoint(set(split.test))

    def test_reproducibility(self, tmp_path):
        """Test that same seed produces same split."""
        for i in range(10):
            (tmp_path / f"test__test-{i}.traj").touch()

        files = list(tmp_path.glob("*.traj"))

        split1 = random_split(files, ratio=0.5, seed=42)
        split2 = random_split(files, ratio=0.5, seed=42)

        assert split1.train == split2.train
        assert split1.test == split2.test

    def test_different_seeds_different_splits(self, tmp_path):
        """Test that different seeds produce different splits."""
        for i in range(10):
            (tmp_path / f"test__test-{i}.traj").touch()

        files = list(tmp_path.glob("*.traj"))

        split1 = random_split(files, ratio=0.5, seed=42)
        split2 = random_split(files, ratio=0.5, seed=123)

        assert split1.train != split2.train

    def test_custom_ratio(self, tmp_path):
        """Test 70/30 split."""
        for i in range(10):
            (tmp_path / f"test__test-{i}.traj").touch()

        files = list(tmp_path.glob("*.traj"))
        split = random_split(files, ratio=0.7, seed=42)

        assert len(split.train) == 7
        assert len(split.test) == 3
```

**Step 2: Run test to verify it fails**

Run: `cd /Volumes/Mac_Ext/link_cache/codes/ContextGraph/.worktrees/evaluation && python -m pytest tests/evaluation/test_data_splitter.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# agent_memory/evaluation/data_splitter.py
"""Split trajectory data into train/test sets."""

from dataclasses import dataclass
from pathlib import Path
from typing import List
import random


@dataclass
class DataSplit:
    """Result of splitting data into train/test sets."""

    train: List[Path]
    test: List[Path]

    @property
    def train_count(self) -> int:
        return len(self.train)

    @property
    def test_count(self) -> int:
        return len(self.test)


def random_split(
    files: List[Path],
    ratio: float = 0.5,
    seed: int = 42,
) -> DataSplit:
    """
    Randomly split files into train/test sets.

    Args:
        files: List of trajectory file paths
        ratio: Fraction of files for training (default 0.5)
        seed: Random seed for reproducibility

    Returns:
        DataSplit with train and test file lists
    """
    # Copy list to avoid modifying original
    files = list(files)

    # Shuffle with fixed seed
    rng = random.Random(seed)
    rng.shuffle(files)

    # Split
    split_idx = int(len(files) * ratio)
    train = files[:split_idx]
    test = files[split_idx:]

    return DataSplit(train=train, test=test)
```

**Step 4: Update __init__.py**

```python
# agent_memory/evaluation/__init__.py
"""Evaluation module for Agent Memory experiments."""

from agent_memory.evaluation.trajectory_parser import (
    parse_swe_agent_trajectory,
    SWEAgentStep,
)
from agent_memory.evaluation.data_splitter import (
    random_split,
    DataSplit,
)

__all__ = [
    "parse_swe_agent_trajectory",
    "SWEAgentStep",
    "random_split",
    "DataSplit",
]
```

**Step 5: Run test to verify it passes**

Run: `cd /Volumes/Mac_Ext/link_cache/codes/ContextGraph/.worktrees/evaluation && python -m pytest tests/evaluation/test_data_splitter.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add agent_memory/evaluation/ tests/evaluation/
git commit -m "feat(eval): add data splitter for train/test split"
```

---

## Task 3: Graph Builder

**Files:**
- Modify: `agent_memory/evaluation/__init__.py`
- Create: `agent_memory/evaluation/graph_builder.py`
- Test: `tests/evaluation/test_graph_builder.py`

**Step 1: Write the failing test**

```python
# tests/evaluation/test_graph_builder.py
"""Tests for graph builder."""

import pytest
from pathlib import Path
from agent_memory.evaluation.graph_builder import build_graph
from agent_memory import AgentMemory


class TestBuildGraph:
    def test_build_graph_from_trajectories(self, tmp_path):
        """Test building graph from trajectory files."""
        # Create a sample trajectory file
        traj_file = tmp_path / "test__test-123.traj"
        traj_file.write_text('''{
            "environment": "swe_main",
            "trajectory": [
                {"thought": "Explore", "action": "ls", "observation": "file.py"},
                {"thought": "Edit", "action": "edit file.py", "observation": "Done"}
            ],
            "info": {"exit_status": "submitted"}
        }''')

        # Build graph in mock mode
        memory = AgentMemory(neo4j_uri=None, embedding_api_key=None)
        result = build_graph([traj_file], memory)

        assert result.trajectories_loaded == 1
        assert result.fragments_created >= 1
        memory.close()

    def test_build_graph_multiple_trajectories(self, tmp_path):
        """Test building graph from multiple trajectories."""
        for i in range(3):
            traj_file = tmp_path / f"test__test-{i}.traj"
            traj_file.write_text(f'''{{
                "environment": "swe_main",
                "trajectory": [
                    {{"thought": "Step", "action": "action{i}", "observation": "obs"}}
                ],
                "info": {{"exit_status": "submitted"}}
            }}''')

        memory = AgentMemory(neo4j_uri=None, embedding_api_key=None)
        result = build_graph(list(tmp_path.glob("*.traj")), memory)

        assert result.trajectories_loaded == 3
        memory.close()

    def test_build_graph_with_consolidation(self, tmp_path):
        """Test that consolidation is triggered after building."""
        traj_file = tmp_path / "test__test-1.traj"
        traj_file.write_text('''{
            "environment": "swe_main",
            "trajectory": [
                {"thought": "Fix", "action": "edit", "observation": "Fixed"}
            ],
            "info": {"exit_status": "submitted"}
        }''')

        memory = AgentMemory(neo4j_uri=None, embedding_api_key=None)
        result = build_graph([traj_file], memory, consolidate=True)

        assert result.consolidation_run is True
        memory.close()
```

**Step 2: Run test to verify it fails**

Run: `cd /Volumes/Mac_Ext/link_cache/codes/ContextGraph/.worktrees/evaluation && python -m pytest tests/evaluation/test_graph_builder.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# agent_memory/evaluation/graph_builder.py
"""Build memory graph from trajectory files."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, TYPE_CHECKING
import logging

from agent_memory.evaluation.trajectory_parser import parse_swe_agent_trajectory

if TYPE_CHECKING:
    from agent_memory import AgentMemory

logger = logging.getLogger(__name__)


@dataclass
class BuildResult:
    """Result of building memory graph."""

    trajectories_loaded: int
    fragments_created: int
    errors: List[str]
    consolidation_run: bool = False


def build_graph(
    trajectory_files: List[Path],
    memory: "AgentMemory",
    consolidate: bool = True,
) -> BuildResult:
    """
    Build memory graph from trajectory files.

    Args:
        trajectory_files: List of .traj file paths
        memory: AgentMemory instance to populate
        consolidate: Whether to run consolidation after loading

    Returns:
        BuildResult with loading statistics
    """
    loaded = 0
    fragments = 0
    errors = []

    for path in trajectory_files:
        try:
            raw = parse_swe_agent_trajectory(path)
            memory.learn(raw)
            loaded += 1
            # Estimate fragments (actual count would require store query)
            fragments += max(1, len(raw.steps) // 3)
            logger.info(f"Loaded trajectory from {path.name}")
        except Exception as e:
            error_msg = f"Failed to load {path.name}: {e}"
            logger.warning(error_msg)
            errors.append(error_msg)

    consolidation_run = False
    if consolidate and loaded > 0:
        try:
            memory.consolidator.consolidate()
            consolidation_run = True
            logger.info("Consolidation completed")
        except Exception as e:
            errors.append(f"Consolidation failed: {e}")

    return BuildResult(
        trajectories_loaded=loaded,
        fragments_created=fragments,
        errors=errors,
        consolidation_run=consolidation_run,
    )
```

**Step 4: Update __init__.py**

```python
# agent_memory/evaluation/__init__.py
"""Evaluation module for Agent Memory experiments."""

from agent_memory.evaluation.trajectory_parser import (
    parse_swe_agent_trajectory,
    SWEAgentStep,
)
from agent_memory.evaluation.data_splitter import (
    random_split,
    DataSplit,
)
from agent_memory.evaluation.graph_builder import (
    build_graph,
    BuildResult,
)

__all__ = [
    "parse_swe_agent_trajectory",
    "SWEAgentStep",
    "random_split",
    "DataSplit",
    "build_graph",
    "BuildResult",
]
```

**Step 5: Run test to verify it passes**

Run: `cd /Volumes/Mac_Ext/link_cache/codes/ContextGraph/.worktrees/evaluation && python -m pytest tests/evaluation/test_graph_builder.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add agent_memory/evaluation/ tests/evaluation/
git commit -m "feat(eval): add graph builder for trajectory ingestion"
```

---

## Task 4: Metrics Calculator

**Files:**
- Modify: `agent_memory/evaluation/__init__.py`
- Create: `agent_memory/evaluation/metrics.py`
- Test: `tests/evaluation/test_metrics.py`

**Step 1: Write the failing test**

```python
# tests/evaluation/test_metrics.py
"""Tests for evaluation metrics."""

import pytest
from agent_memory.evaluation.metrics import (
    ProblemResult,
    EvaluationMetrics,
    calculate_metrics,
)


class TestProblemResult:
    def test_pass_at_1_success(self):
        """Test pass@1 when first attempt succeeds."""
        result = ProblemResult(
            problem_id="test-1",
            attempts=[True, False, False, False, False],
            tokens=[100, 0, 0, 0, 0],
        )
        assert result.pass_at_1 is True
        assert result.first_success_attempt == 1

    def test_pass_at_1_failure(self):
        """Test pass@1 when first attempt fails."""
        result = ProblemResult(
            problem_id="test-2",
            attempts=[False, True, False, False, False],
            tokens=[100, 150, 0, 0, 0],
        )
        assert result.pass_at_1 is False
        assert result.pass_at_3 is True
        assert result.first_success_attempt == 2

    def test_pass_at_k_all_fail(self):
        """Test pass@k when all attempts fail."""
        result = ProblemResult(
            problem_id="test-3",
            attempts=[False, False, False, False, False],
            tokens=[100, 100, 100, 100, 100],
        )
        assert result.pass_at_1 is False
        assert result.pass_at_3 is False
        assert result.pass_at_5 is False
        assert result.first_success_attempt is None

    def test_total_tokens(self):
        """Test total token calculation."""
        result = ProblemResult(
            problem_id="test-4",
            attempts=[True],
            tokens=[500],
        )
        assert result.total_tokens == 500


class TestCalculateMetrics:
    def test_calculate_metrics(self):
        """Test metric calculation from multiple results."""
        results = [
            ProblemResult("p1", [True, False, False, False, False], [100, 0, 0, 0, 0]),
            ProblemResult("p2", [False, True, False, False, False], [100, 100, 0, 0, 0]),
            ProblemResult("p3", [False, False, False, False, False], [100, 100, 100, 100, 100]),
            ProblemResult("p4", [False, False, False, True, False], [100, 100, 100, 100, 0]),
        ]

        metrics = calculate_metrics(results)

        assert metrics.pass_at_1 == 0.25  # 1/4
        assert metrics.pass_at_3 == 0.5   # 2/4
        assert metrics.pass_at_5 == 0.75  # 3/4
        assert metrics.total_problems == 4
        assert metrics.avg_tokens_per_problem == 350  # (100+200+500+400)/4

    def test_calculate_efficiency(self):
        """Test efficiency metric (avg attempts to first success)."""
        results = [
            ProblemResult("p1", [True], [100]),           # 1 attempt
            ProblemResult("p2", [False, False, True], [100, 100, 100]),  # 3 attempts
            ProblemResult("p3", [False, False, False, False, False], [100]*5),  # No success
        ]

        metrics = calculate_metrics(results)

        # Only count successful ones: (1 + 3) / 2 = 2.0
        assert metrics.avg_attempts_to_success == 2.0
```

**Step 2: Run test to verify it fails**

Run: `cd /Volumes/Mac_Ext/link_cache/codes/ContextGraph/.worktrees/evaluation && python -m pytest tests/evaluation/test_metrics.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# agent_memory/evaluation/metrics.py
"""Evaluation metrics calculation."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ProblemResult:
    """Result of running agent on a single problem."""

    problem_id: str
    attempts: List[bool]  # Success/fail for each attempt
    tokens: List[int]     # Tokens used per attempt

    @property
    def pass_at_1(self) -> bool:
        return len(self.attempts) > 0 and self.attempts[0]

    @property
    def pass_at_3(self) -> bool:
        return any(self.attempts[:3])

    @property
    def pass_at_5(self) -> bool:
        return any(self.attempts[:5])

    @property
    def first_success_attempt(self) -> Optional[int]:
        """1-indexed attempt number of first success, or None if all failed."""
        for i, success in enumerate(self.attempts):
            if success:
                return i + 1
        return None

    @property
    def total_tokens(self) -> int:
        return sum(self.tokens)


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics."""

    pass_at_1: float
    pass_at_3: float
    pass_at_5: float
    total_problems: int
    avg_tokens_per_problem: float
    avg_attempts_to_success: Optional[float]


def calculate_metrics(results: List[ProblemResult]) -> EvaluationMetrics:
    """
    Calculate aggregated metrics from problem results.

    Args:
        results: List of ProblemResult objects

    Returns:
        EvaluationMetrics with pass@k rates and efficiency metrics
    """
    if not results:
        return EvaluationMetrics(
            pass_at_1=0.0,
            pass_at_3=0.0,
            pass_at_5=0.0,
            total_problems=0,
            avg_tokens_per_problem=0.0,
            avg_attempts_to_success=None,
        )

    n = len(results)

    # pass@k rates
    pass_1 = sum(1 for r in results if r.pass_at_1) / n
    pass_3 = sum(1 for r in results if r.pass_at_3) / n
    pass_5 = sum(1 for r in results if r.pass_at_5) / n

    # Token consumption
    total_tokens = sum(r.total_tokens for r in results)
    avg_tokens = total_tokens / n

    # Efficiency (average attempts to first success, only for successful problems)
    success_attempts = [r.first_success_attempt for r in results if r.first_success_attempt is not None]
    avg_attempts = sum(success_attempts) / len(success_attempts) if success_attempts else None

    return EvaluationMetrics(
        pass_at_1=pass_1,
        pass_at_3=pass_3,
        pass_at_5=pass_5,
        total_problems=n,
        avg_tokens_per_problem=avg_tokens,
        avg_attempts_to_success=avg_attempts,
    )
```

**Step 4: Update __init__.py**

```python
# agent_memory/evaluation/__init__.py
"""Evaluation module for Agent Memory experiments."""

from agent_memory.evaluation.trajectory_parser import (
    parse_swe_agent_trajectory,
    SWEAgentStep,
)
from agent_memory.evaluation.data_splitter import (
    random_split,
    DataSplit,
)
from agent_memory.evaluation.graph_builder import (
    build_graph,
    BuildResult,
)
from agent_memory.evaluation.metrics import (
    ProblemResult,
    EvaluationMetrics,
    calculate_metrics,
)

__all__ = [
    "parse_swe_agent_trajectory",
    "SWEAgentStep",
    "random_split",
    "DataSplit",
    "build_graph",
    "BuildResult",
    "ProblemResult",
    "EvaluationMetrics",
    "calculate_metrics",
]
```

**Step 5: Run test to verify it passes**

Run: `cd /Volumes/Mac_Ext/link_cache/codes/ContextGraph/.worktrees/evaluation && python -m pytest tests/evaluation/test_metrics.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add agent_memory/evaluation/ tests/evaluation/
git commit -m "feat(eval): add metrics calculator for pass@k and efficiency"
```

---

## Task 5: SWE-agent Tool (query_memory)

**Files:**
- Modify: `agent_memory/evaluation/__init__.py`
- Create: `agent_memory/evaluation/swe_agent_tool.py`
- Test: `tests/evaluation/test_swe_agent_tool.py`

**Step 1: Write the failing test**

```python
# tests/evaluation/test_swe_agent_tool.py
"""Tests for SWE-agent query_memory tool."""

import pytest
from agent_memory.evaluation.swe_agent_tool import (
    QueryMemoryTool,
    QueryMemoryInput,
    QueryMemoryOutput,
)
from agent_memory import AgentMemory


class TestQueryMemoryTool:
    def test_tool_creation(self):
        """Test tool can be created with AgentMemory."""
        memory = AgentMemory(neo4j_uri=None, embedding_api_key=None)
        tool = QueryMemoryTool(memory)

        assert tool.name == "query_memory"
        assert "memory" in tool.description.lower()
        memory.close()

    def test_tool_schema(self):
        """Test tool returns proper JSON schema."""
        memory = AgentMemory(neo4j_uri=None, embedding_api_key=None)
        tool = QueryMemoryTool(memory)

        schema = tool.get_schema()

        assert "name" in schema
        assert "parameters" in schema
        assert "current_error" in schema["parameters"]["properties"]
        memory.close()

    def test_tool_invoke(self):
        """Test invoking the tool returns structured output."""
        memory = AgentMemory(neo4j_uri=None, embedding_api_key=None)
        tool = QueryMemoryTool(memory)

        input_data = QueryMemoryInput(
            current_error="ImportError: No module named 'foo'",
            task_description="Fix import error",
            phase="fixing",
        )

        output = tool.invoke(input_data)

        assert isinstance(output, QueryMemoryOutput)
        assert isinstance(output.methodologies, list)
        assert isinstance(output.similar_fragments, list)
        assert isinstance(output.warnings, list)
        memory.close()

    def test_tool_to_json(self):
        """Test output can be serialized to JSON."""
        memory = AgentMemory(neo4j_uri=None, embedding_api_key=None)
        tool = QueryMemoryTool(memory)

        input_data = QueryMemoryInput(
            current_error="TypeError",
            task_description="Fix type error",
            phase="fixing",
        )

        output = tool.invoke(input_data)
        json_output = output.to_json()

        assert isinstance(json_output, str)
        import json
        parsed = json.loads(json_output)
        assert "methodologies" in parsed
        memory.close()
```

**Step 2: Run test to verify it fails**

Run: `cd /Volumes/Mac_Ext/link_cache/codes/ContextGraph/.worktrees/evaluation && python -m pytest tests/evaluation/test_swe_agent_tool.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# agent_memory/evaluation/swe_agent_tool.py
"""SWE-agent tool implementation for querying agent memory."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import json

from agent_memory.models import State

if TYPE_CHECKING:
    from agent_memory import AgentMemory


@dataclass
class QueryMemoryInput:
    """Input for query_memory tool."""

    current_error: str
    task_description: str
    phase: str  # exploring | understanding | fixing | verifying


@dataclass
class MethodologyInfo:
    """Methodology information for tool output."""

    situation: str
    strategy: str
    confidence: float


@dataclass
class FragmentInfo:
    """Fragment information for tool output."""

    error_type: str
    resolution: str
    from_trajectory: str


@dataclass
class QueryMemoryOutput:
    """Output from query_memory tool."""

    methodologies: List[MethodologyInfo] = field(default_factory=list)
    similar_fragments: List[FragmentInfo] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps({
            "methodologies": [asdict(m) for m in self.methodologies],
            "similar_fragments": [asdict(f) for f in self.similar_fragments],
            "warnings": self.warnings,
        }, indent=2)


class QueryMemoryTool:
    """
    SWE-agent tool for querying agent memory.

    This tool can be registered with SWE-agent to provide
    memory-augmented capabilities during problem solving.
    """

    def __init__(self, memory: "AgentMemory"):
        self.memory = memory
        self.name = "query_memory"
        self.description = (
            "Query agent memory for similar past experiences and methodologies. "
            "Use this to get guidance based on previously solved problems."
        )

    def get_schema(self) -> Dict[str, Any]:
        """Return JSON schema for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "current_error": {
                        "type": "string",
                        "description": "The error message currently being debugged",
                    },
                    "task_description": {
                        "type": "string",
                        "description": "Brief description of the current task",
                    },
                    "phase": {
                        "type": "string",
                        "enum": ["exploring", "understanding", "fixing", "verifying"],
                        "description": "Current phase of problem solving",
                    },
                },
                "required": ["current_error", "task_description", "phase"],
            },
        }

    def invoke(self, input_data: QueryMemoryInput) -> QueryMemoryOutput:
        """
        Invoke the tool with given input.

        Args:
            input_data: QueryMemoryInput with current context

        Returns:
            QueryMemoryOutput with relevant methodologies and fragments
        """
        # Map phase to valid State phase
        phase_map = {
            "exploring": "understanding",
            "understanding": "understanding",
            "fixing": "fixing",
            "verifying": "testing",
        }
        mapped_phase = phase_map.get(input_data.phase, "fixing")

        # Create state for query
        state = State(
            tools=["bash", "edit", "view"],
            repo_summary="",
            task_description=input_data.task_description,
            current_error=input_data.current_error,
            phase=mapped_phase,
        )

        # Query memory
        context = self.memory.query(state)

        # Convert to output format
        methodologies = [
            MethodologyInfo(
                situation=m.situation,
                strategy=m.strategy,
                confidence=m.confidence,
            )
            for m in context.methodologies[:5]  # Limit to top 5
        ]

        fragments = [
            FragmentInfo(
                error_type=f.fragment_type,
                resolution=f.description,
                from_trajectory=f.id,
            )
            for f in context.similar_fragments[:5]  # Limit to top 5
        ]

        return QueryMemoryOutput(
            methodologies=methodologies,
            similar_fragments=fragments,
            warnings=context.warnings,
        )
```

**Step 4: Update __init__.py**

```python
# agent_memory/evaluation/__init__.py
"""Evaluation module for Agent Memory experiments."""

from agent_memory.evaluation.trajectory_parser import (
    parse_swe_agent_trajectory,
    SWEAgentStep,
)
from agent_memory.evaluation.data_splitter import (
    random_split,
    DataSplit,
)
from agent_memory.evaluation.graph_builder import (
    build_graph,
    BuildResult,
)
from agent_memory.evaluation.metrics import (
    ProblemResult,
    EvaluationMetrics,
    calculate_metrics,
)
from agent_memory.evaluation.swe_agent_tool import (
    QueryMemoryTool,
    QueryMemoryInput,
    QueryMemoryOutput,
)

__all__ = [
    "parse_swe_agent_trajectory",
    "SWEAgentStep",
    "random_split",
    "DataSplit",
    "build_graph",
    "BuildResult",
    "ProblemResult",
    "EvaluationMetrics",
    "calculate_metrics",
    "QueryMemoryTool",
    "QueryMemoryInput",
    "QueryMemoryOutput",
]
```

**Step 5: Run test to verify it passes**

Run: `cd /Volumes/Mac_Ext/link_cache/codes/ContextGraph/.worktrees/evaluation && python -m pytest tests/evaluation/test_swe_agent_tool.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add agent_memory/evaluation/ tests/evaluation/
git commit -m "feat(eval): add query_memory tool for SWE-agent integration"
```

---

## Task 6: Results Analyzer

**Files:**
- Modify: `agent_memory/evaluation/__init__.py`
- Create: `agent_memory/evaluation/analyzer.py`
- Test: `tests/evaluation/test_analyzer.py`

**Step 1: Write the failing test**

```python
# tests/evaluation/test_analyzer.py
"""Tests for results analyzer."""

import pytest
from agent_memory.evaluation.metrics import ProblemResult, EvaluationMetrics, calculate_metrics
from agent_memory.evaluation.analyzer import (
    compare_results,
    ComparisonReport,
)


class TestCompareResults:
    def test_compare_basic(self):
        """Test basic comparison of control vs treatment."""
        control_results = [
            ProblemResult("p1", [False, True], [100, 100]),
            ProblemResult("p2", [False, False, False], [100, 100, 100]),
        ]
        treatment_results = [
            ProblemResult("p1", [True], [80]),
            ProblemResult("p2", [False, True], [100, 90]),
        ]

        control = calculate_metrics(control_results)
        treatment = calculate_metrics(treatment_results)

        report = compare_results(control, treatment)

        assert isinstance(report, ComparisonReport)
        assert report.pass_at_1_improvement > 0  # Treatment better
        assert report.token_reduction > 0  # Treatment uses fewer tokens

    def test_compare_efficiency(self):
        """Test efficiency comparison."""
        control_results = [
            ProblemResult("p1", [False, False, True], [100, 100, 100]),
        ]
        treatment_results = [
            ProblemResult("p1", [True], [100]),
        ]

        control = calculate_metrics(control_results)
        treatment = calculate_metrics(treatment_results)

        report = compare_results(control, treatment)

        # Treatment succeeds faster (1 attempt vs 3)
        assert report.efficiency_gain > 0

    def test_report_summary(self):
        """Test report summary generation."""
        control = EvaluationMetrics(
            pass_at_1=0.5,
            pass_at_3=0.7,
            pass_at_5=0.8,
            total_problems=10,
            avg_tokens_per_problem=1000,
            avg_attempts_to_success=2.5,
        )
        treatment = EvaluationMetrics(
            pass_at_1=0.6,
            pass_at_3=0.8,
            pass_at_5=0.9,
            total_problems=10,
            avg_tokens_per_problem=900,
            avg_attempts_to_success=1.8,
        )

        report = compare_results(control, treatment)
        summary = report.to_summary()

        assert "pass@1" in summary.lower()
        assert "improvement" in summary.lower() or "%" in summary
```

**Step 2: Run test to verify it fails**

Run: `cd /Volumes/Mac_Ext/link_cache/codes/ContextGraph/.worktrees/evaluation && python -m pytest tests/evaluation/test_analyzer.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# agent_memory/evaluation/analyzer.py
"""Analyze and compare evaluation results."""

from dataclasses import dataclass
from typing import Optional

from agent_memory.evaluation.metrics import EvaluationMetrics


@dataclass
class ComparisonReport:
    """Comparison between control and treatment groups."""

    # Improvements (positive = treatment better)
    pass_at_1_improvement: float  # Percentage point improvement
    pass_at_3_improvement: float
    pass_at_5_improvement: float

    # Token efficiency (positive = treatment uses fewer)
    token_reduction: float  # Percentage reduction

    # Efficiency (positive = treatment succeeds faster)
    efficiency_gain: float  # Reduction in avg attempts

    # Raw metrics
    control: EvaluationMetrics
    treatment: EvaluationMetrics

    def to_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 50,
            "EVALUATION COMPARISON REPORT",
            "=" * 50,
            "",
            f"Problems evaluated: {self.control.total_problems}",
            "",
            "PASS@K RATES:",
            f"  pass@1: {self.control.pass_at_1:.1%} → {self.treatment.pass_at_1:.1%} ({self._fmt_change(self.pass_at_1_improvement)})",
            f"  pass@3: {self.control.pass_at_3:.1%} → {self.treatment.pass_at_3:.1%} ({self._fmt_change(self.pass_at_3_improvement)})",
            f"  pass@5: {self.control.pass_at_5:.1%} → {self.treatment.pass_at_5:.1%} ({self._fmt_change(self.pass_at_5_improvement)})",
            "",
            "TOKEN CONSUMPTION:",
            f"  Control avg: {self.control.avg_tokens_per_problem:.0f} tokens",
            f"  Treatment avg: {self.treatment.avg_tokens_per_problem:.0f} tokens",
            f"  Reduction: {self.token_reduction:.1%}",
            "",
            "EFFICIENCY (attempts to first success):",
        ]

        if self.control.avg_attempts_to_success and self.treatment.avg_attempts_to_success:
            lines.extend([
                f"  Control avg: {self.control.avg_attempts_to_success:.1f} attempts",
                f"  Treatment avg: {self.treatment.avg_attempts_to_success:.1f} attempts",
                f"  Improvement: {self.efficiency_gain:.1f} fewer attempts",
            ])
        else:
            lines.append("  (insufficient data)")

        lines.extend([
            "",
            "=" * 50,
        ])

        return "\n".join(lines)

    def _fmt_change(self, change: float) -> str:
        """Format change as +X.X% or -X.X%."""
        if change >= 0:
            return f"+{change:.1%}"
        else:
            return f"{change:.1%}"


def compare_results(
    control: EvaluationMetrics,
    treatment: EvaluationMetrics,
) -> ComparisonReport:
    """
    Compare control and treatment evaluation results.

    Args:
        control: Metrics from control group (no memory)
        treatment: Metrics from treatment group (with memory)

    Returns:
        ComparisonReport with improvement metrics
    """
    # Pass@k improvements (percentage points)
    pass_1_imp = treatment.pass_at_1 - control.pass_at_1
    pass_3_imp = treatment.pass_at_3 - control.pass_at_3
    pass_5_imp = treatment.pass_at_5 - control.pass_at_5

    # Token reduction (percentage)
    if control.avg_tokens_per_problem > 0:
        token_reduction = (control.avg_tokens_per_problem - treatment.avg_tokens_per_problem) / control.avg_tokens_per_problem
    else:
        token_reduction = 0.0

    # Efficiency gain (fewer attempts)
    if control.avg_attempts_to_success and treatment.avg_attempts_to_success:
        efficiency_gain = control.avg_attempts_to_success - treatment.avg_attempts_to_success
    else:
        efficiency_gain = 0.0

    return ComparisonReport(
        pass_at_1_improvement=pass_1_imp,
        pass_at_3_improvement=pass_3_imp,
        pass_at_5_improvement=pass_5_imp,
        token_reduction=token_reduction,
        efficiency_gain=efficiency_gain,
        control=control,
        treatment=treatment,
    )
```

**Step 4: Update __init__.py**

```python
# agent_memory/evaluation/__init__.py
"""Evaluation module for Agent Memory experiments."""

from agent_memory.evaluation.trajectory_parser import (
    parse_swe_agent_trajectory,
    SWEAgentStep,
)
from agent_memory.evaluation.data_splitter import (
    random_split,
    DataSplit,
)
from agent_memory.evaluation.graph_builder import (
    build_graph,
    BuildResult,
)
from agent_memory.evaluation.metrics import (
    ProblemResult,
    EvaluationMetrics,
    calculate_metrics,
)
from agent_memory.evaluation.swe_agent_tool import (
    QueryMemoryTool,
    QueryMemoryInput,
    QueryMemoryOutput,
)
from agent_memory.evaluation.analyzer import (
    compare_results,
    ComparisonReport,
)

__all__ = [
    "parse_swe_agent_trajectory",
    "SWEAgentStep",
    "random_split",
    "DataSplit",
    "build_graph",
    "BuildResult",
    "ProblemResult",
    "EvaluationMetrics",
    "calculate_metrics",
    "QueryMemoryTool",
    "QueryMemoryInput",
    "QueryMemoryOutput",
    "compare_results",
    "ComparisonReport",
]
```

**Step 5: Run test to verify it passes**

Run: `cd /Volumes/Mac_Ext/link_cache/codes/ContextGraph/.worktrees/evaluation && python -m pytest tests/evaluation/test_analyzer.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add agent_memory/evaluation/ tests/evaluation/
git commit -m "feat(eval): add results analyzer for comparison reports"
```

---

## Task 7: CLI Script

**Files:**
- Create: `scripts/run_evaluation.py`

**Step 1: Write the CLI script**

```python
#!/usr/bin/env python
# scripts/run_evaluation.py
"""CLI for running Agent Memory evaluation experiments."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_memory import AgentMemory
from agent_memory.evaluation import (
    random_split,
    build_graph,
    calculate_metrics,
    compare_results,
    parse_swe_agent_trajectory,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run Agent Memory evaluation experiment"
    )
    parser.add_argument(
        "trajectories_dir",
        type=Path,
        help="Directory containing SWE-agent .traj files",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.5,
        help="Train/test split ratio (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default=None,
        help="Neo4j URI (default: None for mock mode)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for results (default: stdout)",
    )

    args = parser.parse_args()

    # Find trajectory files
    traj_files = list(args.trajectories_dir.glob("*.traj"))
    if not traj_files:
        print(f"No .traj files found in {args.trajectories_dir}")
        sys.exit(1)

    print(f"Found {len(traj_files)} trajectory files")

    # Split data
    split = random_split(traj_files, ratio=args.split_ratio, seed=args.seed)
    print(f"Train: {split.train_count}, Test: {split.test_count}")

    # Build graph from training data
    print("\nBuilding memory graph from training trajectories...")
    memory = AgentMemory(neo4j_uri=args.neo4j_uri, embedding_api_key=None)
    build_result = build_graph(split.train, memory, consolidate=True)
    print(f"Loaded {build_result.trajectories_loaded} trajectories")

    # TODO: Actual evaluation would run SWE-agent here
    # For now, print summary
    print("\n" + "=" * 50)
    print("EVALUATION SETUP COMPLETE")
    print("=" * 50)
    print(f"Training trajectories: {build_result.trajectories_loaded}")
    print(f"Test problems: {split.test_count}")
    print("\nTo run full evaluation, integrate with SWE-agent runner.")

    memory.close()


if __name__ == "__main__":
    main()
```

**Step 2: Make script executable**

Run: `chmod +x /Volumes/Mac_Ext/link_cache/codes/ContextGraph/.worktrees/evaluation/scripts/run_evaluation.py`

**Step 3: Create scripts directory if needed and commit**

```bash
mkdir -p scripts
git add scripts/run_evaluation.py
git commit -m "feat(eval): add CLI script for running evaluation"
```

---

## Task 8: Final Integration Test

**Files:**
- Test: `tests/evaluation/test_integration.py`

**Step 1: Write integration test**

```python
# tests/evaluation/test_integration.py
"""Integration tests for evaluation module."""

import pytest
from pathlib import Path
from agent_memory import AgentMemory
from agent_memory.evaluation import (
    parse_swe_agent_trajectory,
    random_split,
    build_graph,
    calculate_metrics,
    compare_results,
    ProblemResult,
    QueryMemoryTool,
    QueryMemoryInput,
)


class TestEvaluationIntegration:
    def test_full_pipeline(self, tmp_path):
        """Test the complete evaluation pipeline."""
        # 1. Create sample trajectory files
        for i in range(4):
            success = i % 2 == 0
            traj_file = tmp_path / f"test__test-{i}.traj"
            traj_file.write_text(f'''{{
                "environment": "swe_main",
                "trajectory": [
                    {{"thought": "Exploring", "action": "ls", "observation": "file.py"}},
                    {{"thought": "Editing", "action": "edit file.py", "observation": "Done"}}
                ],
                "info": {{"exit_status": "{'submitted' if success else 'failed'}"}}
            }}''')

        # 2. Split data
        files = list(tmp_path.glob("*.traj"))
        split = random_split(files, ratio=0.5, seed=42)

        assert len(split.train) == 2
        assert len(split.test) == 2

        # 3. Build graph from training data
        memory = AgentMemory(neo4j_uri=None, embedding_api_key=None)
        build_result = build_graph(split.train, memory)

        assert build_result.trajectories_loaded == 2
        assert build_result.errors == []

        # 4. Use query_memory tool
        tool = QueryMemoryTool(memory)
        output = tool.invoke(QueryMemoryInput(
            current_error="ImportError",
            task_description="Fix import",
            phase="fixing",
        ))

        assert output is not None

        # 5. Calculate metrics (simulated results)
        control_results = [
            ProblemResult("p1", [False, True], [100, 100]),
            ProblemResult("p2", [False, False, False], [100, 100, 100]),
        ]
        treatment_results = [
            ProblemResult("p1", [True], [80]),
            ProblemResult("p2", [False, True], [100, 90]),
        ]

        control = calculate_metrics(control_results)
        treatment = calculate_metrics(treatment_results)

        # 6. Compare results
        report = compare_results(control, treatment)

        assert report.pass_at_1_improvement > 0
        summary = report.to_summary()
        assert "pass@1" in summary.lower()

        memory.close()
```

**Step 2: Run test to verify it passes**

Run: `cd /Volumes/Mac_Ext/link_cache/codes/ContextGraph/.worktrees/evaluation && python -m pytest tests/evaluation/test_integration.py -v`
Expected: PASS

**Step 3: Run all evaluation tests**

Run: `cd /Volumes/Mac_Ext/link_cache/codes/ContextGraph/.worktrees/evaluation && python -m pytest tests/evaluation/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add tests/evaluation/
git commit -m "test(eval): add integration test for evaluation pipeline"
```

---

## Task 9: Update Package Exports

**Files:**
- Modify: `agent_memory/__init__.py`

**Step 1: Update main package __init__.py**

Add to the existing `agent_memory/__init__.py`:

```python
# Add this import at the end of the existing imports
from agent_memory import evaluation

# Add to __all__
__all__ = [
    # ... existing exports ...
    # Evaluation module
    "evaluation",
]
```

**Step 2: Run all tests**

Run: `cd /Volumes/Mac_Ext/link_cache/codes/ContextGraph/.worktrees/evaluation && python -m pytest tests/ -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add agent_memory/__init__.py
git commit -m "feat: export evaluation module from main package"
```

---

## Summary

This plan implements 6 new modules in `agent_memory/evaluation/`:

1. **trajectory_parser.py** - Parse SWE-agent .traj files to RawTrajectory
2. **data_splitter.py** - Random train/test split with reproducibility
3. **graph_builder.py** - Build memory graph from training trajectories
4. **metrics.py** - Calculate pass@k, tokens, efficiency metrics
5. **swe_agent_tool.py** - query_memory tool for SWE-agent integration
6. **analyzer.py** - Compare control vs treatment results

Plus:
- `scripts/run_evaluation.py` - CLI for running experiments
- Comprehensive tests for each module

Total: 9 tasks, ~30 steps
