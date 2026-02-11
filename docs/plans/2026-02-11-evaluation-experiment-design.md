# Agent Memory Evaluation Experiment Design

## Overview

Evaluate the effectiveness of Agent Memory (ContextGraph V2) by comparing SWE-agent performance with and without memory augmentation.

## Experiment Architecture

```
SWE-agent Trajectories (100%)
           │
           ▼
    ┌──────────────┐
    │  Random Split │
    │    50/50     │
    └──────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐  ┌─────────┐
│  Train  │  │  Test   │
│   50%   │  │   50%   │
└─────────┘  └─────────┘
     │              │
     ▼              ▼
┌─────────┐   ┌───────────────────────┐
│  Build  │   │    Run Evaluation     │
│  Graph  │   │  ┌─────────────────┐  │
└─────────┘   │  │ Control (no mem)│  │
     │        │  └─────────────────┘  │
     │        │  ┌─────────────────┐  │
     └───────►│  │Treatment (w/mem)│  │
              │  └─────────────────┘  │
              └───────────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │ Compare Results │
              │ - pass@k        │
              │ - tokens        │
              │ - efficiency    │
              └─────────────────┘
```

## Metrics

| Metric | Description | How to Measure |
|--------|-------------|----------------|
| **pass@1** | Success rate on first attempt | `successes / total_problems` |
| **pass@3** | Success rate within 3 attempts | `any_success_in_3 / total_problems` |
| **pass@5** | Success rate within 5 attempts | `any_success_in_5 / total_problems` |
| **Token consumption** | Total tokens used per problem | Sum of input + output tokens |
| **Efficiency** | Attempts needed to first success | `first_success_attempt` (1-5, or None if all fail) |

## SWE-agent Tool Integration

### New Tool: `query_memory`

```python
{
    "name": "query_memory",
    "description": "Query agent memory for similar past experiences and methodologies",
    "parameters": {
        "current_error": "The error message currently being debugged",
        "task_description": "Brief description of the current task",
        "phase": "Current phase: exploring|understanding|fixing|verifying"
    }
}
```

### Integration Point

Add to SWE-agent's tool configuration:

```python
# In SWE-agent config
tools:
  - name: query_memory
    enabled: true  # Treatment group only
```

### Response Format

```json
{
    "methodologies": [
        {
            "situation": "ImportError in Django models",
            "strategy": "Check circular imports, verify INSTALLED_APPS",
            "confidence": 0.85
        }
    ],
    "similar_fragments": [
        {
            "error_type": "ImportError",
            "resolution": "Added missing __init__.py",
            "from_trajectory": "django__django-12345"
        }
    ],
    "warnings": [
        "Pattern detected: repeated edit-test cycles without progress"
    ]
}
```

## Data Pipeline

### Phase 1: Data Preparation

```python
# trajectory_parser.py
def parse_swe_agent_json(path: str) -> RawTrajectory:
    """Convert SWE-agent output to RawTrajectory format."""

# data_splitter.py
def random_split(trajectories: List[Path], ratio: float = 0.5) -> Tuple[List, List]:
    """Random 50/50 split with fixed seed for reproducibility."""
```

### Phase 2: Graph Building

```python
# graph_builder.py
def build_graph(train_trajectories: List[Path], memory: AgentMemory):
    """Load training trajectories into memory graph."""
    for path in train_trajectories:
        raw = parse_swe_agent_json(path)
        memory.learn(raw)

    # Trigger consolidation to extract methodologies
    memory.consolidator.consolidate()
```

### Phase 3: Evaluation

```python
# runner.py
def run_evaluation(
    test_problems: List[Problem],
    memory: Optional[AgentMemory],
    k: int = 5
) -> EvaluationResults:
    """
    Run SWE-agent on test problems.

    Args:
        memory: None for control group, AgentMemory instance for treatment
        k: Number of attempts per problem
    """
```

### Phase 4: Analysis

```python
# analyzer.py
def compare_results(
    control: EvaluationResults,
    treatment: EvaluationResults
) -> ComparisonReport:
    """Generate comparison statistics and visualizations."""
```

## Project Structure

```
agent_memory/
├── evaluation/
│   ├── __init__.py
│   ├── data_splitter.py     # Random 50/50 split
│   ├── graph_builder.py     # Build graph from trajectories
│   ├── trajectory_parser.py # Parse SWE-agent JSON → RawTrajectory
│   ├── swe_agent_tool.py    # query_memory tool implementation
│   ├── runner.py            # Run evaluation experiments
│   ├── metrics.py           # Calculate pass@k, tokens, efficiency
│   └── analyzer.py          # Compare results, generate reports
├── ... (existing modules)

scripts/
├── run_evaluation.py        # CLI for running experiments
└── analyze_results.py       # CLI for generating reports

tests/
├── evaluation/
│   ├── test_data_splitter.py
│   ├── test_trajectory_parser.py
│   └── test_metrics.py
```

## Experiment Protocol

1. **Setup**
   - Collect N SWE-agent trajectory files
   - Split randomly 50/50 (seed=42 for reproducibility)
   - Initialize Neo4j database

2. **Build Memory Graph**
   - Parse training trajectories
   - Load into AgentMemory
   - Run consolidation

3. **Run Control Group**
   - For each test problem, run SWE-agent 5 times
   - No memory augmentation
   - Record: success/fail, tokens, attempt number

4. **Run Treatment Group**
   - For each test problem, run SWE-agent 5 times
   - Enable `query_memory` tool
   - Record: success/fail, tokens, attempt number, queries made

5. **Analyze**
   - Calculate pass@1, pass@3, pass@5 for both groups
   - Compare token consumption
   - Compare efficiency (first success attempt)
   - Statistical significance tests

## Success Criteria

The memory system is considered effective if:

- **pass@1 improvement**: Treatment group shows ≥5% higher pass@1
- **Token reduction**: Treatment group uses ≤10% fewer tokens on average
- **Efficiency gain**: Treatment group succeeds in fewer attempts

## Timeline

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Implement evaluation module | 1 batch |
| 2 | Collect & prepare trajectories | 1 batch |
| 3 | Run experiments | 1 batch |
| 4 | Analyze & report | 1 batch |
