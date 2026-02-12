#!/usr/bin/env python
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
