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
