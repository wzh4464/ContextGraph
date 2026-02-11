"""Tests for data models."""

import pytest
from agent_memory.models import Trajectory, Fragment, State, Methodology, ErrorPattern


class TestTrajectory:
    def test_trajectory_creation(self):
        traj = Trajectory(
            id="traj_001",
            instance_id="django__django-12345",
            repo="django/django",
            task_type="bug_fix",
            success=True,
            total_steps=25,
            summary="Fixed import error in models.py",
        )
        assert traj.id == "traj_001"
        assert traj.instance_id == "django__django-12345"
        assert traj.success is True

    def test_trajectory_to_dict(self):
        traj = Trajectory(
            id="traj_001",
            instance_id="test",
            repo="test/repo",
            task_type="bug_fix",
            success=True,
            total_steps=10,
            summary="Test summary",
        )
        d = traj.to_dict()
        assert d["id"] == "traj_001"
        assert "embedding" in d
