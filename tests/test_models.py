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


class TestFragment:
    def test_fragment_creation(self):
        frag = Fragment(
            id="frag_001",
            step_range=(5, 12),
            fragment_type="error_recovery",
            description="Recovered from ImportError by fixing module path",
            action_sequence=["search", "open", "edit"],
            outcome="success",
        )
        assert frag.id == "frag_001"
        assert frag.step_range == (5, 12)
        assert frag.fragment_type == "error_recovery"

    def test_fragment_types_valid(self):
        valid_types = ["error_recovery", "exploration", "successful_fix", "failed_attempt", "loop"]
        for ft in valid_types:
            frag = Fragment(
                id=f"frag_{ft}",
                step_range=(0, 1),
                fragment_type=ft,
                description="test",
                action_sequence=[],
                outcome="test",
            )
            assert frag.fragment_type == ft


class TestState:
    def test_state_creation(self):
        state = State(
            tools=["bash", "search", "edit", "view"],
            repo_summary="Django web framework, Python 3.8+",
            task_description="Fix failing import in models.py",
            current_error="ImportError: cannot import name 'Model'",
            phase="fixing",
        )
        assert state.tools == ["bash", "search", "edit", "view"]
        assert state.phase == "fixing"
        assert "ImportError" in state.current_error

    def test_state_to_situation_string(self):
        state = State(
            tools=["bash", "edit"],
            repo_summary="Python web app",
            task_description="Fix bug",
            current_error="TypeError: NoneType",
            phase="locating",
        )
        situation = state.to_situation_string()
        assert "TypeError" in situation
        assert "locating" in situation

    def test_state_valid_phases(self):
        valid_phases = ["understanding", "locating", "fixing", "testing"]
        for phase in valid_phases:
            state = State(
                tools=["bash"],
                repo_summary="test",
                task_description="test",
                current_error="",
                phase=phase,
            )
            assert state.phase == phase

    def test_state_invalid_phase_raises(self):
        with pytest.raises(ValueError):
            State(
                tools=["bash"],
                repo_summary="test",
                task_description="test",
                current_error="",
                phase="invalid_phase",
            )
