"""Analyze and compare evaluation results."""

from dataclasses import dataclass

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
