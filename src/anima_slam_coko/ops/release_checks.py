"""Release validation gates for SLAM-COKO (Paper Section 4)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CheckResult:
    name: str
    passed: bool
    message: str
    value: float | None = None
    threshold: float | None = None


def check_depth_l1(value: float, max_threshold: float = 1.5) -> CheckResult:
    return CheckResult(
        name="depth_l1",
        passed=value <= max_threshold,
        message=f"Depth-L1 {value:.4f} {'<=' if value <= max_threshold else '>'} {max_threshold}",
        value=value,
        threshold=max_threshold,
    )


def check_bandwidth(per_agent_mb: float, max_mb: float = 150.0) -> CheckResult:
    return CheckResult(
        name="bandwidth_per_agent",
        passed=per_agent_mb <= max_mb,
        message=f"Bandwidth {per_agent_mb:.1f} MB/agent {'<=' if per_agent_mb <= max_mb else '>'} {max_mb}",
        value=per_agent_mb,
        threshold=max_mb,
    )


def check_loop_detection(loops_found: int, min_loops: int = 0) -> CheckResult:
    return CheckResult(
        name="loop_detection",
        passed=loops_found >= min_loops,
        message=f"Loops found: {loops_found} (min: {min_loops})",
        value=float(loops_found),
        threshold=float(min_loops),
    )


def check_gaussian_count(count: int, min_count: int = 1000) -> CheckResult:
    return CheckResult(
        name="gaussian_count",
        passed=count >= min_count,
        message=f"Gaussians: {count:,} (min: {min_count:,})",
        value=float(count),
        threshold=float(min_count),
    )


def check_submap_count(count: int, min_count: int = 2) -> CheckResult:
    return CheckResult(
        name="submap_count",
        passed=count >= min_count,
        message=f"Submaps: {count} (min: {min_count})",
        value=float(count),
        threshold=float(min_count),
    )


def validate_release(results: dict) -> list[CheckResult]:
    """Run all release checks on a training results dict."""
    checks = []
    fusion = results.get("fusion", {})
    evaluation = results.get("evaluation", {})

    checks.append(check_depth_l1(evaluation.get("depth_l1_mean", 999.0)))
    checks.append(check_bandwidth(fusion.get("bandwidth_per_agent_mb", 999.0)))
    checks.append(check_loop_detection(
        fusion.get("intra_loops", 0) + fusion.get("inter_loops", 0)
    ))
    checks.append(check_gaussian_count(fusion.get("post_refine_gaussians", 0)))
    checks.append(check_submap_count(fusion.get("total_submaps", 0)))

    return checks


def release_verdict(checks: list[CheckResult]) -> tuple[bool, str]:
    """Return (passed, summary_message) for all checks."""
    all_pass = all(c.passed for c in checks)
    lines = []
    for c in checks:
        icon = "PASS" if c.passed else "FAIL"
        lines.append(f"  [{icon}] {c.message}")
    summary = "\n".join(lines)
    verdict = "RELEASE APPROVED" if all_pass else "RELEASE BLOCKED"
    return all_pass, f"{verdict}\n{summary}"
