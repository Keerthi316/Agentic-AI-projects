"""
Giskard integration package for the Multi-Agent Recruitment System.

Provides:
- GiskardScanner: runs both structural vulnerability checks and optional
  Giskard LLM-based scanning for prompt injection, hallucination, bias,
  excessive autonomy, and infinite loop vulnerabilities.
- GiskardScanResult: aggregated findings with severity classification.
- VulnerabilityFinding: individual finding with severity and remediation.

Degrades gracefully when giskard is not installed — structural checks
always run regardless.
"""

from .scanner import (
    GiskardScanner,
    GiskardScanResult,
    VulnerabilityFinding,
    VulnerabilitySeverity,
    VulnerabilityCategory,
    GISKARD_AVAILABLE,
)

__all__ = [
    "GiskardScanner",
    "GiskardScanResult",
    "VulnerabilityFinding",
    "VulnerabilitySeverity",
    "VulnerabilityCategory",
    "GISKARD_AVAILABLE",
]
