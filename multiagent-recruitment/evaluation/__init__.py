"""
Evaluation framework for the Multi-Agent Recruitment System.

This package provides a complete, standalone evaluation pipeline that
measures quality, correctness, safety, and reliability of the recruitment
workflow WITHOUT modifying the workflow itself.

Layers:
    - traces/       Captures and validates LangGraph execution traces
    - metrics/      Implements evaluation metrics (DeepEval, custom)
    - datasets/     Reusable evaluation dataset + schema + loader
    - tests/        pytest test suites (trace, tool-call, output, red-team)
    - reports/      Report generation and persistence
    - deepeval/     DeepEval metric definitions and test cases
    - giskard/      Giskard vulnerability scan integration
    - promptfoo/    Promptfoo red-team configuration

Usage:
    # Run the full evaluation pipeline
    python evaluation/run_evaluation.py

    # Run individual test suites
    pytest evaluation/tests/ -v
"""

__version__ = "0.1.0"
