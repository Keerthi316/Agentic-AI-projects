"""
Promptfoo integration package for the Multi-Agent Recruitment System.

Generates promptfoo YAML configuration for multi-step red-team trajectory
attacks. Promptfoo is a Node.js tool (npm install -g promptfoo) that tests
adversarial prompt scenarios against the recruitment workflow.

Usage:
    from evaluation.promptfoo_suite import generate_promptfoo_config
    from evaluation.datasets.loader import load_dataset

    dataset = load_dataset()
    config = generate_promptfoo_config(dataset)
    # config written to evaluation/promptfoo_suite/promptfoo.yaml

    # Then run with:
    # promptfoo eval --config evaluation/promptfoo_suite/promptfoo.yaml
"""

from .config_generator import (
    generate_promptfoo_config,
    load_promptfoo_config,
    PromptfooConfig,
    PromptfooTestCase,
    PromptfooAssertion,
)

__all__ = [
    "generate_promptfoo_config",
    "load_promptfoo_config",
    "PromptfooConfig",
    "PromptfooTestCase",
    "PromptfooAssertion",
]
