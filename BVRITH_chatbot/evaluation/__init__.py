"""
evaluation/ — DeepEval-based evaluation suite for the BVRIT RAG Chatbot.

Modules:
    test_deepeval — pytest test file using DeepEval metrics:
                    HallucinationMetric, FaithfulnessMetric,
                    BiasMetric, ToxicityMetric, AnswerRelevancyMetric
    deepeval_runner — Programmatic runner (no pytest required) for
                      integration with the Streamlit governance tab
"""
