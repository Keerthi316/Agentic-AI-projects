"""Test script to debug retrieval scores."""
import sys
sys.path.insert(0, '.')
from vector_store import get_vector_store, retrieve_documents
from utils import get_config
import os

os.environ['OPENROUTER_API_KEY'] = get_config()['OPENROUTER_API_KEY']

vs, status = get_vector_store()
print(f"Status: {status}")

queries = [
    "who is the principal of this college",
    "what is the fee structure for btech",
    "how many departments are there",
    "tell me about placements",
]

for q in queries:
    docs, score = retrieve_documents(vs, q, k=8)
    print(f"\nQuery: {q}")
    print(f"  Max score: {score:.4f}")
    for d in docs[:3]:
        dist = d.metadata.get("distance", "?")
        rel = d.metadata.get("relevance_score", "?")
        section = d.metadata.get("section", "?")[:60]
        print(f"  dist={dist} score={rel} section=[{section}]")