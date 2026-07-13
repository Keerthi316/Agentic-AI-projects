"""
Memory Extractor — Extract Important Facts from User Conversations.

This module analyzes user messages and assistant responses to detect
meaningful personal facts that should be persisted as long-term memories.

Detection patterns include:
- "My name is X" → name memory
- "I am interested in X" → interest memory
- "I prefer X" / "I like X" / "I don't like X" → preference memory
- "I speak X" / "I like X language" → language memory
- "I already know X" / "I am good at X" → skill memory
- "I am in first/second/third/fourth year" → year memory
- "My favorite branch is X" / "I am studying X" → branch memory
- Extracting key preferences from assistant responses about user choices
"""

import re
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Extraction Patterns
# ──────────────────────────────────────────────

# Pattern: "My name is X" or "I am X" (name)
NAME_PATTERNS = [
    r"(?:my\s+name\s+is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    r"(?:i\s+am\s+called)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    r"(?:call\s+me)\s+([A-Z][a-z]+)",
    r"(?:i'm\s+|i\s+am\s+)([A-Z][a-z]+)\b(?!\s+(?:interested|from|in|a|an|the|here|new|looking|student|first|second|third|fourth))",
]

# Pattern: "I am interested in X" or "my interest is X"
INTEREST_PATTERNS = [
    r"(?:i\s+am\s+interested\s+in)\s+(.+?)(?:\.|$|\?|\,|\sand\s)",
    r"(?:my\s+interest\s+is)\s+(.+?)(?:\.|$|\?|\,)",
    r"(?:i\s+(?:like|love)\s+(?:the\s+)?(?:field\s+of\s+)?)(.+?)(?:\.|$|\?|\,)",
]

# Pattern: "I prefer X" or "my preference is X"
PREFERENCE_PATTERNS = [
    r"(?:i\s+prefer)\s+(.+?)(?:\.|$|\?|\,)",
    r"(?:my\s+preference\s+is)\s+(.+?)(?:\.|$|\?|\,)",
    r"(?:i\s+would\s+like)\s+(.+?)(?:\.|$|\?|\,)",
]

# Pattern: "I like X" or "I don't like X"
LIKE_DISLIKE_PATTERNS = [
    r"(?:i\s+like)\s+(.+?)(?:\.|$|\?|\,)",
    r"(?:i\s+don't\s+like|i\s+do\s+not\s+like)\s+(.+?)(?:\.|$|\?|\,)",
]

# Pattern: "I speak X" / "I like X language"
LANGUAGE_PATTERNS = [
    r"(?:i\s+speak)\s+(\w+(?:\s+\w+)?)(?:\.|$|\?|\,)",
    r"(?:i\s+(?:like|prefer)\s+(\w+(?:\s+\w+)?)\s+language)",
    r"(?:my\s+(?:native|mother)\s+tongue\s+is)\s+(\w+)",
    r"(?:my\s+language\s+is)\s+(\w+)",
]

# Pattern: "I already know X" / "I am good at X" / "I have experience in X"
SKILL_PATTERNS = [
    r"(?:i\s+already\s+know)\s+(.+?)(?:\.|$|\?|\,)",
    r"(?:i\s+(?:am\s+)?(?:good\s+at|proficient\s+in|skilled\s+in))\s+(.+?)(?:\.|$|\?|\,)",
    r"(?:i\s+have\s+experience\s+in)\s+(.+?)(?:\.|$|\?|\,)",
    r"(?:i\s+(?:know|understand))\s+(.+?)(?:\.|$|\?|\,)",
]

# Pattern: "I am in X year" (year/level)
YEAR_PATTERNS = [
    r"(?:i\s+am\s+in)\s+(first|second|third|fourth|1st|2nd|3rd|4th)\s+(?:year|yr)",
    r"(?:i\s+am\s+a)\s+(first|second|third|fourth|1st|2nd|3rd|4th)\s+year\s+student",
    r"(?:my\s+(?:current\s+)?year\s+is)\s+(first|second|third|fourth|1st|2nd|3rd|4th)",
]

# Pattern: "My favorite branch is X" / "I am studying X"
BRANCH_PATTERNS = [
    r"(?:my\s+favorite\s+branch\s+is|my\s+branch\s+is)\s+(.+?)(?:\.|$|\?|\,)",
    r"(?:i\s+am\s+studying)\s+(.+?)(?:\.|$|\?|\,)",
    r"(?:i\s+am\s+(?:pursuing|doing))\s+(.+?)(?:\.|$|\?|\,)",
]

# Combined list of all extractors with their memory type
EXTRACTORS: List[Tuple[str, List[str], float]] = [
    # (memory_type, pattern_list, base_importance)
    ("name", NAME_PATTERNS, 0.95),
    ("interest", INTEREST_PATTERNS, 0.80),
    ("preference", PREFERENCE_PATTERNS, 0.75),
    ("preference", LIKE_DISLIKE_PATTERNS, 0.70),
    ("language", LANGUAGE_PATTERNS, 0.85),
    ("skill", SKILL_PATTERNS, 0.85),
    ("year", YEAR_PATTERNS, 0.90),
    ("branch", BRANCH_PATTERNS, 0.90),
]

# Stop words / phrases that should NOT be treated as memories
IGNORE_PHRASES = [
    "it",
    "this",
    "that",
    "these",
    "those",
    "them",
    "you",
    "your",
    "he",
    "she",
    "we",
    "they",
    "to",
    "for",
    "from",
    "with",
    "about",
    "the answer",
]


class MemoryExtractor:
    """
    Extracts meaningful personal facts from conversation text.

    Uses regex patterns to detect statements that contain
    long-term useful information about the user.
    """

    def extract_memories(
        self,
        user_message: str,
        assistant_response: str = "",
    ) -> List[Dict[str, any]]:
        """
        Extract memory entries from user messages and assistant responses.

        Args:
            user_message: The user's message text.
            assistant_response: The assistant's response text (optional).

        Returns:
            List[Dict]: List of memory dictionaries with keys:
                - memory_type (str): Type of memory
                - content (str): Extracted content value
                - importance (float): Importance score 0–1
        """
        memories = []

        # Extract from user message
        user_memories = self._extract_from_text(user_message)
        memories.extend(user_memories)

        # Extract from assistant response (if provided)
        if assistant_response:
            assistant_memories = self._extract_from_text(assistant_response)
            memories.extend(assistant_memories)

        # Filter out generic/ignored phrases
        memories = [m for m in memories if not self._is_ignored(m["content"])]

        # Deduplicate by memory_type + content
        seen = set()
        unique_memories = []
        for m in memories:
            key = f"{m['memory_type']}:{m['content'].lower().strip()}"
            if key not in seen:
                seen.add(key)
                unique_memories.append(m)

        if unique_memories:
            logger.info(f"Extracted {len(unique_memories)} memories from conversation")

        return unique_memories

    def _extract_from_text(self, text: str) -> List[Dict[str, any]]:
        """
        Run all extraction patterns against a text string.

        Args:
            text: The text to extract memories from.

        Returns:
            List[Dict]: Extracted memory entries.
        """
        memories = []

        for memory_type, patterns, importance in EXTRACTORS:
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    content = match.group(1).strip()

                    # Clean up the extracted content
                    content = self._clean_content(content)

                    if content and len(content) > 1:
                        # Adjust importance based on match quality
                        adjusted_importance = self._adjust_importance(
                            memory_type, content, importance
                        )

                        memories.append({
                            "memory_type": memory_type,
                            "content": content,
                            "importance": adjusted_importance,
                        })

        return memories

    def _clean_content(self, content: str) -> str:
        """
        Clean up extracted content by removing trailing junk.

        Args:
            content: Raw extracted content.

        Returns:
            str: Cleaned content string.
        """
        # Remove trailing punctuation
        content = re.sub(r'[.,!?;:]+$', '', content)

        # Remove trailing filler words
        filler_words = [
            r'\s+in\s+the$',
            r'\s+and$',
            r'\s+or$',
            r'\s+to$',
            r'\s+for$',
            r'\s+a$',
            r'\s+an$',
            r'\s+the$',
            r'\s+with$',
            r'\s+at$',
            r'\s+by$',
            r'\s+on$',
            r'\s+is$',
            r'\s+are$',
            r'\s+very$',
            r'\s+much$',
            r'\s+please$',
        ]
        for filler in filler_words:
            content = re.sub(filler, '', content)

        return content.strip()

    def _adjust_importance(
        self, memory_type: str, content: str, base_importance: float
    ) -> float:
        """
        Adjust the importance score based on content and context.

        Args:
            memory_type: Type of memory.
            content: Extracted content.
            base_importance: Default importance for this memory type.

        Returns:
            float: Adjusted importance score 0–1.
        """
        # Name memories are always high importance
        if memory_type == "name":
            return 0.95

        # Shorter content = less specific = lower importance
        if len(content) <= 2:
            return base_importance * 0.5

        # Capitalized content often indicates proper nouns (higher importance)
        if content[0].isupper():
            return min(1.0, base_importance + 0.1)

        return base_importance

    def _is_ignored(self, content: str) -> bool:
        """
        Check if extracted content should be ignored.

        Args:
            content: Extracted content to check.

        Returns:
            bool: True if this content should be ignored.
        """
        content_lower = content.lower().strip()
        return content_lower in [p.lower() for p in IGNORE_PHRASES]

    def detect_clear_data_command(self, user_message: str) -> bool:
        """
        Detect if the user is requesting to clear their data.

        Triggers on phrases like:
        - "clear my data"
        - "delete my data"
        - "forget me"
        - "remove my information"
        - "erase my memories"

        Args:
            user_message: The user's message.

        Returns:
            bool: True if the user wants their data cleared.
        """
        clear_patterns = [
            r"clear\s+my\s+data",
            r"delete\s+my\s+(data|information|memories|history)",
            r"forget\s+me",
            r"remove\s+my\s+(data|information|memories)",
            r"erase\s+my\s+(data|information|memories)",
            r"delete\s+all\s+(my\s+)?(data|information|memories)",
        ]

        text_lower = user_message.lower().strip()
        for pattern in clear_patterns:
            if re.search(pattern, text_lower):
                return True

        return False