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
    # Matches "I'm Priya" / "I am Priya" but excludes words that are clearly
    # not names: prepositions, articles, pronouns, adjectives, verb gerunds
    # (planning, studying, pursuing, looking, considering, etc.)
    r"(?:i'm\s+|i\s+am\s+)([A-Z][a-z]+)\b(?!\s+(?:interested|from|in|a|an|the|here|new|looking|student|first|second|third|fourth|planning|studying|pursuing|doing|considering|going|trying|hoping|working|learning|applying|joining))",
]

# Sentence-end sentinel reused in all content-capturing patterns.
# \.(?!\w) matches a dot only when NOT followed by a word character, so
# abbreviations like "B.Tech" and "M.Sc." are captured whole rather than
# truncated at the first internal dot.
_END = r"(?:\.(?!\w)|$|\?|,|\sand\s)"

# Pattern: "I am interested in X" or "my interest is X"
INTEREST_PATTERNS = [
    r"(?:i\s+am\s+interested\s+in)\s+(.+?)" + _END,
    r"(?:my\s+interest\s+is)\s+(.+?)" + _END,
    r"(?:i\s+(?:like|love)\s+(?:the\s+)?(?:field\s+of\s+)?)(.+?)" + _END,
    r"(?:i\s+(?:want|wish)\s+to\s+(?:study|join|pursue|do|take))\s+(.+?)" + _END,
    r"(?:i\s+am\s+interested\s+in\s+(?:joining|pursuing|studying))\s+(.+?)" + _END,
    r"(?:i\s+am\s+planning\s+to\s+(?:study|join|pursue|do))\s+(.+?)" + _END,
    r"(?:i\s+(?:am\s+)?considering)\s+(.+?)" + _END,
]

# Pattern: "I prefer X" or "my preference is X"
PREFERENCE_PATTERNS = [
    r"(?:i\s+prefer)\s+(.+?)" + _END,
    r"(?:my\s+preference\s+is)\s+(.+?)" + _END,
    r"(?:i\s+would\s+like)\s+(.+?)" + _END,
    r"(?:i\s+am\s+looking\s+for)\s+(.+?)" + _END,
]

# Pattern: "I like X" or "I don't like X"
LIKE_DISLIKE_PATTERNS = [
    r"(?:i\s+like)\s+(.+?)" + _END,
    r"(?:i\s+don't\s+like|i\s+do\s+not\s+like)\s+(.+?)" + _END,
    r"(?:i\s+love)\s+(.+?)" + _END,
    r"(?:i\s+enjoy)\s+(.+?)" + _END,
]

# Pattern: "I speak X" / "I like X language"
LANGUAGE_PATTERNS = [
    r"(?:i\s+speak)\s+(\w+(?:\s+\w+)?)" + _END,
    r"(?:i\s+(?:like|prefer)\s+(\w+(?:\s+\w+)?)\s+language)",
    r"(?:my\s+(?:native|mother)\s+tongue\s+is)\s+(\w+)",
    r"(?:my\s+language\s+is)\s+(\w+)",
]

# Pattern: "I already know X" / "I am good at X" / "I have experience in X"
SKILL_PATTERNS = [
    r"(?:i\s+already\s+know)\s+(.+?)" + _END,
    r"(?:i\s+(?:am\s+)?(?:good\s+at|proficient\s+in|skilled\s+in))\s+(.+?)" + _END,
    r"(?:i\s+have\s+experience\s+in)\s+(.+?)" + _END,
    r"(?:i\s+(?:know|understand))\s+(.+?)" + _END,
]

# Pattern: "I am in X year" (year/level)
YEAR_PATTERNS = [
    r"(?:i\s+am\s+in)\s+(first|second|third|fourth|1st|2nd|3rd|4th)\s+(?:year|yr)",
    r"(?:i\s+am\s+a)\s+(first|second|third|fourth|1st|2nd|3rd|4th)\s+year\s+student",
    r"(?:my\s+(?:current\s+)?year\s+is)\s+(first|second|third|fourth|1st|2nd|3rd|4th)",
]

# Pattern: "My favorite branch is X" / "I am studying X"
BRANCH_PATTERNS = [
    r"(?:my\s+(?:favorite\s+)?branch\s+is)\s+(.+?)" + _END,
    r"(?:i\s+am\s+(?:studying|pursuing|doing|in))\s+(.+?)" + _END,
    r"(?:i\s+(?:want|wish)\s+to\s+(?:study|join|pursue|do|take))\s+(.+?)" + _END,
    r"(?:i\s+am\s+a)\s+(.+?)\s+student" + _END,
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

        Contractions are expanded first (e.g. "I'm" → "I am") so that all
        patterns written with "i\\s+am" also match colloquial input.

        Args:
            text: The text to extract memories from.

        Returns:
            List[Dict]: Extracted memory entries.
        """
        # Expand common contractions so patterns like `i\s+am` match "I'm" too
        text = self._expand_contractions(text)

        memories = []

        for memory_type, patterns, importance in EXTRACTORS:
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    content = match.group(1).strip()

                    # Name patterns use [A-Z][a-z]+ but IGNORECASE makes them
                    # also match lowercase words like "planning" or "studying".
                    # Guard: a captured name must start with an uppercase letter
                    # in the *original* (pre-lowercased) text.
                    if memory_type == "name" and not content[0:1].isupper():
                        continue

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

    def _expand_contractions(self, text: str) -> str:
        """
        Expand common English contractions so regex patterns match them.

        Without this, "I'm interested in X" never matches patterns written
        as ``i\\s+am\\s+interested`` because the apostrophe breaks the match.

        Args:
            text: Raw input text possibly containing contractions.

        Returns:
            str: Text with contractions replaced by their full forms.
        """
        contractions = {
            r"\bi'm\b":      "I am",
            r"\bi've\b":     "I have",
            r"\bi'll\b":     "I will",
            r"\bi'd\b":      "I would",
            r"\bi'm\b":      "I am",      # curly apostrophe variant handled below
            r"\bdon't\b":    "do not",
            r"\bdon\u2019t\b": "do not",  # Unicode right single quotation mark
            r"\bi\u2019m\b": "I am",
            r"\bi\u2019ve\b": "I have",
            r"\bi\u2019ll\b": "I will",
            r"\bi\u2019d\b":  "I would",
        }
        for pattern, replacement in contractions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

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