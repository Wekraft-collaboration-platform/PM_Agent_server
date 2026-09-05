import re
from typing import List

# Comprehensive list of standard, informal, and multilingual greetings
GREETINGS_LIST: List[str] = [
    "hi", "hello", "hey", "heyy", "heyyy", "hiii",
    "good morning", "good afternoon", "good evening", "good day", "good night",
    "greetings", "howdy", "sup", "yo", "yoo", "yooo",
    "namaste", "hola", "bonjour", "hallo", "ciao", "oi", "aloha",
    "whatsup", "whats up", "what's up", "how are you", "how r u",
    "hows it going", "how's it going", "how is it going",
    "hey there", "hi there", "hello there", "morning", "afternoon", "evening",
    "welcome", "ahoy", "salutations", "peace", "ssup"
]

# Compiled Regex patterns with word boundaries
GREETING_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b" + re.escape(g) + r"\b", re.IGNORECASE) for g in GREETINGS_LIST
]


def is_greeting_query(text: str) -> bool:
    """
    Central helper to check if a user prompt is a greeting or casual opener.
    Uses regex word boundaries and string inspection.
    """
    if not text or not text.strip():
        return False

    lowered = text.lower().strip()

    # Exact match or starts with greeting
    if lowered in GREETINGS_LIST:
        return True

    # Regex pattern check
    for pattern in GREETING_PATTERNS:
        if pattern.search(lowered):
            return True

    return False
