"""
Complexity scoring using Information Theory.

Shannon Entropy + Compression Ratio to detect AI/SEO slop vs human writing.
The "Goldilocks Zone" - structured but unpredictable.
"""
import zlib
import math
from collections import Counter


def shannon_entropy(text: str) -> float:
    """
    Calculate Shannon entropy of text.
    H(X) = -sum(p(x) * log2(p(x)))
    
    Low entropy (~3-4): Repetitive, simple (SEO spam, code)
    Medium entropy (~4.5-5.5): Natural language (human writing)
    High entropy (~6+): Random/encrypted (gibberish spam)
    """
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def compression_ratio(text: str) -> float:
    """
    Approximate Kolmogorov complexity via compression.
    
    Low ratio (<0.3): Highly repetitive (SEO, keyword stuffing)
    Medium ratio (0.3-0.5): Normal content
    High ratio (>0.5): Dense, unique (literature, technical)
    """
    if not text or len(text) < 100:
        return 0.5
    compressed = zlib.compress(text.encode('utf-8'), level=6)
    return len(compressed) / len(text)


def word_entropy(text: str) -> float:
    """
    Shannon entropy at word level (vocabulary richness).
    """
    words = text.lower().split()
    if len(words) < 10:
        return 0.0
    counts = Counter(words)
    total = len(words)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def calculate_complexity_score(text: str) -> float:
    """
    Combined complexity score. Returns 0.0 to 1.0.
    
    Rewards the "Goldilocks Zone":
    - Not too repetitive (SEO)
    - Not too random (spam)
    - Rich vocabulary
    - Unique structure
    """
    if not text or len(text) < 200:
        return 0.5
    
    char_ent = shannon_entropy(text)
    comp_ratio = compression_ratio(text)
    word_ent = word_entropy(text)
    
    score = 0.0
    
    # Character entropy: reward natural language range
    if 4.0 < char_ent < 5.5:
        score += 0.3
    elif 3.5 < char_ent < 6.0:
        score += 0.15
    
    # Compression: reward low compressibility (unique content)
    if comp_ratio > 0.45:
        score += 0.35
    elif comp_ratio > 0.35:
        score += 0.2
    
    # Word entropy: reward rich vocabulary
    if word_ent > 8.0:
        score += 0.35
    elif word_ent > 6.0:
        score += 0.2
    
    return min(1.0, score)


def detect_repetition_patterns(text: str) -> float:
    """
    Detect SEO-style repetition (same phrases repeated).
    Returns penalty 0.0 (no repetition) to 1.0 (heavy repetition).
    """
    words = text.lower().split()
    if len(words) < 50:
        return 0.0
    
    # Check 3-gram repetition
    trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    counts = Counter(trigrams)
    
    if not counts:
        return 0.0
    
    # How many trigrams appear more than twice?
    repeated = sum(1 for c in counts.values() if c > 2)
    repetition_ratio = repeated / len(counts)
    
    return min(1.0, repetition_ratio * 3)  # Scale up
