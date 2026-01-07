"""
Frequency-Aware Spell Checker (Server-Side)
Based on Peter Norvig's approach but weighted by database frequency.
"""
import re
from collections import Counter
from typing import Dict, Set

class SpellChecker:
    def __init__(self):
        self.vocab: Counter = Counter()
        self.total_words = 0
        
    def load_vocab(self, data: Dict[str, int]):
        """Load dictionary from DB: {'word': frequency}"""
        self.vocab = Counter(data)
        self.total_words = sum(self.vocab.values())
        print(f"📚 SpellChecker loaded {len(self.vocab)} words")

    def P(self, word: str) -> float:
        """Probability of word."""
        N = self.total_words or 1
        return self.vocab[word] / N

    def correction(self, word: str) -> str:
        """Most probable spelling correction for word."""
        # Prevent DoS: edits2() generates O(n^4) candidates
        # A 20-char word creates millions of candidates
        if len(word) > 15:
            return word
        
        # 1. Known word? (e.g. "Trump")
        if word in self.vocab:
            return word
            
        # 2. Distance 1 candidates (e.g. "Turmp" -> "Trump")
        candidates = self.candidates(word)
        
        # 3. Pick the candidate with the highest frequency in our index
        # This fixes the "Turmus" problem. "Trump" (freq: 5000) wins vs "Turmus" (freq: 5).
        best_word = max(candidates, key=self.P)
        
        # Only correct if the best candidate is actually known
        if self.vocab[best_word] > 0:
            return best_word
        return word

    def candidates(self, word: str) -> Set[str]:
        """Generate possible spelling corrections for word."""
        # Priority:
        # 1. Exact match
        # 2. Distance 1 edits
        # 3. Distance 2 edits
        # 4. Original word (give up)
        return (self.known([word]) or 
                self.known(self.edits1(word)) or 
                self.known(self.edits2(word)) or 
                {word})

    def known(self, words: Set[str]) -> Set[str]:
        """The subset of `words` that appear in the dictionary."""
        return set(w for w in words if w in self.vocab)

    def edits1(self, word: str) -> Set[str]:
        """All edits that are one edit away from `word`."""
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word: str) -> Set[str]:
        """All edits that are two edits away from `word`."""
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

# Global instance
spell_checker = SpellChecker()
