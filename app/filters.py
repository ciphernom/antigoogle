"""
Content Quality Filters - Spam, Slop, Quality Analysis
"""
import re
import math
import logging # <--- Added
from urllib.parse import urlparse
from collections import Counter
from typing import Tuple, Optional
from bs4 import BeautifulSoup
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
import statistics

from .config import get_settings, SPAM_TRUSTED_DOMAINS, TRUSTED_DOMAINS
from .database import SpamTraining

settings = get_settings()

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("filters")

# ============================================================
# AI SLOP DETECTOR
# ============================================================

class SlopDetector:
    """
    Advanced detector for AI-generated content (Slop).
    Returns a float score (0.0 to 1.0) where 1.0 is certain slop.
    """

    TIER_1_PHRASES = [
        r"as an ai\s*(language)?\s*model", r"regenerate response", r"my knowledge cutoff",
        r"cannot browse the internet", r"I do not have personal opinions",
        r"apologize for the confusion", r"text-davinci", r"gpt-3", r"gpt-4"
    ]
    
    TIER_2_PHRASES = {
        # Corporate/AI Speak
        "delve into": 0.9, "dive deep": 0.8, "let's dive in": 0.9, "unleash the power": 1.0,
        "unlock the potential": 1.0, "transformative": 0.7, "paradigm shift": 0.9,
        "tapestry": 0.9, "symphony": 0.8, "testament to": 0.9, "underscores the": 0.7,
        "crucial role": 0.9, "pivotal": 0.8, "nuanced": 0.7, "comprehensive guide": 0.9,
        "in today's digital age": 1.0, "rapidly evolving": 0.8, "ever-evolving": 0.8,
        "cutting-edge": 0.8, "seamlessly": 0.7, "game-changer": 0.9, "one-stop solution": 0.9,
        "take it to the next level": 0.9, "supercharge": 0.8, "elevate your": 0.8,
        "in conclusion": 0.5, "to summarize": 0.5, "ultimately": 0.5, "it is worth noting": 0.6,
        "without further ado": 0.7, "embarked on": 0.7, "fostering": 0.7, "harnessing": 0.7,
        
        # Hype & Marketing
        "empower your": 0.8, "in the realm of": 0.8, "journey through": 0.9,
        "leverage the power": 0.9, "ever-changing landscape": 1.0, "holistic approach": 0.8,
        "at its core": 0.7, "imperative for": 0.8, "unlock new avenues": 0.9, "future-proof": 0.8,
        "unparalleled": 0.9, "invaluable": 0.9, "embracing change": 0.8,
        
        # Modern LLM Patterns (2025-2026)
        "crucial": 0.6, "crucially": 0.7, "plays a crucial role": 0.9, "crucial role in shaping": 1.0,
        "play a significant role": 0.9, "significant role in shaping": 1.0,
        "it's critical to": 0.8, "critical to": 0.6,
        "showcasing": 0.8, "aligns": 0.7, "aims to explore": 0.8, "intricate": 0.8,
        "vibrant": 0.7, "dynamic": 0.7, "innovative": 0.7, "plethora": 0.8, "myriad": 0.8,
        "utilize": 0.7, "leverage": 0.7, "enhance": 0.7, "adept": 0.7, "actionable insights": 0.9,
        "a multitude of": 0.8, "a plethora of": 0.9, "a testament to": 0.9, "as such": 0.6,
        "at the end of the day": 0.7, "interplay": 0.8, "key": 0.5, "landscape": 0.6,
        "realm": 0.7, "delve": 0.8, "explore": 0.6, "emphasizing": 0.7, "enduring": 0.7,
        "garner": 0.7, "highlight": 0.6, "valuable": 0.6
    }
    
    TRANSITION_HEDGING = {
        "however": 0.5, "moreover": 0.5, "in addition": 0.6, "furthermore": 0.5,
        "may": 0.4, "could": 0.4, "often": 0.4, "potentially": 0.6, "typically": 0.5,
        "additionally": 0.5, "accordingly": 0.5, "arguably": 0.5, "certainly": 0.5,
        "consequently": 0.5, "hence": 0.5, "indeed": 0.5, "nevertheless": 0.5,
        "nonetheless": 0.5, "notwithstanding": 0.5, "thus": 0.5, "undoubtedly": 0.5,
        "as a result": 0.6, "at length": 0.6
    }
    
    FUNCTION_WORDS = {
        "the", "a", "an", "in", "of", "to", "and", "is", "that", "it", "for", "on", "with",
        "as", "at", "by", "this", "from", "but", "or", "be", "was", "are", "were", "which",
        "not", "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
        "should", "he", "she", "they", "we", "i", "you", "his", "her", "their", "our", "my", "your"
    }
    
    HYPE_TITLE_WORDS = {
        "ultimate": 0.8, "revolutionary": 0.9, "game-changing": 0.9, "essential": 0.7,
        "complete guide": 0.8, "everything you need": 0.9, "unlock": 0.7, "master": 0.7
    }

    def _get_sentence_stats(self, text: str) -> dict:
        """Calculate structural statistics"""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
        if not sentences:
            return {'variance': 0, 'burstiness': 0, 'lexical_div': 0, 'repetition': 0, 'count': 0}
        
        word_lengths = [len(s.split()) for s in sentences]
        variance = statistics.stdev(word_lengths) if len(word_lengths) > 1 else 0
        
        # Limit lexical diversity to first 1000 words (prevents punishing long articles)
        all_words = ' '.join(sentences[:50]).lower().split()
        lexical_div = len(set(all_words)) / len(all_words) if all_words else 0
        
        burstiness = variance / (statistics.mean(word_lengths) + 1e-5) if word_lengths else 0
        
        # Repetition of non-function words
        content_words = [w for w in all_words if w not in self.FUNCTION_WORDS]
        repetition = 0
        if content_words:
            word_freq = Counter(content_words)
            repetition = max(word_freq.values()) / len(content_words)
        
        return {
            'variance': variance,
            'burstiness': burstiness,
            'lexical_div': lexical_div,
            'repetition': repetition,
            'count': len(sentences)
        }

    def _analyze_punctuation(self, text: str) -> float:
        """Analyze punctuation patterns"""
        text_len = max(len(text), 1)
        dash_count = text.count('—')
        colon_count = text.count(':')
        ellipsis_count = text.count('...')
        semi_count = text.count(';')
        
        # Relaxed thresholds (1 per 150 chars is suspicious)
        dash_score = min(1.0, (dash_count * 150) / text_len)
        colon_score = min(1.0, (colon_count * 200) / text_len)
        ellipsis_score = min(1.0, (ellipsis_count * 300) / text_len)
        semi_score = min(1.0, (semi_count * 250) / text_len)
        
        return (dash_score + colon_score + ellipsis_score + semi_score) / 4

    def _analyze_style(self, text: str, lexical_div: float) -> float:
        """Analyze stylometry with academic protection"""
        words = text.lower().split()
        word_count = len(words)
        if word_count == 0: return 0.0
        
        # 1. Function Word Ratio
        func_count = sum(1 for w in words if w in self.FUNCTION_WORDS)
        func_ratio = func_count / word_count
        func_score = max(0.0, min(1.0, (0.35 - func_ratio) / 0.35))
        
        # 2. Transition Density (Protected)
        th_hits = sum(weight for phrase, weight in self.TRANSITION_HEDGING.items() if phrase in text.lower())
        th_density = th_hits / (word_count / 1000) if word_count > 0 else 0
        
        # If vocabulary is rich (>0.55), allow more transitions (likely academic/technical)
        tolerance = 6.0 if lexical_div < 0.55 else 12.0
        th_score = min(1.0, th_density / tolerance)
        
        return (func_score + th_score) / 2

    def _analyze_formatting(self, soup: BeautifulSoup) -> float:
        """Analyze HTML formatting"""
        if not soup: return 0.0
        score = 0.0
        
        # 1. Bolded Label Pattern
        list_items = soup.find_all('li')
        if list_items:
            sus_items = 0
            for li in list_items:
                start_bold = li.find(['strong', 'b'])
                if start_bold:
                    bold_text = start_bold.get_text(strip=True)
                    if not bold_text: continue
                    next_node = start_bold.next_sibling
                    after_text = next_node.string if next_node and isinstance(next_node.string, str) else ""
                    if bold_text.endswith(':') or after_text.strip().startswith(':'):
                        sus_items += 1
            if len(list_items) > 2:
                ratio = sus_items / len(list_items)
                if ratio > 0.4: score += 0.4
        
        # 2. Nested Lists
        nested_depth = 0
        for ul in soup.find_all(['ul', 'ol']):
            parents = len(ul.find_parents(['ul', 'ol']))
            nested_depth = max(nested_depth, parents)
        if nested_depth > 2: score += 0.3
        
        # 3. Heading Density
        headings = len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
        text_len = len(soup.get_text())
        if text_len > 0 and (headings / (text_len / 200)) > 1.2:
            score += 0.2
        
        return min(1.0, score)

    def _analyze_title(self, title: str) -> float:
        """Score title for hype"""
        if not title: return 0.0
        title_lower = title.lower()
        hits = sum(weight for phrase, weight in self.HYPE_TITLE_WORDS.items() if phrase in title_lower)
        if hits > 0:
            logger.info(f"Triggered Title Hype: {hits} for '{title}'") # LOG
        return min(1.0, hits / 2.0)

    def score(self, text: str, title: str = "", soup: BeautifulSoup = None) -> float:
        """
        Calculate final slop score. Returns float 0.0 - 1.0.
        """
        if len(text) < 200:
            return 0.0
        
        text_lower = text.lower()
        if not soup and '<' in text and '>' in text:
            soup = BeautifulSoup(text, 'html.parser')
        
        # Tier 1: Immediate Fail
        for p in self.TIER_1_PHRASES:
            if re.search(p, text_lower):
                logger.warning(f"🚫 Slop Tier 1 Match: '{p}'") # LOG
                return 1.0
        
        # Tier 2: Phrase Analysis
        word_count = len(text.split())
        t2_hits = 0.0
        triggered_phrases = [] # LOG
        for phrase, weight in self.TIER_2_PHRASES.items():
            if phrase in text_lower:
                t2_hits += weight
                triggered_phrases.append(phrase)

        phrase_density = t2_hits / (word_count / 1000) if word_count > 0 else 0
        phrase_score = min(1.0, phrase_density / 5.0)
        
        if triggered_phrases:
            logger.info(f"Slop Phrases Found: {triggered_phrases} (Score contribution: {phrase_score:.2f})") # LOG
        
        # Structural Analysis
        stats = self._get_sentence_stats(text)
        variance_score = 0.0
        if stats['count'] > 5:
            var_part = max(0.0, min(1.0, 1.0 - (stats['variance'] / 8)))
            burst_part = max(0.0, min(1.0, 1.0 - (stats['burstiness'] / 0.4)))
            lex_part = max(0.0, min(1.0, 1.0 - (stats['lexical_div'] / 0.4)))
            variance_score = (var_part + burst_part + lex_part) / 3
        
        # Other Components
        punct_score = self._analyze_punctuation(text)
        style_score = self._analyze_style(text, stats.get('lexical_div', 0.5))
        fmt_score = self._analyze_formatting(soup) if soup else 0.0
        title_score = self._analyze_title(title)
        
        # Final Ensemble
        final_score = (
            phrase_score * 0.25 +
            punct_score * 0.10 +
            style_score * 0.15 +
            variance_score * 0.20 +
            fmt_score * 0.20 +
            title_score * 0.10
        )
        
        logger.info(f"🧠 Slop Breakdown: Final={final_score:.2f} | Phrase={phrase_score:.2f} Punct={punct_score:.2f} Style={style_score:.2f} Var={variance_score:.2f} Fmt={fmt_score:.2f}") # LOG
        
        return min(1.0, final_score)

# ============================================================
# SPAM FILTER (Naive Bayes)
# ============================================================
class SpamFilter:
    """
    Naive Bayes spam filter with trusted domain bypass.
    Training data stored in database for persistence.
    """
    
    # Bootstrap spam indicators
    SPAM_WORDS = [
        'casino', 'viagra', 'cialis', 'crypto', 'nft', 'lottery', 'winner', 'prize',
        'buy now', 'cheap', 'free money', 'porn', 'xxx', 'dating', 'singles', 'hookup',
        'pills', 'weight loss', 'make money', 'work from home', 'bitcoin profit',
        'forex', 'binary options', 'click here', 'act now', 'limited time',
    ]
    
    # Bootstrap ham indicators - EXPANDED
    HAM_WORDS = [
        # -- ORIGINAL TECH WORDS --
        'github', 'stackoverflow', 'wikipedia', 'arxiv', 'documentation', 'docs',
        'tutorial', 'guide', 'python', 'javascript', 'rust', 'golang', 'java',
        'api', 'reference', 'manual', 'research', 'paper', 'study', 'analysis',
        'open source', 'library', 'framework', 'function', 'class', 'method',
        'example', 'code', 'programming', 'developer', 'software', 'engineering',
        'algorithm', 'data structure', 'database', 'server', 'client', 'http',
        
        # -- NEW GENERAL ENGLISH WORDS (Fix for False Positives) --
        'news', 'report', 'today', 'world', 'business', 'company', 'market',
        'government', 'official', 'state', 'public', 'policy', 'department',
        'said', 'says', 'year', 'time', 'people', 'life', 'day', 'work',
        'new', 'good', 'first', 'last', 'long', 'great', 'little', 'own',
        'other', 'old', 'right', 'big', 'high', 'different', 'small', 'large',
        'next', 'early', 'young', 'important', 'few', 'public', 'bad', 'same',
        'able', 'group', 'place', 'problem', 'case', 'week', 'company', 'system',
        'program', 'question', 'work', 'government', 'number', 'night', 'point',
        'home', 'water', 'room', 'mother', 'area', 'money', 'story', 'fact',
        'month', 'lot', 'study', 'book', 'eye', 'job', 'word', 'business',
        'issue', 'side', 'kind', 'head', 'house', 'service', 'friend', 'father',
        'power', 'hour', 'game', 'line', 'end', 'member', 'law', 'car',
        'city', 'community', 'name', 'president', 'team', 'minute', 'idea',
        'kid', 'body', 'information', 'back', 'parent', 'face', 'others',
        'level', 'office', 'door', 'health', 'person', 'art', 'war',
        'history', 'party', 'result', 'change', 'morning', 'reason', 'research',
        'girl', 'guy', 'food', 'moment', 'air', 'teacher', 'force', 'education'
    ]
    
    # Spam TLDs
    SPAM_TLDS = {'.xyz', '.top', '.click', '.loan', '.work', '.gq', '.ml', '.tk', '.cf', '.ga'}
    
    def __init__(self):
        self.spam_counts = Counter()
        self.ham_counts = Counter()
        self.total_spam = 1
        self.total_ham = 50  # Give ham a head start
        self._bootstrap()
    
    def _bootstrap(self):
        """Initialize with seed data"""
        for w in self.SPAM_WORDS:
            self.spam_counts[w] += 50
        for w in self.HAM_WORDS:
            self.ham_counts[w] += 50
    
    async def load_from_db(self, db: AsyncSession):
        """Load training data from database"""
        try:
            result = await db.execute(select(SpamTraining))
            for row in result.scalars():
                self.spam_counts[row.token] = row.spam_count
                self.ham_counts[row.token] = row.ham_count
            logger.info("Loaded spam training data from DB")
        except Exception as e:
            logger.warning(f"Could not load spam data: {e}")
    
    async def save_to_db(self, db: AsyncSession, token: str, is_spam: bool):
        """Save training update to database"""
        from sqlalchemy.dialects.postgresql import insert
        
        stmt = insert(SpamTraining).values(
            token=token,
            spam_count=1 if is_spam else 0,
            ham_count=0 if is_spam else 1
        ).on_conflict_do_update(
            index_elements=['token'],
            set_={
                'spam_count': SpamTraining.spam_count + (1 if is_spam else 0),
                'ham_count': SpamTraining.ham_count + (0 if is_spam else 1)
            }
        )
        await db.execute(stmt)
    
    def _tokenize(self, url: str, title: str = '', text: str = '') -> list[str]:
        """Tokenize content for classification"""
        s = f"{url} {title} {text[:1000]}".lower()
        domain = urlparse(url).netloc.lower()
        tld = '.' + domain.split('.')[-1] if '.' in domain else ''
        tokens = re.findall(r'\b[a-z0-9]+\b', s)
        tokens.extend([domain, tld])
        return tokens
    
    def predict(self, url: str, title: str = '', text: str = '') -> Tuple[bool, float]:
        """
        Predict if content is spam.
        
        Returns:
            (is_spam, spam_probability)
        """
        # Trusted domains bypass
        domain = urlparse(url).netloc.lower().replace('www.', '')
        # Check BOTH trusted lists
        if domain in SPAM_TRUSTED_DOMAINS or domain in TRUSTED_DOMAINS:
            return False, 0.0
        if any(domain.endswith(f".{t}") for t in SPAM_TRUSTED_DOMAINS) or any(domain.endswith(f".{t}") for t in TRUSTED_DOMAINS):
            return False, 0.0
        
        # Check spam TLDs
        tld = '.' + domain.split('.')[-1] if '.' in domain else ''
        if tld in self.SPAM_TLDS:
            logger.warning(f"🚫 Spam TLD detected: {tld}") # LOG
            return True, 0.9
        
        # Naive Bayes
        tokens = self._tokenize(url, title, text)
        vocab_size = len(set(self.spam_counts.keys()) | set(self.ham_counts.keys())) + 1
        
        log_spam = math.log(self.total_spam / (self.total_spam + self.total_ham))
        log_ham = math.log(self.total_ham / (self.total_spam + self.total_ham))
        
        total_spam_words = sum(self.spam_counts.values()) + vocab_size
        total_ham_words = sum(self.ham_counts.values()) + vocab_size
        
        for token in tokens:
            log_spam += math.log((self.spam_counts.get(token, 0) + 1) / total_spam_words)
            log_ham += math.log((self.ham_counts.get(token, 0) + 1) / total_ham_words)
        
        # Softmax
        max_log = max(log_spam, log_ham)
        p_spam = math.exp(log_spam - max_log) / (math.exp(log_spam - max_log) + math.exp(log_ham - max_log))
        
        logger.info(f"📧 Spam Prob: {p_spam:.4f} (Threshold: {settings.SPAM_THRESHOLD}) URL: {url}") # LOG
        
        # Log specific triggers if spam is high
        if p_spam > settings.SPAM_THRESHOLD:
            # Simple check to see what triggered it
            triggers = [t for t in tokens if t in self.SPAM_WORDS]
            if triggers:
                logger.warning(f"  -> Triggers found: {triggers}")
        
        return p_spam > settings.SPAM_THRESHOLD, p_spam
    
    async def train(self, db: AsyncSession, url: str, is_spam: bool, title: str = '', text: str = ''):
        """Update model with new example and persist to database"""
        tokens = self._tokenize(url, title, text)
        if is_spam:
            for t in tokens:
                self.spam_counts[t] += 1
                await self.save_to_db(db, t, is_spam=True)
            self.total_spam += 1
        else:
            for t in tokens:
                self.ham_counts[t] += 1
                await self.save_to_db(db, t, is_spam=False)
            self.total_ham += 1
        await db.commit()

# ============================================================
# QUALITY ANALYZER
# ============================================================
class QualityAnalyzer:
    """Analyzes page quality based on content and structure"""
    
    def analyze(self, soup: BeautifulSoup, text: str, url: str) -> dict:
        """
        Analyze page quality.
        
        Returns dict with quality metrics.
        """
        domain = urlparse(url).netloc.lower().replace('www.', '')
        
        # Domain trust
        domain_trust = TRUSTED_DOMAINS.get(domain, 0.5)
        for trusted in TRUSTED_DOMAINS:
            if domain.endswith(f".{trusted}"):
                domain_trust = max(domain_trust, TRUSTED_DOMAINS[trusted] * 0.9)
        
        # Content metrics
        word_count = len(text.split())
        
        # Structure quality
        has_headings = bool(soup.find(['h1', 'h2', 'h3']))
        has_paragraphs = len(soup.find_all('p')) > 2
        has_lists = bool(soup.find(['ul', 'ol']))
        has_code = bool(soup.find(['code', 'pre']))
        
        # Negative signals
        ad_indicators = ['advertisement', 'sponsored', 'affiliate', 'buy now', 'click here']
        ad_score = sum(1 for ad in ad_indicators if ad in text.lower()) / len(ad_indicators)
        
        # Link density (Relaxed logic)
        links = soup.find_all('a')
        # Only penalize if density is VERY high (>20% instead of >10%)
        # and reduce the max penalty from 0.5 to 0.3
        link_density = len(links) / max(word_count, 1) * 100
        link_penalty = min(link_density / 50, 0.3) if link_density > 20 else 0
        
        return {
            'domain_trust': domain_trust,
            'word_count': word_count,
            'has_headings': has_headings,
            'has_paragraphs': has_paragraphs,
            'has_lists': has_lists,
            'has_code': has_code,
            'ad_score': ad_score,
            'link_penalty': link_penalty,
        }
    
    def compute_score(self, metrics: dict, slop_score: float) -> float:
        """
        Compute final quality score from metrics.
        
        Returns:
            Quality score 0.0 - 1.0
        """
        score = 0.5
        
        # Domain trust (high weight)
        score += (metrics['domain_trust'] - 0.5) * 0.4
        
        # Content structure
        if metrics['has_headings']:
            score += 0.05
        if metrics['has_paragraphs']:
            score += 0.05
        if metrics['has_code']:
            score += 0.1  # Code = usually technical content
        
        # Content length (log scale)
        if metrics['word_count'] > 300:
            score += min(math.log(metrics['word_count'] / 300) * 0.05, 0.15)
        
        # Penalties
        score -= metrics['ad_score'] * 0.2
        score -= metrics['link_penalty']
        score -= slop_score * 0.3
        
        final_score = max(0.0, min(1.0, score))
        logger.info(f"📉 Quality Score: {final_score:.2f} (Metrics: {metrics})") # LOG
        
        return final_score

# ============================================================
# TAG EXTRACTION
# ============================================================
def extract_tags(soup: BeautifulSoup, url: str) -> list[str]:
    """Extract category tags from page"""
    tags = []
    domain = urlparse(url).netloc.lower()
    
    # Domain-based tags
    if 'arxiv' in domain:
        tags.append('research')
    if 'github' in domain:
        tags.append('code')
    if any(x in domain for x in ['docs.', 'documentation', 'doc.', 'developer.']):
        tags.append('docs')
    if any(x in domain for x in ['news.ycombinator', 'lobste.rs']):
        tags.append('forum')
    if 'blog' in url or 'blog' in domain:
        tags.append('blog')
    
    # Content-based tags
    text = soup.get_text().lower()
    if any(x in text for x in ['python', 'javascript', 'rust', 'golang', 'java ']):
        tags.append('programming')
    if any(x in text for x in ['machine learning', 'neural network', 'deep learning', 'ai ']):
        tags.append('ml')
    if any(x in text for x in ['tutorial', 'guide', 'how to', 'getting started']):
        tags.append('tutorial')
    
    return list(set(tags))[:5]  # Max 5 tags

# ============================================================
# TITLE NORMALIZATION (for dedup)
# ============================================================
def normalize_title(title: str) -> str:
    """Normalize title for deduplication"""
    t = title.lower().strip()
    # Remove arxiv IDs like [2510.07578] or [2510.07578v1]
    t = re.sub(r'\[\d{4}\.\d{4,5}(v\d+)?\]\s*', '', t)
    # Remove version suffixes
    t = re.sub(r'\s*v\d+\s*$', '', t)
    # Remove common suffixes
    t = re.sub(r'\s*[-|·]\s*(arxiv|github|medium).*$', '', t)
    return t.strip()

# Global instances
slop_detector = SlopDetector()
spam_filter = SpamFilter()
quality_analyzer = QualityAnalyzer()
