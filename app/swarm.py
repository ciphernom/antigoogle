"""
Swarm Ingestion - Handles incoming events from the Nostr network

Processes:
- URL Discovery events (4242) -> Add to crawl queue
- Crawl Result events (4243) -> Add to local index
- Vote Signal events (4244) -> Update ratings
- Genesis Config events (4245) -> Update shared config

Uses PoW (same as web UI) for anti-spam instead of VDF.
"""
import asyncio
import hashlib
import logging
import secrets
from typing import Optional
from collections import Counter
from urllib.parse import urlparse
from datetime import datetime
from sqlalchemy import select, update,  func, text

from sqlalchemy.dialects.postgresql import insert

from .config import get_settings, BLOCKED_DOMAINS, TRUSTED_DOMAINS
from .database import async_session, Page, Vocabulary, CrawlQueue 
from .nostr import (
    NostrService, EventKind, 
    URLDiscoveryEvent, CrawlResultEvent, VoteSignalEvent, GenesisConfigEvent
)
from .vrf import get_lottery_manager
from .ratings import BayesianRatings
from .embedder import get_embedder
settings = get_settings()
logger = logging.getLogger("swarm")


# ============================================================
# POW FUNCTIONS (same as api.py)
# ============================================================
def fnv1a_hash(challenge: str, nonce: int) -> int:
    """FNV-1a hash matching client implementation"""
    h = 2166136261
    for b in challenge.encode():
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    for i in range(4):
        b = (nonce >> (i * 8)) & 0xFF
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def count_leading_zero_hex(h: int) -> int:
    """Count leading zero hex digits"""
    if h == 0:
        return 8
    z = 0
    while z < 8 and ((h >> (28 - z * 4)) & 0xF) == 0:
        z += 1
    return z


def solve_pow(challenge: str, difficulty: int) -> int:
    """Solve PoW challenge - find nonce with required leading zeros"""
    nonce = 0
    while True:
        h = fnv1a_hash(challenge, nonce)
        if count_leading_zero_hex(h) >= difficulty:
            return nonce
        nonce += 1
        if nonce > 100_000_000:  # Safety limit
            raise RuntimeError("PoW solve exceeded max iterations")


def verify_pow(challenge: str, nonce: int, difficulty: int) -> bool:
    """Verify a PoW solution"""
    h = fnv1a_hash(challenge, nonce)
    return count_leading_zero_hex(h) >= difficulty


# ============================================================
# TOKENIZER
# ============================================================
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
}

def tokenize_content(text: str) -> list[str]:
    """Tokenize content for spell check vocabulary"""
    import re
    words = re.findall(r'\b[a-z0-9]+\b', text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def get_url_hash(url: str) -> str:
    """Get hash of normalized URL"""
    url = url.split('#')[0].rstrip('/')
    parsed = urlparse(url)
    normalized = f"{parsed.scheme}://{parsed.netloc.lower()}{parsed.path}"
    return hashlib.md5(normalized.encode()).hexdigest()


def get_title_hash(domain: str, title: str) -> str:
    """Get hash of normalized title for dedup"""
    import re
    t = title.lower().strip()
    t = re.sub(r'\[\d{4}\.\d{4,5}(v\d+)?\]\s*', '', t)
    t = re.sub(r'\s*v\d+\s*$', '', t)
    t = re.sub(r'\s*[-|·]\s*(arxiv|github|medium).*$', '', t)
    return hashlib.md5(f"{domain}:{t}".encode()).hexdigest()


# ============================================================
# URL DISCOVERY HANDLER (Kind 4242)
# ============================================================
async def handle_url_discovery(event: dict):
    """Handle incoming URL discovery event with PoW validation."""
    try:
        content = URLDiscoveryEvent.from_content(event['content'])
    except Exception as e:
        logger.warning(f"Invalid URL discovery event: {e}")
        return
    
    url = content.url
    domain = content.domain
    
    if domain in BLOCKED_DOMAINS:
        return
    
    # PoW verification
    if content.pow_challenge and content.pow_nonce is not None:
        if not verify_pow(content.pow_challenge, content.pow_nonce, settings.SWARM_POW_DIFFICULTY):
            logger.warning(f"Invalid PoW for {url}")
            return
    else:
        logger.debug(f"Missing PoW for {url}")
        return
    
    # VRF lottery check
    lottery = get_lottery_manager()
    should_crawl, ticket = lottery.should_crawl(domain, redundancy=settings.VRF_REDUNDANCY)
    
    if not should_crawl:
        logger.debug(f"Lost lottery for {domain}")
        return
    
    async with async_session() as db:
        url_hash = get_url_hash(url)
        result = await db.execute(select(Page.id).where(Page.url_hash == url_hash))
        if result.scalar_one_or_none():
            return
        
        result = await db.execute(select(CrawlQueue.id).where(CrawlQueue.url == url))
        if result.scalar_one_or_none():
            return
        
        trust = TRUSTED_DOMAINS.get(domain, 0.5)
        priority = content.priority * trust
        
        db.add(CrawlQueue(url=url, priority=priority))
        try:
            await db.commit()
            logger.info(f"📥 Queued from swarm: {url[:60]}")
        except Exception:
            await db.rollback()


# ============================================================
# CRAWL RESULT HANDLER (Kind 4243)
# ============================================================
async def handle_crawl_result(event: dict):
    """
    Handle incoming crawl result (lightweight - no embedding).
    We re-embed locally from title + description and update search vectors.
    """
    try:
        content = CrawlResultEvent.from_content(event['content'])
    except Exception as e:
        logger.warning(f"Invalid crawl result event: {e}")
        return
    
    domain = content.domain
    
    if domain in BLOCKED_DOMAINS:
        return
    
    if content.quality_score < settings.MIN_QUALITY:
        return
    
    if content.slop_score > settings.SLOP_THRESHOLD:
        return
    
    async with async_session() as db:
        result = await db.execute(select(Page.id).where(Page.url_hash == content.url_hash))
        if result.scalar_one_or_none():
            return
        
        title_hash = get_title_hash(domain, content.title)
        result = await db.execute(select(Page.id).where(Page.title_hash == title_hash))
        if result.scalar_one_or_none():
            return
        
        # USE PROVIDED EMBEDDING OR FALLBACK
        embedding = None
        
        # 1. Try using the vector from the event (Best Quality)
        if content.embedding and len(content.embedding) == settings.STORED_DIM:
            embedding = content.embedding
        else:
            # 2. Fallback: Re-embed locally (Lower Quality, only title/desc)
            try:
                from .embedder import get_embedder
                embedder = await get_embedder()
                text_for_embed = f"{content.title}. {content.description}"
                embedding_np = await embedder.encode_async(text_for_embed, reduce=False)
                embedding = embedding_np.tolist()
            except Exception as e:
                logger.warning(f"Re-embedding failed: {e}")
                return
        
        # Ensure dimension matches storage
        # This will now PASS because len(embedding) is 384 and settings.STORED_DIM is 384
        if len(embedding) != settings.STORED_DIM:
            logger.warning(f"Wrong embedding dim: {len(embedding)}")
            return
        
        # 1. Insert Page
        page = Page(
            url=content.url,
            url_hash=content.url_hash,
            title_hash=title_hash,
            title=content.title,
            description=content.description,
            domain=domain,
            quality_score=content.quality_score,
            slop_score=content.slop_score,
            spam_score=content.spam_score,
            domain_trust=TRUSTED_DOMAINS.get(domain, 0.5),
            content_length=0,
            word_count=content.word_count,
            embedding=embedding.tolist(), # Convert numpy to list for pgvector
            tags=','.join(content.tags),
        )
        db.add(page)
        await db.flush()
        
        # 2. Update Search Vector (Postgres Native)
        # Note: Swarm results don't have full body text, so we index Title+Description
        await db.execute(
            text("""
                UPDATE pages SET 
                    updated_at = :now,
                    search_vector = 
                        setweight(to_tsvector('english', :title), 'A'::"char") ||
                        setweight(to_tsvector('english', :body), 'B'::"char")
            WHERE id = :page_id
            """).bindparams(
                now=datetime.utcnow(),
                title=content.title,
                body=content.description or "",
                page_id=page.id
            )
        )
        
        # 3. Update Vocabulary (Upsert)
        vocab_text = f"{content.title} {content.description}" 
        terms = tokenize_content(vocab_text)                  
        unique_words = set(terms)
        
        if unique_words:
            insert_stmt = insert(Vocabulary).values(
                [{'word': w, 'doc_count': 1} for w in unique_words]
            )
            do_update_stmt = insert_stmt.on_conflict_do_update(
                index_elements=['word'],
                set_={'doc_count': Vocabulary.doc_count + 1}
            )
            await db.execute(do_update_stmt)
        
        await db.commit()
        logger.info(f"📥 Indexed from swarm: {content.title[:50]}")


# ============================================================
# VOTE SIGNAL HANDLER (Kind 4244)
# ============================================================
async def handle_vote_signal(event: dict):
    """Handle incoming vote signal with PoW validation."""
    pubkey = event['pubkey']
    
    try:
        content = VoteSignalEvent.from_content(event['content'])
    except Exception as e:
        logger.warning(f"Invalid vote signal event: {e}")
        return
    
    # PoW verification
    if content.pow_challenge and content.pow_nonce is not None:
        if not verify_pow(content.pow_challenge, content.pow_nonce, settings.SWARM_POW_DIFFICULTY):
            return
    else:
        return
    
    async with async_session() as db:
        result = await db.execute(select(Page.id).where(Page.url_hash == content.url_hash))
        page_id = result.scalar_one_or_none()
        
        if not page_id:
            return
        
        fingerprint = pubkey[:16]
        stats = await BayesianRatings.record_vote(db, page_id, content.is_good, fingerprint, weight=0.5)
        
        if stats.get('recorded'):
            logger.info(f"📥 Vote from swarm: {'👍' if content.is_good else '👎'}")


# ============================================================
# GENESIS CONFIG HANDLER (Kind 4245)
# ============================================================
_genesis_config: Optional[GenesisConfigEvent] = None

async def handle_genesis_config(event: dict):
    """Handle genesis configuration event."""
    global _genesis_config
    
    try:
        content = GenesisConfigEvent.from_content(event['content'])
    except Exception as e:
        logger.warning(f"Invalid genesis config event: {e}")
        return
    
    if _genesis_config and content.version <= _genesis_config.version:
        return
    
    _genesis_config = content
    logger.info(f"📥 New genesis config v{content.version}")
    
    #update the LSH planes
    try:
        embedder = await get_embedder()
        embedder.set_lsh_planes(
            content.lsh_l1_planes,
            content.lsh_l2_planes
        )
    except Exception as e:
        logger.error(f"Failed to apply genesis LSH config: {e}")


def get_genesis_config() -> Optional[GenesisConfigEvent]:
    return _genesis_config


# ============================================================
# SWARM CLASSES
# ============================================================
class SwarmIngestion:
    """Registers handlers with NostrService."""
    
    def __init__(self, nostr: NostrService):
        self.nostr = nostr
        self._stats = {
            'discoveries_received': 0,
            'results_received': 0,
            'votes_received': 0,
        }
    
    def register_handlers(self):
        @self.nostr.on_event(EventKind.URL_DISCOVERY)
        async def on_discovery(event):
            self._stats['discoveries_received'] += 1
            if settings.SWARM_ACCEPT_EXTERNAL:
                await handle_url_discovery(event)
        
        @self.nostr.on_event(EventKind.CRAWL_RESULT)
        async def on_result(event):
            self._stats['results_received'] += 1
            if settings.SWARM_ACCEPT_EXTERNAL:
                await handle_crawl_result(event)
        
        @self.nostr.on_event(EventKind.VOTE_SIGNAL)
        async def on_vote(event):
            self._stats['votes_received'] += 1
            await handle_vote_signal(event)
        
        @self.nostr.on_event(EventKind.GENESIS_CONFIG)
        async def on_genesis(event):
            await handle_genesis_config(event)
        
        logger.info("✅ Swarm handlers registered")
    
    def get_stats(self) -> dict:
        return self._stats.copy()


class SwarmPublisher:
    """Publishes events to swarm with PoW anti-spam."""
    
    def __init__(self, nostr: NostrService):
        self.nostr = nostr
        self._stats = {
            'discoveries_published': 0,
            'results_published': 0,
            'votes_published': 0,
        }
    
    async def _make_pow(self) -> tuple[str, int]:
        """Generate challenge and solve PoW without blocking event loop"""
        challenge = secrets.token_hex(32)
        # Run CPU-bound PoW in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        nonce = await loop.run_in_executor(None, solve_pow, challenge, settings.SWARM_POW_DIFFICULTY)
        return challenge, nonce
    
    async def publish_discovery(self, url: str, priority: float = 0.5, source_url: str = None) -> bool:
        if not settings.SWARM_PUBLISH_DISCOVERY:
            return False
        
        domain = urlparse(url).netloc.lower().replace('www.', '')
        challenge, nonce = await self._make_pow()  # Now async
        
        event = URLDiscoveryEvent(
            url=url,
            domain=domain,
            priority=priority,
            source_url=source_url,
            pow_challenge=challenge,
            pow_nonce=nonce,
        )
        
        if await self.nostr.publish_discovery(event):
            self._stats['discoveries_published'] += 1
            return True
        return False
    
    async def publish_result(
        self,
        url: str,
        title: str,
        description: str,
        quality_score: float,
        slop_score: float,
        spam_score: float,
        word_count: int,
        tags: list[str],
        embedding: list[float] = None,
        experts: dict = None,
    ) -> bool:
        """Publish crawl result to swarm (LIGHTWEIGHT - no embedding)."""
        if not settings.SWARM_PUBLISH_RESULTS:
            return False
        
        if quality_score < settings.SWARM_MIN_QUALITY_PUBLISH:
            return False
        
        domain = urlparse(url).netloc.lower().replace('www.', '')
        url_hash = get_url_hash(url)
        
        lottery = get_lottery_manager()
        _, ticket = lottery.should_crawl(domain)
        
        event = CrawlResultEvent(
            url=url,
            url_hash=url_hash,
            title=title,
            description=description[:200],  # Truncate for bandwidth
            domain=domain,
            quality_score=quality_score,
            slop_score=slop_score,
            spam_score=spam_score,
            word_count=word_count,
            tags=tags[:5],  # Max 5 tags
            embedding=embedding,
            experts=experts or {},
            vrf_proof=ticket.vrf_proof.hex() if ticket.vrf_proof else None,
        )
        
        if await self.nostr.publish_result(event):
            self._stats['results_published'] += 1
            logger.info(f"📤 Published: {title[:40]}")
            return True
        return False
    
    async def publish_vote(self, url_hash: str, is_good: bool, cluster_id: int) -> bool:
        challenge, nonce = await self._make_pow()  # Now async
        
        event = VoteSignalEvent(
            url_hash=url_hash,
            is_good=is_good,
            cluster_id=cluster_id,
            pow_challenge=challenge,
            pow_nonce=nonce,
        )
        
        if await self.nostr.publish_vote(event):
            self._stats['votes_published'] += 1
            return True
        return False
    
    def get_stats(self) -> dict:
        return self._stats.copy()


# ============================================================
# SINGLETON MANAGER
# ============================================================
_swarm_ingestion: Optional[SwarmIngestion] = None
_swarm_publisher: Optional[SwarmPublisher] = None


async def init_swarm(nostr: NostrService):
    global _swarm_ingestion, _swarm_publisher
    _swarm_ingestion = SwarmIngestion(nostr)
    _swarm_ingestion.register_handlers()
    _swarm_publisher = SwarmPublisher(nostr)
    logger.info("✅ Swarm integration initialized")


def get_swarm_publisher() -> Optional[SwarmPublisher]:
    return _swarm_publisher


def get_swarm_stats() -> dict:
    stats = {}
    if _swarm_ingestion:
        stats['ingestion'] = _swarm_ingestion.get_stats()
    if _swarm_publisher:
        stats['publishing'] = _swarm_publisher.get_stats()
    return stats
