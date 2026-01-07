"""
Crawler Worker - Decentralized Swarm-Enabled Crawler

Key changes from centralized version:
- Uses VRF lottery to determine domain responsibility
- Publishes results to Nostr swarm
- Publishes discovered URLs to swarm
- Can receive URLs from other nodes
"""
import asyncio
import hashlib
import re
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
from collections import Counter
from .topology import boost_indie_domains
import aiohttp
from bs4 import BeautifulSoup
from celery import Celery
from sqlalchemy import select, func, delete, update, cast, text as sql_text, CHAR
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects.postgresql import insert    
import redis.asyncio as redis
from .config import get_settings, BLOCKED_DOMAINS, TRUSTED_DOMAINS
from .database import async_session, Page, Vocabulary, CrawlQueue, DomainLink
from .experts import council

from .filters import (
    slop_detector, spam_filter, quality_analyzer,
    extract_tags, normalize_title
)
from .embedder import get_embedder
from .vrf import get_lottery_manager

settings = get_settings()

# Robots.txt cache: {domain: (RobotFileParser, expiry_time)}
_robots_cache: Dict[str, tuple] = {}
ROBOTS_CACHE_TTL = 3600  # 1 hour

# Celery app
celery_app = Celery(
    'crawler',
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_routes={
        'crawler.*': {'queue': 'crawler'},
    },
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

# ============================================================
# URL UTILITIES
# ============================================================
def normalize_url(url: str) -> str:
    """Normalize URL for deduplication"""
    url = url.split('#')[0].rstrip('/')
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc.lower()}{parsed.path}"

def get_url_hash(url: str) -> str:
    """Get hash of normalized URL"""
    return hashlib.md5(normalize_url(url).encode()).hexdigest()

def get_title_hash(domain: str, title: str) -> str:
    """Get hash of normalized title for dedup"""
    normalized = normalize_title(title)
    return hashlib.md5(f"{domain}:{normalized}".encode()).hexdigest()

# ============================================================
# ROBOTS.TXT COMPLIANCE
# ============================================================
async def get_robots_parser(domain: str, scheme: str = "https") -> Optional[RobotFileParser]:
    """Get or fetch robots.txt parser for domain."""
    global _robots_cache
    
    now = datetime.utcnow()
    
    if domain in _robots_cache:
        parser, expiry = _robots_cache[domain]
        if now < expiry:
            return parser
    
    robots_url = f"{scheme}://{domain}/robots.txt"
    parser = RobotFileParser()
    
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(robots_url) as resp:
                if resp.status == 200:
                    content = await resp.text()
                    parser.parse(content.splitlines())
                else:
                    parser.allow_all = True
    except Exception:
        parser.allow_all = True
    
    _robots_cache[domain] = (parser, now + timedelta(seconds=ROBOTS_CACHE_TTL))
    return parser

async def is_allowed_by_robots(url: str) -> bool:
    """Check if URL is allowed by robots.txt"""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    
    parser = await get_robots_parser(domain, parsed.scheme)
    if parser is None:
        return True
    
    return parser.can_fetch(settings.USER_AGENT, url)

# ============================================================
# TOKENIZER
# ============================================================
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
}

def tokenize_content(text: str) -> List[str]:
    """Tokenize content for BM25 index"""
    words = re.findall(r'\b[a-z0-9]+\b', text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]

async def push_to_swarm_outbox(action: str, data: dict):
    """Push payload to Redis for API to publish"""
    if not settings.NOSTR_ENABLED:
        return
        
    try:
        r = redis.from_url(settings.REDIS_URL)
        payload = json.dumps({'action': action, 'data': data})
        await r.lpush('swarm_outbox', payload)
        await r.close()
    except Exception as e:
        print(f"⚠️ Failed to push to swarm outbox: {e}")

# ============================================================
# SWARM INTEGRATION
# ============================================================
async def get_swarm_publisher():
    """Get swarm publisher if available"""
    if not settings.NOSTR_ENABLED:
        return None
    try:
        from .swarm import get_swarm_publisher as _get_publisher
        return _get_publisher()
    except Exception:
        return None

async def check_domain_lottery(domain: str) -> tuple[bool, Optional[object]]:
    """
    Check VRF lottery for domain.
    
    Returns:
        (should_crawl, ticket)
    """
    if not settings.NOSTR_ENABLED:
        return True, None  # No lottery in centralized mode
    
    try:
        lottery = get_lottery_manager()
        return lottery.should_crawl(domain, redundancy=settings.VRF_REDUNDANCY)
    except Exception:
        return True, None  # Fallback to allowing crawl

# ============================================================
# CRAWLER TASKS
# ============================================================
@celery_app.task(name='crawler.crawl_url')
def crawl_url_task(url: str, priority: float = 0.5):
    """Celery task wrapper for async crawl"""
    return asyncio.get_event_loop().run_until_complete(
        crawl_url(url, priority)
    )

async def crawl_url(url: str, source_trust: float = 0.5) -> Optional[Dict]:
    """
    Crawl a single URL.
    
    Now includes:
    - VRF lottery check (only crawl if we "won" the domain)
    - Publish results to swarm
    - Publish discovered URLs to swarm
    
    Returns:
        Page dict if successful, None if filtered/error
    """
    domain = urlparse(url).netloc.lower().replace('www.', '')
    
    # --- 1. ROBUST BLOCKLIST CHECK ---
    if domain in BLOCKED_DOMAINS:
        return None
    
    # Check root domain (e.g., "ads.google.com" -> "google.com")
    parts = domain.split('.')
    if len(parts) > 2:
        root = f"{parts[-2]}.{parts[-1]}"
        if root in BLOCKED_DOMAINS:
            return None
    
    # VRF Lottery check - only crawl if we won this domain
    should_crawl, lottery_ticket = await check_domain_lottery(domain)
    if not should_crawl:
        print(f"🎲 Lost lottery for {domain}, skipping")
        return None
    
    # Robots.txt check
    if not await is_allowed_by_robots(url):
        print(f"🤖 Blocked by robots.txt: {url[:50]}")
        return None
    
    url_hash = get_url_hash(url)
    
    async with async_session() as db:
        # Check if already indexed
        result = await db.execute(
            select(Page.id).where(Page.url_hash == url_hash)
        )
        if result.scalar_one_or_none():
            return None
        
        try:
            # Fetch page
            timeout = aiohttp.ClientTimeout(total=settings.CRAWL_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                headers = {'User-Agent': settings.USER_AGENT}
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        print(f"⏭️ HTTP {resp.status}: {url[:50]}")
                        return None
                    
                    content_type = resp.headers.get('content-type', '')
                    if 'text/html' not in content_type.lower():
                        print(f"⏭️ Not HTML: {url[:50]}")
                        return None
                    
                    html = await resp.text()
            
            # Parse
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract title
            title = ""
            if soup.title and soup.title.string:
                title = soup.title.string.strip()[:500]
            if not title:
                h1 = soup.find('h1')
                title = h1.get_text().strip()[:500] if h1 else domain
            
            # Check for duplicate title
            title_hash = get_title_hash(domain, title)
            result = await db.execute(
                select(Page.id).where(Page.title_hash == title_hash)
            )
            if result.scalar_one_or_none():
                print(f"⏭️ Duplicate: {title[:40]}")
                return None
            
            # Extract description
            desc = ""
            meta = soup.find('meta', attrs={'name': 'description'})
            if meta and meta.get('content'):
                desc = meta['content'][:500]
            
            # Extract text
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript']):
                tag.decompose()
            text = soup.get_text(' ', strip=True)
            
            # Content length check
            if len(text) < settings.MIN_CONTENT_LENGTH:
                print(f"⏭️ Too short ({len(text)}): {url[:50]}")
                return None
            
            # Spam check
            is_spam, spam_score = spam_filter.predict(url, title, text)
            if is_spam:
                print(f"🚫 Spam ({spam_score:.2f}): {title[:40]}")
                return None
            
            # Quality analysis
            metrics = quality_analyzer.analyze(soup, text, url)
            slop_score = slop_detector.score(text, title, soup)
            quality_score = quality_analyzer.compute_score(metrics, slop_score)
            
            # Quality threshold
            if quality_score < settings.MIN_QUALITY or slop_score > settings.SLOP_THRESHOLD:
                print(f"⛔ Q:{quality_score:.2f} S:{slop_score:.2f} | {title[:40]}")
                return None
            
            # Extract tags
            tags = extract_tags(soup, url)
            
            # Get embedding (FIX: reduce=False for 384 dims)
            embedder = await get_embedder()
            embedding_text = f"{title} {desc} {text[:2000]}"
            embedding = embedder.encode(embedding_text, reduce=False) 
            
            # RUN EXPERTS ANALYSIS
            expert_data = council.analyze(embedding_text)
            
            # Tokenize for BM25
            terms = tokenize_content(f"{title} {desc} {text}")
            term_counts = Counter(terms)
            word_count = len(terms)
            
            # Create page record
            page = Page(
                url=url,
                url_hash=url_hash,
                title_hash=title_hash,
                title=title,
                description=desc,
                domain=domain,
                quality_score=quality_score,
                slop_score=slop_score,
                spam_score=spam_score,
                domain_trust=metrics['domain_trust'],
                content_length=len(text),
                word_count=word_count,
                embedding=embedding.tolist(),
                tags=','.join(tags),
                expert_data=expert_data
            )
            db.add(page)
            await db.flush()  # Get page ID
            
            # 1. COMPUTE AND SAVE SEARCH VECTOR (Weighted: Title=A, Body=B)
            # Using raw SQL because SQLAlchemy 2.0's update() doesn't support bindparams
            # Note: setweight() requires "char" type, not varchar, hence the explicit casts
            await db.execute(
                sql_text("""
                    UPDATE pages SET search_vector = 
                        setweight(to_tsvector('english', :title), 'A'::"char") ||
                        setweight(to_tsvector('english', :body), 'B'::"char")
                    WHERE id = :page_id
                """).bindparams(
                    title=title,
                    body=f"{desc} {text[:5000]}",
                    page_id=page.id
                )
            )

            # 2. UPDATE VOCABULARY FOR SPELL CHECKER
            # Efficient Upsert: Insert new words, increment count for existing ones
            unique_words = set(terms) # 'terms' was computed in existing code above
            
            if unique_words:
                insert_stmt = insert(Vocabulary).values(
                    [{'word': w, 'doc_count': 1} for w in unique_words]
                )
                
                do_update_stmt = insert_stmt.on_conflict_do_update(
                    index_elements=['word'],
                    set_={'doc_count': Vocabulary.doc_count + 1}
                )
                
                await db.execute(do_update_stmt)

            
            # Train spam filter (ham example)
            await spam_filter.train(db, url, False, title, text)
            
            # ============================================================
            # QUEUE FOR SWARM (Outbox Pattern)
            # ============================================================
            if quality_score >= settings.SWARM_MIN_QUALITY_PUBLISH and settings.SWARM_PUBLISH_RESULTS:
                await push_to_swarm_outbox('result', {
                    'url': url,
                    'title': title,
                    'description': desc,
                    'quality_score': quality_score,
                    'slop_score': slop_score,
                    'spam_score': spam_score,
                    'word_count': word_count,
                    'tags': tags,
                    'experts': expert_data,
                    'embedding': embedding.tolist(),
                })
            
            # ============================================================
            # OUTLINK DISCOVERY
            # ============================================================
            link_priority = quality_score * source_trust
            queued = 0
            discovered_urls = []
            seen_link_domains = set()  # For domain_links dedup within this page
            
            for a in soup.find_all('a', href=True)[:settings.MAX_OUTLINKS]:
                href = a.get('href', '')
                if href.startswith('http'):
                    
                    link_domain = urlparse(href).netloc.lower().replace('www.', '')
                    
                    if link_domain in BLOCKED_DOMAINS: 
                        continue
                    parts = link_domain.split('.')
                    if len(parts) > 2 and f"{parts[-2]}.{parts[-1]}" in BLOCKED_DOMAINS:
                        continue
                    
                    if link_domain not in BLOCKED_DOMAINS:
                        link_trust = TRUSTED_DOMAINS.get(link_domain, 0.5)
                        link_hash = get_url_hash(href)
                        
                        # ============================================================
                        # TRACK DOMAIN-TO-DOMAIN LINKS (for topology analysis)
                        # ============================================================
                        if link_domain != domain and link_domain not in seen_link_domains:
                            seen_link_domains.add(link_domain)
                            try:
                                # Upsert domain link
                                link_stmt = insert(DomainLink).values(
                                    source=domain,
                                    target=link_domain,
                                    link_count=1
                                ).on_conflict_do_update(
                                    constraint='uq_domain_link',
                                    set_={'link_count': DomainLink.link_count + 1}
                                )
                                await db.execute(link_stmt)
                            except Exception:
                                pass  # Non-critical
                        
                        # Check if not already indexed
                        exists = await db.execute(
                            select(Page.id).where(Page.url_hash == link_hash)
                        )
                        if not exists.scalar_one_or_none():
                            normalized_href = normalize_url(href)
                            final_priority = link_priority * link_trust
                            
                            # Add to local queue
                            queue_item = CrawlQueue(
                                url=normalized_href,
                                priority=final_priority,
                                source_page_id=page.id
                            )
                            # --- START FIX ---
                            try:
                                # Create a savepoint to isolate this insert
                                async with db.begin_nested():
                                    db.add(queue_item)
                                    await db.flush() # Force SQL execution immediately
                                queued += 1
                                discovered_urls.append((normalized_href, final_priority))
                            except IntegrityError:
                                # If it fails, it rolls back to the savepoint automatically
                                pass  # Duplicate URL, ignore safely
                            except Exception:
                                pass
                            # --- END FIX ---
            
            await db.commit()
            
            # Queue high-priority discoveries for Swarm
            if settings.SWARM_PUBLISH_DISCOVERY:
                for disc_url, disc_priority in discovered_urls:
                    if disc_priority >= 0.6:
                        await push_to_swarm_outbox('discovery', {
                            'url': disc_url,
                            'priority': disc_priority,
                            'source_url': url
                        })
            
            swarm_indicator = "[s]" if settings.NOSTR_ENABLED else ""
            print(f"✅ Q:{quality_score:.2f} S:{slop_score:.2f} +{queued} {swarm_indicator} | {title[:50]}")
            
            return {
                'id': page.id,
                'title': title,
                'url': url,
                'quality': quality_score,
                'tags': tags
            }
            
        except Exception as e:
            print(f"❌ {url[:50]} - {type(e).__name__}: {e}")
            return None

@celery_app.task(name='crawler.process_queue')
def process_queue_task(batch_size: int = None):
    """Process crawl queue"""
    batch_size = batch_size or settings.CRAWL_BATCH_SIZE
    return asyncio.get_event_loop().run_until_complete(
        process_queue(batch_size)
    )

async def process_queue(batch_size: int = None) -> Dict:
    """
    Process batch of URLs from queue.
    
    Now with VRF lottery awareness - may skip URLs for domains
    we didn't win.
    """
    batch_size = batch_size or settings.CRAWL_BATCH_SIZE
    
    async with async_session() as db:
        # Get top priority URLs with row locking
        result = await db.execute(
            select(CrawlQueue)
            .order_by(CrawlQueue.priority.desc())
            .limit(batch_size)
            .with_for_update(skip_locked=True)
        )
        queue_items = result.scalars().all()
        
        if not queue_items:
            return {'processed': 0, 'message': 'Queue empty'}
        
        # Delete claimed items from queue
        urls = [(item.url, item.priority) for item in queue_items]
        await db.execute(
            delete(CrawlQueue).where(
                CrawlQueue.id.in_([item.id for item in queue_items])
            )
        )
        await db.commit()
    
    # Crawl URLs (VRF lottery checked inside crawl_url)
    success = 0
    skipped = 0
    
    for url, priority in urls:
        result = await crawl_url(url, priority)
        if result:
            success += 1
        elif result is None:
            # Could be skipped due to lottery
            skipped += 1
        await asyncio.sleep(settings.CRAWL_DELAY)
    
    # Get stats
    async with async_session() as db:
        total_pages = await db.scalar(select(func.count(Page.id)))
        queue_size = await db.scalar(select(func.count(CrawlQueue.id)))
        avg_quality = await db.scalar(select(func.avg(Page.quality_score))) or 0
    
    print(f"📊 {total_pages} pages | Avg quality: {avg_quality:.0%} | Queue: {queue_size}")
    
    return {
        'processed': len(urls),
        'success': success,
        'skipped': skipped,
        'total_pages': total_pages,
        'queue_size': queue_size
    }

@celery_app.task(name='crawler.seed_queue')
def seed_queue_task():
    """Seed queue with initial URLs"""
    return asyncio.get_event_loop().run_until_complete(seed_queue())

async def seed_queue() -> int:
    """Seed queue with initial URLs if empty."""
    seeds = [
        # ============================================================
        # TECH NEWS & AGGREGATORS
        # ============================================================
        ("https://news.ycombinator.com/", 1.0),
        ("https://lobste.rs/", 1.0),
        ("https://github.com/trending", 1.0),
        ("https://lwn.net/", 0.95),
        ("https://tilde.news/", 0.9),
        ("https://slashdot.org/", 0.85),
        
        # ============================================================
        # PROGRAMMING DOCUMENTATION
        # ============================================================
        ("https://developer.mozilla.org/en-US/", 1.0),
        ("https://docs.python.org/3/", 1.0),
        ("https://go.dev/doc/", 1.0),
        ("https://doc.rust-lang.org/book/", 1.0),
        ("https://www.postgresql.org/docs/", 0.95),
        ("https://redis.io/docs/", 0.9),
        ("https://kubernetes.io/docs/", 0.9),
        ("https://docs.djangoproject.com/", 0.9),
        ("https://flask.palletsprojects.com/", 0.9),
        ("https://reactjs.org/docs/", 0.9),
        ("https://vuejs.org/guide/", 0.9),
        ("https://htmx.org/docs/", 0.9),
        ("https://sqlite.org/docs.html", 0.95),
        ("https://nginx.org/en/docs/", 0.85),
        
        # ============================================================
        # SCIENCE & RESEARCH
        # ============================================================
        ("https://arxiv.org/list/cs.AI/recent", 1.0),
        ("https://arxiv.org/list/cs.LG/recent", 1.0),
        ("https://arxiv.org/list/cs.PL/recent", 0.9),
        ("https://arxiv.org/list/math/recent", 0.9),
        ("https://arxiv.org/list/physics/recent", 0.9),
        ("https://www.nature.com/news", 0.9),
        ("https://www.quantamagazine.org/", 1.0),
        ("https://phys.org/", 0.85),
        ("https://www.sciencedaily.com/", 0.8),
        
        # ============================================================
        # QUALITY BLOGS & PUBLICATIONS
        # ============================================================
        ("https://arstechnica.com/", 0.9),
        ("https://www.wired.com/", 0.85),
        ("https://spectrum.ieee.org/", 0.9),
        ("https://cacm.acm.org/", 0.95),
        ("https://queue.acm.org/", 0.95),
        ("https://blog.codinghorror.com/", 0.9),
        ("https://www.joelonsoftware.com/", 0.9),
        ("https://martinfowler.com/", 0.95),
        ("https://danluu.com/", 0.95),
        ("https://jvns.ca/", 0.95),
        ("https://rachelbythebay.com/w/", 0.9),
        ("https://drewdevault.com/", 0.9),
        ("https://fasterthanli.me/", 0.9),
        ("https://blog.cleancoder.com/", 0.9),
        ("https://www.kalzumeus.com/", 0.9),
        
        # ============================================================
        # ENCYCLOPEDIAS & REFERENCES
        # ============================================================
        ("https://en.wikipedia.org/wiki/Main_Page", 0.95),
        ("https://www.britannica.com/", 0.9),
        ("https://plato.stanford.edu/", 1.0),
        ("https://www.gutenberg.org/", 0.9),
        
        # ============================================================
        # INDIE WEB & SMALL WEB
        # ============================================================
        ("https://indieweb.org/", 0.95),
        ("https://100r.co/", 0.9),
        ("https://solar.lowtechmagazine.com/", 0.9),
        ("https://cheapskatesguide.org/", 0.85),
        ("https://suckless.org/", 0.9),
        ("https://cat-v.org/", 0.85),
        ("https://landchad.net/", 0.85),
        ("https://based.cooking/", 0.8),
        
        # ============================================================
        # EDUCATIONAL
        # ============================================================
        ("https://ocw.mit.edu/", 0.95),
        ("https://www.khanacademy.org/", 0.9),
        ("https://teachyourselfcs.com/", 0.95),
        ("https://missing.csail.mit.edu/", 0.95),
        ("https://craftinginterpreters.com/", 0.95),
        ("https://www.nand2tetris.org/", 0.95),
        
        # ============================================================
        # TOOLS & UTILITIES
        # ============================================================
        ("https://alternativeto.net/", 0.85),
        ("https://privacyguides.org/", 0.9),
        ("https://www.privacytools.io/", 0.85),
        ("https://sr.ht/", 0.9),
        ("https://codeberg.org/", 0.9),
        
        # ============================================================
        # NEWS - MAINSTREAM (balanced selection)
        # ============================================================
        ("https://www.reuters.com/", 0.95),
        ("https://apnews.com/", 0.95),
        ("https://www.npr.org/", 0.9),
        ("https://www.bbc.com/news", 0.9),
        ("https://www.theguardian.com/", 0.85),
        ("https://www.economist.com/", 0.9),
        ("https://www.theatlantic.com/", 0.85),
        ("https://www.csmonitor.com/", 0.9),
        
        # ============================================================
        # NEWS - INTERNATIONAL
        # ============================================================
        ("https://www.aljazeera.com/", 0.85),
        ("https://www.dw.com/en/", 0.9),
        ("https://www.france24.com/en/", 0.85),
        ("https://www.scmp.com/", 0.8),
        ("https://english.kyodonews.net/", 0.85),
        ("https://www.abc.net.au/news/", 0.9),
        ("https://www.cbc.ca/news", 0.9),
        
        # ============================================================
        # NEWS - INDEPENDENT & INVESTIGATIVE
        # ============================================================
        ("https://theintercept.com/", 0.85),
        ("https://www.propublica.org/", 0.95),
        ("https://www.icij.org/", 0.95),
        ("https://www.bellingcat.com/", 0.9),
        ("https://theconversation.com/", 0.9),
        ("https://restofworld.org/", 0.9),
        ("https://www.currentaffairs.org/", 0.85),
        
        # ============================================================
        # NEWS - BUSINESS & FINANCE
        # ============================================================
        ("https://www.ft.com/", 0.9),
        ("https://www.bloomberg.com/", 0.85),
        ("https://www.marketwatch.com/", 0.8),
        ("https://wolfstreet.com/", 0.85),
        ("https://www.nakedcapitalism.com/", 0.8),
        
        # ============================================================
        # NEWS - LONG-FORM & ANALYSIS
        # ============================================================
        ("https://www.newyorker.com/", 0.9),
        ("https://www.lrb.co.uk/", 0.9),
        ("https://aeon.co/", 0.95),
        ("https://www.noemamag.com/", 0.9),
        ("https://www.palladiummag.com/", 0.85),
        ("https://worksinprogress.co/", 0.9),
        ("https://asteriskmag.com/", 0.9),
    ]
    
    async with async_session() as db:
        count = await db.scalar(select(func.count(CrawlQueue.id)))
        if count > 0:
            return 0
        
        for url, priority in seeds:
            db.add(CrawlQueue(url=url, priority=priority))
        
        try:
            await db.commit()
            return len(seeds)
        except IntegrityError:
            await db.rollback()
            return 0

# ============================================================
# CELERY BEAT SCHEDULE
# ============================================================
celery_app.conf.beat_schedule = {
    'process-queue-every-30s': {
        'task': 'crawler.process_queue',
        'schedule': 30.0,
    },
}

# ============================================================
# VRF LOTTERY CLEANUP (periodic task)
# ============================================================
@celery_app.task(name='crawler.cleanup_lottery')
def cleanup_lottery_task():
    """Cleanup old lottery tickets"""
    if settings.NOSTR_ENABLED:
        try:
            lottery = get_lottery_manager()
            lottery.cleanup_old_epochs()
        except Exception:
            pass

celery_app.conf.beat_schedule['cleanup-lottery-hourly'] = {
    'task': 'crawler.cleanup_lottery',
    'schedule': 3600.0,  # Every hour
}


# ============================================================
# TOPOLOGY BOOST (Indie Web Support)
# ============================================================
@celery_app.task(name='crawler.run_topology_boost')
def run_topology_boost_task():
    """Run topology analysis to boost indie domains"""
    # Import locally to avoid circular imports if necessary
    from .topology import boost_indie_domains
    
    return asyncio.get_event_loop().run_until_complete(
        run_topology_boost()
    )

async def run_topology_boost():
    print("🌿 Starting topology boost analysis...")
    async with async_session() as db:
        await boost_indie_domains(db)
    print("🌿 Topology boost complete.")

# Schedule it to run once a day (86400 seconds)
celery_app.conf.beat_schedule['topology-boost-daily'] = {
    'task': 'crawler.run_topology_boost',
    'schedule': 86400.0, 
}
