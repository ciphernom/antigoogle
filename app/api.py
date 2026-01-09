"""
AntiGoogle API - FastAPI Application with Swarm Integration
"""
import os
import time
import secrets
import hashlib
import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional
from contextlib import asynccontextmanager
import logging
from .spellcheck import spell_checker
from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis
from .trust import get_trust_dag
from .config import get_settings
from .database import (
    init_db, get_db, async_session,
    Page, CrawlQueue, PageStats, PowChallenge, Vocabulary
)
from .embedder import get_embedder
from .search import search, get_trending
from .ratings import BayesianRatings, record_signal
from .crawler import crawl_url, seed_queue
from .filters import spam_filter
from .templates import CSS, JS, POW_JS, INFINITE_SCROLL_JS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("antigoogle")
settings = get_settings()

# Redis client
redis_client: redis.Redis = None

# Nostr service (initialized in lifespan if enabled)
nostr_service = None

async def process_swarm_outbox():
    """Background task to publish events from Crawler"""
    if not settings.NOSTR_ENABLED:
        return

    print("📨 Swarm Outbox Worker started")
    pub = None
    # Use a new connection for the worker
    r = redis.from_url(settings.REDIS_URL)
    
    # Rate limiting state
    last_public_publish = 0
    PUBLIC_RATE_LIMIT = 60  # Seconds between public blasts
    
    try:
        while True:
            # 1. Ensure Publisher is ready
            if not pub:
                from .swarm import get_swarm_publisher
                pub = get_swarm_publisher()
                if not pub:
                    await asyncio.sleep(5)
                    continue

            # 2. Get message from Redis (Non-blocking LPOP + Sleep)
            try:
                # Use lpop instead of blpop to avoid asyncio timeout bugs
                data = await r.lpop('swarm_outbox')
            except Exception as e:
                print(f"⚠️ Redis read error: {e}")
                await asyncio.sleep(5)
                continue
            
            if not data:
                await asyncio.sleep(1) # Queue empty, sleep 1s
                continue

            # 3. Process message
            try:
                msg = json.loads(data)
                action = msg.get('action')
                d = msg.get('data')
                
                # Logic: Should we send to public relays?
                now = time.time()
                publish_to_public = (now - last_public_publish) > PUBLIC_RATE_LIMIT
                
                # Snapshot connections so we can restore them
                all_connections = pub.nostr._connections.copy()
                
                if publish_to_public:
                    last_public_publish = now
                    # Send to ALL (Local + Public)
                else:
                    # Filter: Keep ONLY Local
                    for url in list(pub.nostr._connections.keys()):
                        if "localhost" not in url and "relay" not in url:
                            del pub.nostr._connections[url]

                # Perform the publish
                try:
                    if action == 'result':
                        await pub.publish_result(
                            url=d['url'],
                            title=d['title'],
                            description=d['description'],
                            quality_score=d['quality_score'],
                            slop_score=d['slop_score'],
                            spam_score=d['spam_score'],
                            word_count=d['word_count'],
                            tags=d['tags'],
                            embedding=d.get('embedding'),
                            experts=d.get('experts'),
                            trust_proof=d.get('trust_proof'),
                            proof_hash=d.get('proof_hash')
                        )
                    elif action == 'discovery':
                        await pub.publish_discovery(
                            url=d['url'],
                            priority=d['priority'],
                            source_url=d['source_url'],
                            trust_proof=d.get('trust_proof'),
                            proof_hash=d.get('proof_hash')
                        )
                finally:
                    # ALWAYS restore full connection list
                    pub.nostr._connections = all_connections
                    
            except Exception as e:
                logger.error(f"Failed to process swarm message: {e}")
                
    except asyncio.CancelledError:
        print("📨 Swarm Outbox Worker stopped")

# ============================================================
# LIFESPAN
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global redis_client, nostr_service
    
    # Security check
    if not os.environ.get("SECRET_KEY"):
        logger.warning("⚠️  SECRET_KEY not set! Using random key.")
    
    # Initialize database
    await init_db()
    print("✅ Database ready")
    
    # Initialize embedder
    embedder = await get_embedder()
    print("✅ Embedder ready")
    
    # Load spam filter training data
    async with async_session() as db:
        await spam_filter.load_from_db(db)
    print("✅ Spam filter loaded")
    
    # Initialize Redis
    redis_client = redis.from_url(settings.REDIS_URL)
    print("✅ Redis ready")

    # ============================================================
    # LOAD BLOCKLIST (Moved from config.py to avoid blocking import)
    # ============================================================
    print("🛡️ Loading blocklists...")
    try:
        from .config import COMMUNITY_BLOCKLIST, BLOCKED_DOMAINS
        from .blocklists import fetch_blocklist
        
        # Run in thread pool to avoid blocking the loop
        loop = asyncio.get_event_loop()
        blocklist_data = await loop.run_in_executor(None, fetch_blocklist)
        
        if blocklist_data:
            COMMUNITY_BLOCKLIST.update(blocklist_data)
            BLOCKED_DOMAINS.update(blocklist_data)
            print(f"✅ Blocklist loaded: {len(blocklist_data)} domains")
        else:
            print("⚠️ Blocklist load returned empty")
    except Exception as e:
        logger.error(f"❌ Failed to load blocklist: {e}")

    # ============================================================
    # NOSTR/SWARM INITIALIZATION
    # ============================================================
    if settings.NOSTR_ENABLED:
        try:
            from .nostr import NostrService, get_nostr_service, stop_nostr_service
            from .swarm import init_swarm
            from .setup import SwarmConfig
            
            # Check for saved config first
            saved_config = SwarmConfig.load()
            
            if saved_config:
                # Use saved config
                nostr_service = NostrService(
                    private_key_hex=saved_config.private_key,
                    relays=saved_config.relays,
                    trusted_pubkeys=set(saved_config.trusted_pubkeys),
                )
                await nostr_service.start()
                print(f"✅ Nostr swarm ready from config (pubkey: {nostr_service.public_key[:16]}...)")
            else:
                # Use settings (env vars)
                nostr_service = await get_nostr_service()
                print(f"✅ Nostr swarm ready from env (pubkey: {nostr_service.public_key[:16]}...)")
            
            await init_swarm(nostr_service)
            asyncio.create_task(process_swarm_outbox())
            
        except Exception as e:
            logger.error(f"⚠️ Nostr initialization failed: {e}")
            nostr_service = None
    
    # Seed crawler queue
    async with async_session() as db:
        count = await db.scalar(select(func.count(CrawlQueue.id)))
        if count == 0:
            added = await seed_queue()
            print(f"🌱 Seeded {added} URLs")
    
    print("📚 Loading spell checker vocabulary...")
    async with async_session() as db:
        # Fetch top 50,000 words by frequency
        # Adjust 'Vocabulary' to 'Term' if you haven't migrated tables yet
        result = await db.execute(
            select(Vocabulary.word, Vocabulary.doc_count)
            .order_by(Vocabulary.doc_count.desc())
            .limit(50000)
        )
        # Convert to dict {word: freq}
        vocab_data = {row.word: row.doc_count for row in result.fetchall()}
        spell_checker.load_vocab(vocab_data)
    # ----------------------   
    
    print("🌳 Initializing Trust DAG...")
    trust_dag = await get_trust_dag()
    print(f"✅ Trust DAG ready: {trust_dag.get_stats()}")
    
    yield
    
    # Cleanup
    await redis_client.close()
    
    if nostr_service:
        from .nostr import stop_nostr_service
        await stop_nostr_service()

app = FastAPI(title="AntiGoogle", lifespan=lifespan)

# Static files
try:
    app.mount("/wasm", StaticFiles(directory="wasm/pkg"), name="wasm")
except:
    print("⚠️ WASM directory not found")

# ============================================================
# RATE LIMITING
# ============================================================
async def check_rate_limit(request: Request, limit: int = None, window: int = None) -> bool:
    """Check rate limit using Redis"""
    limit = limit or settings.RATE_LIMIT_REQUESTS
    window = window or settings.RATE_LIMIT_WINDOW
    
    ip = request.client.host
    key = f"rate:{ip}"
    
    count = await redis_client.incr(key)
    if count == 1:
        await redis_client.expire(key, window)
    
    return count <= limit

# ============================================================
# PROOF OF WORK
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

async def get_pow_difficulty(db: AsyncSession) -> int:
    """Get dynamic PoW difficulty based on recent activity"""
    count = await redis_client.get("pow:recent") or 0
    count = int(count)
    
    if count < 10:
        return settings.BASE_POW_DIFFICULTY
    
    import math
    return min(
        settings.BASE_POW_DIFFICULTY + int(math.log2(count / 10 + 1)),
        settings.MAX_POW_DIFFICULTY
    )

async def create_pow_challenge(db: AsyncSession) -> dict:
    """Create a new PoW challenge"""
    challenge_id = secrets.token_hex(16)
    challenge = secrets.token_hex(32)
    difficulty = await get_pow_difficulty(db)
    expires = datetime.utcnow() + timedelta(minutes=5)
    
    pow_challenge = PowChallenge(
        id=challenge_id,
        challenge=challenge,
        difficulty=difficulty,
        expires_at=expires
    )
    db.add(pow_challenge)
    await db.commit()
    
    return {
        'challenge_id': challenge_id,
        'challenge': challenge,
        'difficulty': difficulty
    }

async def verify_pow(db: AsyncSession, challenge_id: str, nonce: str) -> tuple[bool, str]:
    result = await db.execute(select(PowChallenge).where(PowChallenge.id == challenge_id))
    pow_challenge = result.scalar_one_or_none()
    
    if not pow_challenge:
        return False, "Invalid challenge"
    
    if datetime.utcnow() > pow_challenge.expires_at:
        await db.delete(pow_challenge)
        await db.commit()
        return False, "Expired"
    
    try:
        nonce_int = int(nonce)
    except:
        return False, "Invalid nonce"
    
    h = fnv1a_hash(pow_challenge.challenge, nonce_int)
    leading_zeros = count_leading_zero_hex(h)
    
    if leading_zeros >= pow_challenge.difficulty:
        await db.delete(pow_challenge)
        await db.commit()
        return True, "OK"
    
    return False, f"Invalid solution"

# ============================================================
# ROUTES - PAGES
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def home(db: AsyncSession = Depends(get_db), setup: str = None):
    """Home page"""
    total = await db.scalar(select(func.count(Page.id)))
    avg_q = await db.scalar(select(func.avg(Page.quality_score))) or 0
    
    # Check if setup is needed
    from .setup import SwarmConfig
    config = SwarmConfig.load()
    setup_link = '<a href="/settings" style="color:var(--dim);text-decoration:none">Settings</a>' if config else '<a href="/setup" style="color:var(--dim);text-decoration:none">Setup</a>'
    
    # Setup complete message
    setup_msg = '<div style="color:#3c3;margin-bottom:20px">Swarm setup complete!</div>' if setup == "complete" else ""
    
    # Swarm indicator
    swarm_status = ""
    if settings.NOSTR_ENABLED and nostr_service:
        connected = len(nostr_service._connections)
        swarm_status = f'<span style="color:var(--dim);font-size:11px">[s] {connected} relays</span>'
    
    return f"""<!DOCTYPE html><html><head>
            <title>AntiGoogle</title>
            <meta name="viewport" content="width=device-width,initial-scale=1">
            {CSS}{JS}
            </head><body>
            <div class="container" style="text-align:center;padding-top:20vh">
            {setup_msg}
            <h1 style="font-size:32px;margin-bottom:20px;border:none">AntiGoogle</h1>
            <form action="/search" method="get" id="sf">
            <input type="text" name="q" placeholder="query..." autofocus>
            <input type="hidden" name="l1" id="l1"><input type="hidden" name="l2" id="l2">
            </form>
            <script>document.getElementById('sf').onsubmit=()=>{{document.getElementById('l1').value=window.agL1||'';document.getElementById('l2').value=window.agL2||''}}</script>
            <div style="margin-top:30px">
            <a href="/trending" style="color:var(--dim);text-decoration:none;margin-right:15px">Trending</a>
            <a href="/add" style="color:var(--dim);text-decoration:none;margin-right:15px">Add</a>
            <a href="/stats" style="color:var(--dim);text-decoration:none;margin-right:15px">Stats ({total:,})</a>
            <a href="/swarm" style="color:var(--dim);text-decoration:none;margin-right:15px">Swarm</a>
            {setup_link}
            </div>
            <div style="margin-top:15px">{swarm_status}</div>
            </div>
            </body></html>"""


@app.get("/api/search")
async def api_search(
    q: str,
    offset: int = 0,
    limit: int = 20,
    l1: Optional[str] = None,  # <--- Changed from int to str
    l2: Optional[str] = None,  # <--- Changed from int to str
    db: AsyncSession = Depends(get_db)
):
    """JSON search endpoint for infinite scroll"""
    # Handle empty strings from forms/JS
    l1_val = int(l1) if l1 and str(l1).strip() else None
    l2_val = int(l2) if l2 and str(l2).strip() else None
    
    return await search(db, q, l1_val, l2_val, limit=limit, offset=offset)


@app.get("/search", response_class=HTMLResponse)
async def search_page(
    q: str = "",
    l1: Optional[str] = None,  # <--- Changed from int to str
    l2: Optional[str] = None,  # <--- Changed from int to str
    db: AsyncSession = Depends(get_db)
):
    """Search results page"""
    if not q:
        return RedirectResponse("/")
    
    #  Handle empty strings from Lynx/Forms
    l1_val = int(l1) if l1 and str(l1).strip() else None
    l2_val = int(l2) if l2 and str(l2).strip() else None
    
    start = time.time()
    # Pass the parsed integer values to the search function
    results = await search(db, q, l1_val, l2_val) 
    elapsed = time.time() - start
    # Build results HTML
    results_html = ""
    for i, r in enumerate(results['results']):
        quality_class = "quality-high" if r['quality_score'] > 0.7 else "quality-med" if r['quality_score'] > 0.4 else "quality-low"
        
        rating_html = ""
        if r.get('vote_count', 0) > 0:
            pct = int(r['user_rating'] * 100) if r['user_rating'] else 0
            rating_html = f'<span class="tag">{pct}% ({r["vote_count"]})</span>'
        
        tags_html = ""
        if r.get('tags'):
            for tag in r['tags'].split(',')[:3]:
                if tag:
                    tags_html += f'<span class="tag">{tag}</span>'
        
        results_html += f"""
        <div class="result">
            <a href="{r['url']}" class="result-title" onclick="AG?.click?.({i},{r['id']})">{r['title']}</a>
            <span class="result-url">{r['domain']}</span>
            <div class="result-desc">{r['description'] or ''}</div>
            <div class="result-meta">
                <span class="{quality_class}">Q:{r['quality_score']:.0%}</span>
                {rating_html}{tags_html}
                <a href="/rate/{r['id']}" style="margin-left:10px;color:var(--dim)">[rate]</a>
            </div>
        </div>
        """
    
    correction_html = ""
    if results.get('corrected'):
        correction_html = f'<div class="correction">Showing results for: {results["corrected"]}</div>'
    
    return f"""<!DOCTYPE html><html><head>
        <title>{q} - AntiGoogle</title>
        <meta name="viewport" content="width=device-width,initial-scale=1">
        {CSS}{JS}{INFINITE_SCROLL_JS} </head><body>
        <div class="container">
        <div class="header">
            <a href="/" class="logo">AntiGoogle</a>
            <div class="nav">
                <a href="/trending">Trending</a>
                <a href="/add">Add</a>
                <a href="/stats">Stats</a>
            </div>
        </div>
        <form action="/search" method="get" id="sf">
            <input type="text" name="q" value="{q}">
            <input type="hidden" name="l1" id="l1"><input type="hidden" name="l2" id="l2">
        </form>
        <script>document.getElementById('sf').onsubmit=()=>{{document.getElementById('l1').value=window.agL1||'';document.getElementById('l2').value=window.agL2||''}}</script>
        <div class="stats">{len(results['results'])} results ({elapsed:.2f}s)</div>
        {correction_html}
        
        <div id="results">
            {results_html}
        </div>
        
        </div></body></html>"""

@app.get("/trending", response_class=HTMLResponse)
async def trending_page(db: AsyncSession = Depends(get_db)):
    """Trending pages"""
    trending = await get_trending(db)
    
    results_html = ""
    for r in trending:
        rating_html = ""
        if r.get('vote_count', 0) > 0:
            pct = int(r['user_rating'] * 100) if r['user_rating'] else 0
            rating_html = f'<span class="tag">{pct}% ({r["vote_count"]})</span>'
        
        results_html += f"""
        <div class="result">
            <a href="{r['url']}" class="result-title">{r['title']}</a>
            <span class="result-url">{r['domain']}</span>
            <div class="result-desc">{r['description'] or ''}</div>
            <div class="result-meta">Q:{r['quality_score']:.0%} {rating_html}
            <a href="/rate/{r['id']}" style="margin-left:10px;color:var(--dim)">[rate]</a></div>
        </div>
        """
    
    return f"""<!DOCTYPE html><html><head>
        <title>Trending - AntiGoogle</title>{CSS}{JS}</head><body>
        <div class="container">
        <div class="header"><a href="/" class="logo">AntiGoogle</a>
        <div class="nav"><a href="/add">Add</a><a href="/stats">Stats</a></div></div>
        <h3>Trending Pages</h3>
        {results_html}
        </div></body></html>"""

@app.get("/add", response_class=HTMLResponse)
async def add_page(db: AsyncSession = Depends(get_db)):
    """Add URL page"""
    pow_data = await create_pow_challenge(db)
    
    return f"""<!DOCTYPE html><html><head>
        <title>Add URL - AntiGoogle</title>{CSS}{JS}{POW_JS}</head><body>
        <div class="container">
        <div class="header"><a href="/" class="logo">AntiGoogle</a>
        <div class="nav"><a href="/trending">Trending</a><a href="/stats">Stats</a></div></div>
        <h3>Submit URL</h3>
        <form action="/add" method="post" onsubmit="return submitWithPoW(event)">
        <label>URL to index:</label>
        <input type="text" name="url" placeholder="https://..." required>
        <input type="hidden" name="challenge_id" value="{pow_data['challenge_id']}">
        <input type="hidden" name="nonce" id="nonce">
        <button type="submit">Submit (PoW required)</button>
        <div id="pow-status" class="pow-status"></div>
        </form>
        </div></body></html>"""

@app.post("/add", response_class=HTMLResponse)
async def add_url(
    url: str = Form(...),
    challenge_id: str = Form(...),
    nonce: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """Handle URL submission"""
    # Verify PoW
    valid, msg = await verify_pow(db, challenge_id, nonce)
    if not valid:
        return f"""<!DOCTYPE html><html><head>{CSS}{JS}</head><body>
        <div class="container"><div class="box">
        <h3>[ ERROR ]</h3><p>Proof of Work failed: {msg}</p>
        </div><p><a href="/add">Try again</a></p></div></body></html>"""
    
    # Crawl
    result = await crawl_url(url, source_trust=0.8)
    
    if result:
        tags = ', '.join(result.get('tags', []))
        return f"""<!DOCTYPE html><html><head>{CSS}{JS}</head><body><div class="container">
<div class="box">
<h3>[ INDEXED ]</h3>
<p>{result['title']}</p>
<p style="font-size:12px;color:var(--dim)">{url}</p>
<p>Quality: {result['quality']:.0%} | Tags: {tags}</p></div>
<p><a href="/add">Add another</a> | <a href="/">Search</a></p></div></body></html>"""
    else:
        return f"""<!DOCTYPE html><html><head>{CSS}{JS}</head><body><div class="container">
<div class="box">
<h3>[ FAILED ]</h3>
<p>Could not index.</p>
<p style="font-size:12px;color:var(--dim)">Check server logs for details.</p>
</div>
<p><a href="/add">Try another</a></p></div></body></html>"""

@app.get("/rate/{page_id}", response_class=HTMLResponse)
async def rate_page(page_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    """Rate a page"""
    result = await db.execute(select(Page).where(Page.id == page_id))
    page = result.scalar_one_or_none()
    if not page:
        return RedirectResponse("/")
    
    stats = await BayesianRatings.get_stats(db, page_id)
    
    score = stats['expected']
    bar_length = 20
    filled = int(score * bar_length)
    ascii_bar = "[" + "=" * filled + "-" * (bar_length - filled) + "]"
    uncertainty = int(stats['uncertainty'] * 100)
    
    return f"""<!DOCTYPE html><html><head><title>Rate - AntiGoogle</title>{CSS}{JS}{POW_JS}
<script>
async function submitRating(isGood) {{
    const st = document.getElementById('pow-status');
    const btns = document.querySelectorAll('button');
    btns.forEach(b => b.disabled = true);
    st.textContent = 'Solving proof-of-work...';
    
    try {{
        const cr = await fetch('/api/pow');
        const cd = await cr.json();
        const nonce = await solvePoW(cd.challenge, cd.difficulty);
        st.textContent = 'Submitting...';
        
        const r = await fetch('/api/rate', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{
                page_id: {page_id},
                is_good: isGood,
                challenge_id: cd.challenge_id,
                nonce: nonce
            }})
        }});
        const data = await r.json();
        if (data.status === 'ok') {{
            document.getElementById('result').innerHTML = '<div class="box">[OK] Vote recorded. New score: ' + Math.round(data.expected * 100) + '%</div>';
        }} else if (data.status === 'duplicate') {{
            document.getElementById('result').innerHTML = '<div class="box">[--] Already voted.</div>';
        }} else {{
            st.textContent = 'Error: ' + data.message;
            btns.forEach(b => b.disabled = false);
        }}
    }} catch(err) {{
        st.textContent = 'Error: ' + err.message;
        btns.forEach(b => b.disabled = false);
    }}
}}
</script>
</head><body><div class="container">
<div class="header"><a href="/" class="logo">AntiGoogle</a></div>
<div class="box">
<h3>Rate This Page</h3>
<p><b>{page.title}</b></p>
<p style="font-size:13px;color:var(--meta)">{page.url}</p>
<div style="margin:20px 0">
<div style="font-family:monospace; font-size:16px; margin-bottom:5px">
    {ascii_bar} <span style="font-weight:bold">{int(score * 100)}%</span>
</div>
<p style="font-size:13px;color:var(--meta)">{stats['total_votes']} votes | +/-{uncertainty}% uncertainty</p>
</div>
<div style="display:flex;gap:15px;margin:25px 0">
<button onclick="submitRating(true)" style="flex:1">[+] Quality</button>
<button onclick="submitRating(false)" style="flex:1">[-] Slop</button>
</div>
<div id="pow-status" class="pow-status"></div>
<div id="result"></div>
</div>
<p><a href="/">&lt;- Back to search</a></p>
</div></body></html>"""

@app.get("/stats", response_class=HTMLResponse)
async def stats_page(db: AsyncSession = Depends(get_db)):
    """Stats page"""
    total = await db.scalar(select(func.count(Page.id)))
    queue = await db.scalar(select(func.count(CrawlQueue.id)))
    avg_q = await db.scalar(select(func.avg(Page.quality_score))) or 0
    
    # Top domains
    result = await db.execute(
        select(Page.domain, func.count(Page.id).label('cnt'))
        .group_by(Page.domain)
        .order_by(func.count(Page.id).desc())
        .limit(15)
    )
    dom_html = "".join(f"<div>{d.cnt} | {d.domain}</div>" for d in result.fetchall())
    
    return f"""<!DOCTYPE html><html><head><title>Stats</title>{CSS}{JS}</head>
<body><div class="container">
<div class="header"><a href="/" class="logo">AntiGoogle</a>
<div class="nav"><a href="/trending">Trending</a><a href="/add">Add</a><a href="/swarm">Swarm</a></div></div>
<h3>System Statistics</h3>
<div class="box">
<div>Pages Indexed: {total:,}</div>
<div>Queue Size: {queue:,}</div>
<div>Avg Quality: {avg_q:.1%}</div>
</div>
<h3>Top Domains</h3>
<div class="box" style="font-family:monospace">
{dom_html}
</div>
</div></body></html>"""

# ============================================================
# SWARM STATUS PAGE
# ============================================================
@app.get("/swarm", response_class=HTMLResponse)
async def swarm_page(db: AsyncSession = Depends(get_db)):
    """Swarm status page"""
    if not settings.NOSTR_ENABLED:
        return f"""<!DOCTYPE html><html><head><title>Swarm</title>{CSS}{JS}</head>
        <body><div class="container">
        <div class="header"><a href="/" class="logo">AntiGoogle</a></div>
        <div class="box"><h3>Swarm Disabled</h3>
        <p>Set NOSTR_ENABLED=true to enable swarm mode.</p></div>
        </div></body></html>"""
    
    # Get stats
    nostr_stats = nostr_service.get_stats() if nostr_service else {}
    
    try:
        from .swarm import get_swarm_stats
        from .vrf import get_lottery_manager
        
        swarm_stats = get_swarm_stats()
        lottery = get_lottery_manager()
        
        epoch = lottery.lottery.get_current_epoch()
    except Exception as e:
        swarm_stats = {}
        epoch = 0
    
    connected = nostr_stats.get('connected_relays', [])
    pubkey = nostr_stats.get('public_key', 'N/A')
    trusted = nostr_stats.get('trusted_pubkeys', 0)
    
    ingestion = swarm_stats.get('ingestion', {})
    publishing = swarm_stats.get('publishing', {})
    
    relays_html = "".join(f"<div>✓ {r}</div>" for r in connected) or "<div>No connections</div>"
    
    return f"""<!DOCTYPE html><html><head><title>Swarm</title>{CSS}{JS}</head>
<body><div class="container">
<div class="header"><a href="/" class="logo">AntiGoogle</a>
<div class="nav"><a href="/settings">Settings</a><a href="/stats">Stats</a><a href="/trending">Trending</a></div></div>

<h3>[s] Swarm Status</h3>

<div class="box">
<h4>Identity</h4>
<div style="font-family:monospace;font-size:12px;word-break:break-all">
Pubkey: {pubkey}
</div>
<div>Trusted peers: {trusted}</div>
<div>Current epoch: {epoch}</div>
<div>PoW difficulty: {settings.SWARM_POW_DIFFICULTY}</div>
</div>

<div class="box">
<h4>Connected Relays ({len(connected)})</h4>
<div style="font-family:monospace;font-size:12px">
{relays_html}
</div>
</div>

<div class="box">
<h4>Ingestion</h4>
<div>Discoveries received: {ingestion.get('discoveries_received', 0)}</div>
<div>Results received: {ingestion.get('results_received', 0)}</div>
<div>Votes received: {ingestion.get('votes_received', 0)}</div>
</div>

<div class="box">
<h4>Publishing</h4>
<div>Discoveries published: {publishing.get('discoveries_published', 0)}</div>
<div>Results published: {publishing.get('results_published', 0)}</div>
<div>Votes published: {publishing.get('votes_published', 0)}</div>
</div>

</div></body></html>"""

# ============================================================
# API ROUTES
# ============================================================
@app.get("/api/pow")
async def api_get_pow(db: AsyncSession = Depends(get_db)):
    """Get PoW challenge"""
    return await create_pow_challenge(db)

@app.post("/api/rate")
async def api_rate(request: Request, db: AsyncSession = Depends(get_db)):
    """Submit a rating"""
    data = await request.json()
    page_id = data.get('page_id')
    is_good = data.get('is_good')
    challenge_id = data.get('challenge_id')
    nonce = data.get('nonce')
    
    if None in (page_id, is_good, challenge_id, nonce):
        return JSONResponse({'status': 'error', 'message': 'Missing fields'})
    
    # Verify PoW
    valid, msg = await verify_pow(db, challenge_id, nonce)
    if not valid:
        return JSONResponse({'status': 'error', 'message': msg})
    
    # Get fingerprint
    ip = request.client.host
    ua = request.headers.get('user-agent', '')
    fp = hashlib.md5(f"{ip}:{ua}".encode()).hexdigest()[:16]
    
    # Validate page exists
    page_result = await db.execute(select(Page).where(Page.id == page_id))
    page = page_result.scalar_one_or_none()
    if not page:
        return JSONResponse({'status': 'error', 'message': 'Page not found'})
    
    # Record vote
    stats = await BayesianRatings.record_vote(db, page_id, is_good, fp)
    
    # Publish vote to swarm
    if settings.NOSTR_ENABLED and stats.get('recorded'):
        try:
            from .swarm import get_swarm_publisher
            publisher = get_swarm_publisher()
            if publisher:
                # Get cluster ID from request if available
                l1 = data.get('l1', 0)
                l2 = data.get('l2', 0)
                cluster_id = l1 * settings.NUM_L2 + l2
                
                await publisher.publish_vote(
                    url_hash=page.url_hash,
                    is_good=is_good,
                    cluster_id=cluster_id,
                )
        except Exception as e:
            logger.warning(f"Failed to publish vote to swarm: {e}")
    
    if not stats.get('recorded', True):
        return JSONResponse({
            'status': 'duplicate',
            'message': 'Already voted',
            'expected': stats['expected'],
            'total_votes': stats['total_votes']
        })
    
    return JSONResponse({
        'status': 'ok',
        'expected': stats['expected'],
        'total_votes': stats['total_votes']
    })

@app.post("/api/signal")
async def api_signal(request: Request, db: AsyncSession = Depends(get_db)):
    """Record personalization signal"""
    try:
        data = await request.json()
        l1 = data.get('l1')
        l2 = data.get('l2')
        page_id = data.get('page_id')
        score = data.get('score')
        fp = data.get('fp_hash', 'anon')
        
        if None in (l1, l2, page_id, score):
            return {'status': 'ignored'}
        
        cluster_id = l1 * settings.NUM_L2 + l2
        await record_signal(db, cluster_id, page_id, score, fp)
        
        return {'status': 'ok'}
    except:
        return {'status': 'error'}

@app.get("/api/lsh")
async def api_lsh():
    """Get LSH planes for client"""
    embedder = await get_embedder()
    return embedder.get_lsh_planes()

@app.get("/api/projection")
async def api_projection():
    """Get projection matrix for client"""
    embedder = await get_embedder()
    return embedder.get_projection_matrix()

@app.get("/api/stats/{page_id}")
async def api_page_stats(page_id: int, db: AsyncSession = Depends(get_db)):
    """Get rating stats for a page"""
    return await BayesianRatings.get_stats(db, page_id)

# ============================================================
# SWARM API
# ============================================================
@app.get("/api/swarm/status")
async def api_swarm_status():
    """Get swarm status"""
    if not settings.NOSTR_ENABLED or not nostr_service:
        return {'enabled': False}
    
    from .swarm import get_swarm_stats
    
    return {
        'enabled': True,
        'nostr': nostr_service.get_stats(),
        'swarm': get_swarm_stats(),
    }

@app.get("/api/swarm/genesis")
async def api_swarm_genesis():
    """Get genesis configuration"""
    embedder = await get_embedder()
    
    return {
        'version': 1,
        'pca_matrix': embedder.get_projection_matrix()['matrix'],
        'lsh_l1_planes': embedder.get_lsh_planes()['l1_planes'],
        'lsh_l2_planes': embedder.get_lsh_planes()['l2_planes'],
        'vrf_epoch_seconds': settings.VRF_EPOCH_SECONDS,
    }

# ============================================================
# HEALTH CHECK
# ============================================================
@app.get("/health")
async def health():
    """Health check endpoint"""
    swarm_ok = (not settings.NOSTR_ENABLED) or (nostr_service and len(nostr_service._connections) > 0)
    
    return {
        "status": "ok" if swarm_ok else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "nostr_enabled": settings.NOSTR_ENABLED,
        "nostr_connected": len(nostr_service._connections) if nostr_service else 0,
    }


# ============================================================
# SETUP & SETTINGS
# ============================================================
@app.get("/setup", response_class=HTMLResponse)
async def setup_page():
    """Setup wizard for first-time configuration"""
    from .setup import SwarmConfig, get_setup_html, get_settings_html
    
    # If already configured, redirect to settings
    config = SwarmConfig.load()
    if config:
        return RedirectResponse("/settings", status_code=302)
    
    return get_setup_html()


@app.post("/setup", response_class=HTMLResponse)
async def setup_submit(request: Request):
    """Handle setup form submission"""
    from .setup import (
        SwarmConfig, get_setup_html, generate_keypair,
        validate_private_key, validate_relay_url, validate_pubkey
    )
    
    form = await request.form()
    
    # Handle key generation or import
    key_mode = form.get("key_mode", "generate")
    if key_mode == "generate":
        private_key, _ = generate_keypair()
    else:
        private_key = form.get("private_key", "").strip()
        valid, msg = validate_private_key(private_key)
        if not valid:
            return get_setup_html(error=f"Invalid private key: {msg}")
    
    # Parse relays
    relays_text = form.get("relays", "")
    relays = [r.strip() for r in relays_text.strip().split("\n") if r.strip()]
    
    if not relays:
        return get_setup_html(error="At least one relay is required")
    
    for relay in relays:
        valid, msg = validate_relay_url(relay)
        if not valid:
            return get_setup_html(error=f"Invalid relay '{relay}': {msg}")
    
    # Parse trusted pubkeys
    trusted_text = form.get("trusted_pubkeys", "")
    trusted_pubkeys = [p.strip() for p in trusted_text.strip().split("\n") if p.strip()]
    
    for pk in trusted_pubkeys:
        valid, msg = validate_pubkey(pk)
        if not valid:
            return get_setup_html(error=f"Invalid pubkey '{pk[:16]}...': {msg}")
    
    # Parse behavior settings
    publish_results = "publish_results" in form
    publish_discovery = "publish_discovery" in form
    accept_external = "accept_external" in form
    
    # Parse advanced settings
    try:
        pow_difficulty = int(form.get("pow_difficulty", 4))
        pow_difficulty = max(1, min(8, pow_difficulty))
    except ValueError:
        pow_difficulty = 4
    
    try:
        vrf_epoch_seconds = int(form.get("vrf_epoch_seconds", 600))
        vrf_epoch_seconds = max(60, min(3600, vrf_epoch_seconds))
    except ValueError:
        vrf_epoch_seconds = 600
    
    try:
        vrf_redundancy = int(form.get("vrf_redundancy", 2))
        vrf_redundancy = max(1, min(10, vrf_redundancy))
    except ValueError:
        vrf_redundancy = 2
    
    # Save config
    config = SwarmConfig(
        private_key=private_key,
        relays=relays,
        trusted_pubkeys=trusted_pubkeys,
        publish_results=publish_results,
        publish_discovery=publish_discovery,
        accept_external=accept_external,
        pow_difficulty=pow_difficulty,
        vrf_epoch_seconds=vrf_epoch_seconds,
        vrf_redundancy=vrf_redundancy,
    )
    config.save()
    
    # Redirect to home with success
    return RedirectResponse("/?setup=complete", status_code=302)


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(success: str = None):
    """Settings page for modifying configuration"""
    from .setup import SwarmConfig, get_settings_html, get_setup_html
    
    config = SwarmConfig.load()
    if not config:
        return RedirectResponse("/setup", status_code=302)
    
    success_msg = "Settings saved successfully!" if success == "saved" else None
    return get_settings_html(config, success=success_msg)


@app.post("/settings", response_class=HTMLResponse)
async def settings_submit(request: Request):
    """Handle settings form submission"""
    from .setup import (
        SwarmConfig, get_settings_html,
        validate_relay_url, validate_pubkey
    )
    
    config = SwarmConfig.load()
    if not config:
        return RedirectResponse("/setup", status_code=302)
    
    form = await request.form()
    
    # Parse relays
    relays_text = form.get("relays", "")
    relays = [r.strip() for r in relays_text.strip().split("\n") if r.strip()]
    
    if not relays:
        return get_settings_html(config, error="At least one relay is required")
    
    for relay in relays:
        valid, msg = validate_relay_url(relay)
        if not valid:
            return get_settings_html(config, error=f"Invalid relay '{relay}': {msg}")
    
    # Parse trusted pubkeys
    trusted_text = form.get("trusted_pubkeys", "")
    trusted_pubkeys = [p.strip() for p in trusted_text.strip().split("\n") if p.strip()]
    
    for pk in trusted_pubkeys:
        valid, msg = validate_pubkey(pk)
        if not valid:
            return get_settings_html(config, error=f"Invalid pubkey '{pk[:16]}...': {msg}")
    
    # Update config
    config.relays = relays
    config.trusted_pubkeys = trusted_pubkeys
    config.publish_results = "publish_results" in form
    config.publish_discovery = "publish_discovery" in form
    config.accept_external = "accept_external" in form
    
    try:
        config.pow_difficulty = max(1, min(8, int(form.get("pow_difficulty", 4))))
    except ValueError:
        pass
    
    try:
        config.vrf_epoch_seconds = max(60, min(3600, int(form.get("vrf_epoch_seconds", 600))))
    except ValueError:
        pass
    
    try:
        config.vrf_redundancy = max(1, min(10, int(form.get("vrf_redundancy", 2))))
    except ValueError:
        pass
    
    config.save()
    
    return RedirectResponse("/settings?success=saved", status_code=302)


@app.post("/settings/regenerate-key", response_class=HTMLResponse)
async def settings_regenerate_key():
    """Regenerate keypair"""
    from .setup import SwarmConfig, generate_keypair, get_settings_html
    
    config = SwarmConfig.load()
    if not config:
        return RedirectResponse("/setup", status_code=302)
    
    # Generate new key
    private_key, _ = generate_keypair()
    config.private_key = private_key
    config.save()
    
    return get_settings_html(config, success="Keypair regenerated! Share your new public key with peers.")


@app.get("/setup/generate-key")
async def setup_generate_key():
    """API endpoint to generate keypair"""
    from .setup import generate_keypair
    private_key, public_key = generate_keypair()
    return {"private_key": private_key, "public_key": public_key}


@app.get("/setup/derive-pubkey")
async def setup_derive_pubkey(key: str):
    """API endpoint to derive pubkey from private key"""
    from .setup import validate_private_key, get_pubkey_from_private
    
    valid, msg = validate_private_key(key)
    if not valid:
        return {"error": msg}
    
    return {"public_key": get_pubkey_from_private(key)}
