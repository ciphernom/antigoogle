"""
AntiGoogle Production Configuration
"""
import os
import secrets
from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings
from .blocklists import fetch_blocklist

# Generate random fallback (not for production!)
_INSECURE_DEFAULT = secrets.token_hex(32)

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://antigoogle:antigoogle@localhost:5432/antigoogle"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 40
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL: int = 300  # 5 minutes
    
    # Embeddings
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBED_DIM: int = 384  # Full dimension from model
    STORED_DIM: int = 384  # could do but dont do PCA reduced for storage
    
    # Search
    TOP_N: int = 20
    BM25_WEIGHT: float = 0.35
    SEMANTIC_WEIGHT: float = 0.65
    
    # Quality thresholds
    MIN_QUALITY: float = 0.3
    SLOP_THRESHOLD: float = 0.5
    SPAM_THRESHOLD: float = 0.7
    MIN_CONTENT_LENGTH: int = 500
    
    # Crawler
    CRAWL_BATCH_SIZE: int = 50
    CRAWL_DELAY: float = 0.5
    CRAWL_TIMEOUT: int = 12
    MAX_OUTLINKS: int = 25
    USER_AGENT: str = "Mozilla/5.0 (compatible; AntiGoogle/2.0; +https://github.com/ciphernom/antigoogle)"
    
    # Rate limiting
    RATE_LIMIT_REQUESTS: int = 60
    RATE_LIMIT_WINDOW: int = 60
    
    # PoW (for web UI)
    BASE_POW_DIFFICULTY: int = 4
    MAX_POW_DIFFICULTY: int = 7
    
    # Personalization (LSH)
    L1_PLANES: int = 6
    L2_PLANES: int = 6
    NUM_L1: int = 64
    NUM_L2: int = 64
    LSH_SEED_L1: int = 42
    LSH_SEED_L2: int = 43
    
    # Security (set SECRET_KEY env var in production!)
    SECRET_KEY: str = _INSECURE_DEFAULT
    
    # ============================================================
    # NOSTR CONFIGURATION (Decentralized Swarm)
    # ============================================================
    
    # Enable/disable Nostr integration
    NOSTR_ENABLED: bool = True
    
    # Private key for signing events (32-byte hex)
    # IMPORTANT: Set NOSTR_PRIVATE_KEY env var in production!
    # Generate with: python -c "import secrets; print(secrets.token_hex(32))"
    NOSTR_PRIVATE_KEY: str = secrets.token_hex(32)
    
    # Relay URLs to connect to
    NOSTR_RELAYS: List[str] = [
        "ws://relay:8080",          # Local Relay (acts as an outbox)
        "wss://relay.damus.io",     # Public (General purpose)
        "wss://nos.lol",            # Public (General purpose)
        "wss://relay.primal.net",   # Public (Very fast, caching)
        "wss://relay.snort.social", # Public
    ]
    
    # Trusted pubkeys (only accept events from these)
    # Empty = accept from all (not recommended for production)
    NOSTR_TRUSTED_PUBKEYS: List[str] = []
    
    # VRF (Domain Lottery)
    VRF_EPOCH_SECONDS: int = 600  # 10 minutes per epoch
    VRF_REDUNDANCY: int = 2  # How many nodes can crawl same domain
    
    # Swarm PoW (Anti-Spam) - reuses same PoW as web UI
    SWARM_POW_DIFFICULTY: int = 4  # Leading zero hex digits required
    
    # Swarm behavior
    SWARM_PUBLISH_RESULTS: bool = True   # Publish crawl results to network
    SWARM_PUBLISH_DISCOVERY: bool = True  # Publish discovered URLs
    SWARM_ACCEPT_EXTERNAL: bool = True    # Accept URLs from other nodes
    SWARM_MIN_QUALITY_PUBLISH: float = 0.5  # Only publish high-quality results
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

# Blocked domains (social media, e-commerce, etc.)
SOCIAL_COMMERCE_BLOCKLIST = {
    'twitter.com', 'x.com', 't.co',
    'facebook.com', 'fb.com', 'instagram.com',
    'tiktok.com', 'snapchat.com',
    'amazon.com', 'ebay.com', 'aliexpress.com',
    'linkedin.com',
}

# Trusted domains (bypass spam filter, higher crawl priority)
TRUSTED_DOMAINS = {
    # Code & Documentation
    'github.com': 0.9, 
    'gitlab.com': 0.8,
    'stackoverflow.com': 0.9, 
    'stackexchange.com': 0.8,
    'developer.mozilla.org': 0.95,
    'docs.python.org': 0.95,
    'doc.rust-lang.org': 0.95,
    'go.dev': 0.9,
    'docs.microsoft.com': 0.8,
    'readthedocs.io': 0.8, 
    'docs.rs': 0.85,
    'docs.djangoproject.com': 0.9,
    'flask.palletsprojects.com': 0.9,
    'fastapi.tiangolo.com': 0.9,
    'htmx.org': 0.9,
    'react.dev': 0.9, 
    'vuejs.org': 0.9, 
    'tailwindcss.com': 0.9,
    'nodejs.org': 0.9, 
    'bun.sh': 0.9, 
    'deno.land': 0.9,
    'sqlite.org': 0.95,
    'nginx.org': 0.85,
    
    # Databases & Infrastructure
    'postgresql.org': 0.95,
    'kubernetes.io': 0.9,
    'redis.io': 0.9,
    
    # Tech News & Forums
    'news.ycombinator.com': 0.9, 
    'lobste.rs': 0.9,
    'lwn.net': 0.9,
    'slashdot.org': 0.8,
    'tilde.news': 0.85,
    'arstechnica.com': 0.85,
    'spectrum.ieee.org': 0.9,
    
    # Research & Academia
    'arxiv.org': 1.0,
    'wikipedia.org': 0.9,
    'plato.stanford.edu': 1.0,
    'quantamagazine.org': 1.0,
    'nature.com': 0.9,
    'sciencedaily.com': 0.8,
    'phys.org': 0.8,
    'acm.org': 0.95,
    'cacm.acm.org': 0.95,
    'queue.acm.org': 0.95,
    'theconversation.com': 0.9,
    
    # Open Source
    'kernel.org': 0.9,
    'gnu.org': 0.85, 
    'apache.org': 0.85,
    'sr.ht': 0.9,
    'codeberg.org': 0.9,
    
    # Quality Blogs
    'paulgraham.com': 0.9,
    'danluu.com': 0.95,
    'jvns.ca': 0.95,
    'martinfowler.com': 0.95,
    'joelonsoftware.com': 0.9,
    'blog.codinghorror.com': 0.9,
    'rachelbythebay.com': 0.9,
    'drewdevault.com': 0.9,
    'fasterthanli.me': 0.9,
    'blog.cleancoder.com': 0.9,
    'kalzumeus.com': 0.9,
    'unenumerated.blogspot.com': 1.0,
    
    # Encyclopedias & References
    'britannica.com': 0.9,
    'gutenberg.org': 0.9,
    
    # Indie Web
    'indieweb.org': 0.9,
    '100r.co': 0.85,
    'solar.lowtechmagazine.com': 0.85,
    'suckless.org': 0.9,
    
    # Educational
    'ocw.mit.edu': 0.95,
    'khanacademy.org': 0.9,
    'teachyourselfcs.com': 0.95,
    'missing.csail.mit.edu': 0.95,
    'craftinginterpreters.com': 0.95,
    'nand2tetris.org': 0.95,
    
    # News - Wire Services & Quality
    'reuters.com': 0.95,
    'apnews.com': 0.95,
    'npr.org': 0.9,
    'bbc.com': 0.9,
    'propublica.org': 0.95,
    'icij.org': 0.95,
    'bellingcat.com': 0.9,
    'restofworld.org': 0.9,
    
    # Privacy & Tools
    'privacyguides.org': 0.9,
    'alternativeto.net': 0.8,
}

# Spam filter trusted (never mark as spam)
SPAM_TRUSTED_DOMAINS = {
    # Code & Documentation
    'github.com', 'gitlab.com', 'stackoverflow.com', 'stackexchange.com',
    'mozilla.org', 'developer.mozilla.org',
    'python.org', 'docs.python.org',
    'rust-lang.org', 'doc.rust-lang.org',
    'go.dev', 'golang.org',
    'readthedocs.io', 'docs.rs', 'crates.io', 'pypi.org', 'npmjs.com',
    'docs.djangoproject.com', 'flask.palletsprojects.com', 'fastapi.tiangolo.com',
    'htmx.org', 'tailwindcss.com', 'nextjs.org', 'svelte.dev',
    'react.dev', 'vuejs.org', 'nodejs.org', 'bun.sh', 'deno.land',
    'sqlite.org', 'nginx.org',
    
    # Databases & Infrastructure
    'postgresql.org', 'kubernetes.io', 'redis.io',
    
    # Research & Academia
    'arxiv.org', 'wikipedia.org', 'wikimedia.org',
    'plato.stanford.edu', 'stanford.edu',
    'quantamagazine.org',
    'nature.com', 'sciencedaily.com', 'phys.org',
    'acm.org', 'cacm.acm.org', 'queue.acm.org',
    'theconversation.com',
    'mit.edu', 'ocw.mit.edu', 'missing.csail.mit.edu',
    
    # Standards & Open Source
    'w3.org', 'ietf.org',
    'kernel.org', 'lwn.net', 'gnu.org', 'apache.org',
    'sr.ht', 'codeberg.org',
    
    # Tech News & Forums
    'news.ycombinator.com', 'lobste.rs', 'tilde.news',
    'slashdot.org', 'arstechnica.com', 'spectrum.ieee.org',
    
    # Quality Blogs
    'paulgraham.com', 'danluu.com', 'jvns.ca', 'martinfowler.com',
    'joelonsoftware.com', 'blog.codinghorror.com', 'rachelbythebay.com',
    'drewdevault.com', 'fasterthanli.me', 'blog.cleancoder.com', 'kalzumeus.com',
    
    # Encyclopedias
    'britannica.com', 'gutenberg.org',
    
    # Indie Web
    'indieweb.org', '100r.co', 'solar.lowtechmagazine.com', 'suckless.org',
    
    # Educational
    'khanacademy.org', 'teachyourselfcs.com', 'craftinginterpreters.com', 'nand2tetris.org',
    
    # News - Wire Services & Quality
    'reuters.com', 'apnews.com', 'npr.org', 'bbc.com', 'bbc.co.uk',
    'propublica.org', 'icij.org', 'bellingcat.com', 'restofworld.org',
    'theguardian.com', 'economist.com', 'theatlantic.com', 'csmonitor.com',
    'aljazeera.com', 'dw.com', 'france24.com', 'abc.net.au', 'cbc.ca',
    'ft.com', 'bloomberg.com',
    'newyorker.com', 'lrb.co.uk', 'aeon.co', 'noemamag.com',
    
    # Privacy & Tools  
    'privacyguides.org', 'alternativeto.net',
    
    # Publishing Platforms (user-generated but often quality)
    'substack.com', 'medium.com', 'dev.to', 'libsyn.com',
    
    # Misc trusted from original
    'businessoffashion.com', 'fmc.gov', 'miamiherald.com', 'apple.com',
}

# --- FINAL BLOCKLIST COMPOSITION ---

# 1. Critical blocks (Google, Tracking)
MANUAL_BLOCKLIST = {"google.com", "facebook.com", "analytics.google.com"}

# 2. The massive community list (Ads, Malware, Adult)
# PERFORMANCE FIX: Do not fetch on import. Initialize empty.
COMMUNITY_BLOCKLIST = set() 

# 3. Combine ALL lists: Social + Manual + Community
BLOCKED_DOMAINS = SOCIAL_COMMERCE_BLOCKLIST.union(MANUAL_BLOCKLIST).union(COMMUNITY_BLOCKLIST)
