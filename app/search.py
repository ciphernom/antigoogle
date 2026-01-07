"""
Search Service - Hybrid BM25 + Semantic with pgvector
"""
import re
import math
import json
import hashlib
from typing import List, Optional, Dict, Any
from sqlalchemy import select, func, text, desc, literal_column
from sqlalchemy.ext.asyncio import AsyncSession
from .spellcheck import spell_checker
import numpy as np

from .config import get_settings
from .database import Page, PageStats, ClusterSignal
from .embedder import get_embedder
from .filters import normalize_title

settings = get_settings()

# Redis client (imported from api at runtime to avoid circular import)
_redis_client = None

async def get_redis():
    """Get Redis client lazily to avoid circular imports"""
    global _redis_client
    if _redis_client is None:
        try:
            import redis.asyncio as redis
            _redis_client = redis.from_url(settings.REDIS_URL)
        except Exception:
            pass
    return _redis_client

# ============================================================
# TOKENIZER
# ============================================================
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
    'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'not', 'only', 'same',
    'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there',
}

def tokenize(text: str) -> List[str]:
    """Tokenize text for BM25"""
    words = re.findall(r'\b[a-z0-9]+\b', text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]

# ============================================================
# SPELL CHECKER (pg_trgm)
# ============================================================
async def correct_query(db: AsyncSession, query: str) -> str:
    """
    Server-side frequency-aware spell check.
    Uses in-memory dictionary loaded at startup.
    """
    if not query: 
        return query
    
    # Split query into words
    tokens = re.findall(r'\b[a-z0-9]+\b', query.lower())
    corrected_tokens = []
    has_change = False
    
    for token in tokens:
        # Ignore numbers or very short words
        if len(token) < 4 or token.isdigit():
            corrected_tokens.append(token)
            continue
            
        # Use the frequency-aware corrector
        correction = spell_checker.correction(token)
        
        if correction != token:
            corrected_tokens.append(correction)
            has_change = True
        else:
            corrected_tokens.append(token)
            
    if has_change:
        return " ".join(corrected_tokens)
    
    return query

# ============================================================
# BM25 SEARCH
# ============================================================
async def bm25_search(
    db: AsyncSession,
    query: str,
    limit: int = 200
) -> Dict[int, float]:
    """
    Replaced manual BM25 with Postgres Full-Text Search (TSVECTOR).
    """
    # 1. Convert user query to a Postgres TSQuery
    # 'websearch_to_tsquery' handles quotes, operators, and stemming automatically.
    ts_query = func.websearch_to_tsquery('english', query)
    
    # 2. Rank results
    # ts_rank_cd provides a "Cover Density" ranking similar to BM25.
    # The array [0.1, 0.2, 0.4, 1.0] sets weights for D, C, B, A labels.
    # We weighted Title as 'A' and Body as 'B' in the crawler.
    weights = literal_column("'{0.1, 0.2, 0.4, 1.0}'::float4[]")
    rank_expr = func.ts_rank_cd(weights, Page.search_vector, ts_query).label('rank')
    
    stmt = (
        select(Page.id, rank_expr)
        .where(Page.search_vector.op('@@')(ts_query)) # Match query
        .order_by(desc('rank'))
        .limit(limit)
    )
    
    result = await db.execute(stmt)
    
    # Return dict {page_id: score}
    return {row.id: float(row.rank) for row in result.fetchall()}

# ============================================================
# VECTOR SEARCH (pgvector)
# ============================================================
async def vector_search(
    db: AsyncSession,
    query_embedding: np.ndarray,
    limit: int = 200
) -> List[tuple[int, float]]:
    """
    Semantic search using pgvector cosine similarity.
    """
    # Convert to list for pgvector
    query_list = query_embedding.tolist()
    
    # Use pgvector's cosine distance operator
    result = await db.execute(
        select(
            Page.id,
            (1 - Page.embedding.cosine_distance(query_list)).label('similarity')
        )
        .order_by(text("similarity DESC"))  # Explicit text sort is safer here
        .limit(limit)
    )
    
    return [(row.id, row.similarity) for row in result.fetchall()]

# ============================================================
# HYBRID SEARCH
# ============================================================
async def search(
    db: AsyncSession,
    query: str,
    l1: Optional[int] = None,
    l2: Optional[int] = None,
    limit: int = None,
    offset: int = 0,
    expert_filter: str = None, # e.g. "is_tech"   
) -> Dict[str, Any]:
    """
    Hybrid search combining BM25 and semantic search.
    
    OPTIMIZED: 
    - Uses batch fetching to avoid N+1 query problem
    - Caches results in Redis for 5 minutes
    """
    limit = limit or settings.TOP_N
    
    # ============================================================
    # REDIS CACHE CHECK
    # ============================================================
    cache_key = f"search:{hashlib.md5(f'{query}:{l1}:{l2}:{limit}'.encode()).hexdigest()}"
    redis = await get_redis()
    
    if redis:
        try:
            cached = await redis.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass  # Cache miss or error, continue with search
    
    # 1. Attempt spelling correction
    corrected_q = await correct_query(db, query)
    used_query = corrected_q if corrected_q else query
    was_corrected = (used_query != query)
    
    # Get embedder
    embedder = await get_embedder()
    
    # BM25 search
    bm25_scores = await bm25_search(db, used_query, limit=200)
    
    # Semantic search
    query_embedding = await embedder.encode_async(used_query, reduce=True)
    vec_results = await vector_search(db, query_embedding, limit=200)
    vec_scores = {pid: score for pid, score in vec_results}
    
    # Combine candidates
    all_ids = list(set(bm25_scores.keys()) | set(vec_scores.keys()))
    if not all_ids:
        return {
            'results': [],
            'corrected': corrected_q if was_corrected else None
        }
    
    # ============================================================
    # BATCH FETCH: 3 queries instead of 3*N queries
    # ============================================================
    
    # Batch fetch all pages (1 query)
    pages_result = await db.execute(
        select(Page).where(Page.id.in_(all_ids))
    )
    pages_by_id = {p.id: p for p in pages_result.scalars().all()}
    
    # Batch fetch all page stats (1 query)
    stats_result = await db.execute(
        select(PageStats).where(PageStats.page_id.in_(all_ids))
    )
    stats_by_id = {s.page_id: s for s in stats_result.scalars().all()}
    
    # Batch fetch personalization signals if enabled (1 query)
    signals_by_id = {}
    if l1 is not None and l2 is not None:
        cluster_id = l1 * settings.NUM_L2 + l2
        signals_result = await db.execute(
            select(ClusterSignal)
            .where(ClusterSignal.cluster_id == cluster_id)
            .where(ClusterSignal.page_id.in_(all_ids))
        )
        signals_by_id = {s.page_id: s for s in signals_result.scalars().all()}
    
    # ============================================================
    # SCORING: No more queries in this loop
    # ============================================================
    
    # Normalize scores
    bm25_max = max(bm25_scores.values()) if bm25_scores else 1
    vec_max = max(vec_scores.values()) if vec_scores else 1
    
    candidates = []
    
    for pid in all_ids:
        # Get page from cache (no query)
        page = pages_by_id.get(pid)
        if not page:
            continue
        
        # Compute hybrid score
        bm25_norm = bm25_scores.get(pid, 0) / bm25_max
        vec_norm = vec_scores.get(pid, 0) / vec_max
        base_score = settings.BM25_WEIGHT * bm25_norm + settings.SEMANTIC_WEIGHT * vec_norm
        
        # Quality multiplier
        algo_quality = page.quality_score or 0.5
        
        # Get user ratings from cache (no query)
        page_stats = stats_by_id.get(pid)
        
        if page_stats and page_stats.vote_count > 0:
            user_quality = page_stats.alpha / (page_stats.alpha + page_stats.beta)
            vote_confidence = min(page_stats.vote_count / 20, 1.0)
            quality = algo_quality * (1 - 0.4 * vote_confidence) + user_quality * (0.4 * vote_confidence)
        else:
            quality = algo_quality
            user_quality = None
        
        quality_mult = 0.5 + quality
        
        # Personalization boost from cache (no query)
        pers_mult = 1.0
        if l1 is not None and l2 is not None:
            cluster_signal = signals_by_id.get(pid)
            if cluster_signal and cluster_signal.signal_count > 0:
                pers_mult = 1.0 + 0.3 * min(cluster_signal.score, 1.0)
        
        # Final score
        final_score = base_score * quality_mult * pers_mult
        
        
        # EXPERT BOOST
        if expert_filter:
            # If the user specifically wants "is_tech", we boost pages 
            # where the tech expert is confident (> 0.5)
            expert_score = page.expert_data.get(expert_filter, 0.0)
            if expert_score > 0.5:
                final_score *= (1.0 + expert_score) # Big boost
            else:
                final_score *= 0.1 # Bury irrelevant content
        
        candidates.append({
            'id': page.id,
            'url': page.url,
            'title': page.title,
            'description': page.description,
            'domain': page.domain,
            'quality_score': page.quality_score,
            'slop_score': page.slop_score,
            'tags': page.tags,
            'search_score': final_score,
            'user_rating': user_quality if page_stats and page_stats.vote_count > 0 else None,
            'vote_count': page_stats.vote_count if page_stats else 0,
        })
    
    # Sort by score
    candidates.sort(key=lambda x: x['search_score'], reverse=True)
    
    # Deduplicate by normalized title
    seen_titles = set()
    deduped = []
    for p in candidates:
        title_key = normalize_title(p['title'])
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            deduped.append(p)
    
    # --- Apply offset and limit slicing ---
    start = offset
    end = offset + limit
    sliced_results = deduped[start:end]
    has_more = len(deduped) > end
    
    result = {
        'results': sliced_results,
        'corrected': corrected_q if was_corrected else None,
        'has_more': has_more  
    }
    
    # ============================================================
    # REDIS CACHE WRITE (5 minute TTL)
    # ============================================================
    if redis:
        try:
            await redis.setex(cache_key, settings.CACHE_TTL, json.dumps(result))
        except Exception:
            pass  # Non-critical
    
    return result

# ============================================================
# TRENDING PAGES
# ============================================================
async def get_trending(db: AsyncSession, min_votes: int = 3, limit: int = 50) -> List[Dict[str, Any]]:
    """Get trending pages by user ratings"""
    
    # 1. Get pages with enough votes (>= min_votes), ordered by expected quality
    result = await db.execute(
        select(Page, PageStats)
        .join(PageStats, Page.id == PageStats.page_id)
        .where(PageStats.vote_count >= min_votes)
        .order_by(desc(PageStats.alpha / (PageStats.alpha + PageStats.beta)))
        .limit(limit)
    )
    
    trending = []
    for page, stats in result.fetchall():
        expected = stats.alpha / (stats.alpha + stats.beta)
        trending.append({
            'id': page.id,
            'url': page.url,
            'title': page.title,
            'description': page.description,
            'domain': page.domain,
            'quality_score': page.quality_score,
            'tags': page.tags,
            'user_rating': expected,
            'vote_count': stats.vote_count,
        })
    
    # 2. Fallback: If no trending pages, show high-quality crawled pages
    # FIX: Use outerjoin to fetch stats if they exist (even if < min_votes)
    if not trending:
        result = await db.execute(
            select(Page, PageStats)
            .outerjoin(PageStats, Page.id == PageStats.page_id)
            .where(Page.quality_score > 0.5)
            .order_by(desc(Page.quality_score))
            .limit(limit)
        )
        
        for page, stats in result.fetchall():
            # Determine votes/rating safely
            if stats:
                v_count = stats.vote_count
                # Only calculate rating if votes exist
                u_rating = (stats.alpha / (stats.alpha + stats.beta)) if v_count > 0 else None
            else:
                v_count = 0
                u_rating = None

            trending.append({
                'id': page.id,
                'url': page.url,
                'title': page.title,
                'description': page.description,
                'domain': page.domain,
                'quality_score': page.quality_score,
                'tags': page.tags,
                'user_rating': u_rating,
                'vote_count': v_count,  # Now reflects actual DB state
            })
    
    return trending
