"""
Graph Topology Analysis - Detecting "Indie" vs "Corporate" sites.

Uses A³ (Adjacency Matrix Cubed) to count triangles.
- Stars (hubs): High degree, zero triangles → C ≈ 0
- Meshes (communities): Triangles exist → C > 0

C(v) = 2T(v) / k(k-1) where T = triangles, k = degree
"""
import logging
from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("topology")


async def calculate_domain_clustering(db: AsyncSession) -> dict:
    """
    Calculate clustering coefficient for all domains using SQL.
    This is the A³ diagonal computation.
    
    Returns dict of {domain: clustering_coefficient}
    """
    # SQL performs the matrix multiplication via JOINs
    # A -> B -> C -> A (closed triangle)
    query = text("""
        WITH 
        adj AS (
            SELECT DISTINCT source, target 
            FROM domain_links 
            WHERE source != target
        ),
        
        -- Degrees (for normalization)
        degrees AS (
            SELECT source, COUNT(DISTINCT target) as k 
            FROM adj 
            GROUP BY source
            HAVING COUNT(DISTINCT target) BETWEEN 2 AND 500
        ),
        
        -- A² paths (A -> B -> C)
        paths2 AS (
            SELECT a.source, b.target
            FROM adj a
            JOIN adj b ON a.target = b.source
            JOIN degrees d ON a.source = d.source
            WHERE a.source != b.target
        ),
        
        -- A³ diagonal (triangles: A -> B -> C -> A)
        triangles AS (
            SELECT p.source as domain, COUNT(*) as tri_count
            FROM paths2 p
            JOIN adj c ON p.target = c.source AND c.target = p.source
            GROUP BY p.source
        )
        
        SELECT 
            t.domain,
            t.tri_count,
            d.k as degree,
            CASE 
                WHEN d.k > 1 THEN (2.0 * t.tri_count) / (d.k * (d.k - 1))
                ELSE 0 
            END as clustering_coef
        FROM triangles t
        JOIN degrees d ON t.domain = d.source
        WHERE t.tri_count > 0
        ORDER BY clustering_coef DESC
    """)
    
    try:
        result = await db.execute(query)
        rows = result.fetchall()
        return {row.domain: row.clustering_coef for row in rows}
    except Exception as e:
        logger.error(f"Clustering calculation failed: {e}")
        return {}


async def boost_indie_domains(db: AsyncSession):
    """
    Boost quality_score for domains with high clustering coefficient.
    These are "community" sites, not hubs.
    """
    from .database import Page
    
    coefficients = await calculate_domain_clustering(db)
    
    boosted = 0
    for domain, coef in coefficients.items():
        # The "Indie" sweet spot: 0.05 < C < 0.6
        # C ≈ 0: Hub/aggregator (Reddit, HN)
        # C ≈ 1: Too tight (possibly spam ring)
        # C ∈ [0.05, 0.6]: Organic community
        
        if 0.05 < coef < 0.6:
            boost = min(0.3, coef)  # Max +0.3 boost
            await db.execute(
                update(Page)
                .where(Page.domain == domain)
                .values(quality_score=Page.quality_score + boost)
            )
            boosted += 1
            logger.info(f"🌿 Boosted {domain} (C={coef:.3f})")
    
    await db.commit()
    logger.info(f"Boosted {boosted} indie domains")
    return boosted


async def get_domain_stats(db: AsyncSession, domain: str) -> dict:
    """Get topology stats for a single domain."""
    query = text("""
        WITH adj AS (
            SELECT source, target FROM domain_links 
            WHERE source = :domain OR target = :domain
        )
        SELECT 
            COUNT(DISTINCT CASE WHEN source = :domain THEN target END) as outlinks,
            COUNT(DISTINCT CASE WHEN target = :domain THEN source END) as inlinks
        FROM adj
    """)
    
    result = await db.execute(query, {"domain": domain})
    row = result.fetchone()
    
    return {
        "domain": domain,
        "outlinks": row.outlinks if row else 0,
        "inlinks": row.inlinks if row else 0,
    }
