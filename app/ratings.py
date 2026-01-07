"""
Bayesian Rating System - Beta-Binomial Conjugate Prior
"""
import math
from typing import Dict, Tuple, Optional
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError

from .database import Rating, PageStats

# ============================================================
# BAYESIAN RATINGS
# ============================================================
class BayesianRatings:
    """
    Beta-Binomial conjugate prior rating system.
    
    Prior: Beta(alpha_0, beta_0)
    Posterior: Beta(alpha_0 + good_votes, beta_0 + bad_votes)
    
    This naturally handles:
    - Cold start (prior dominates with few votes)
    - Uncertainty quantification
    - Smooth transitions as votes accumulate
    """
    
    # Prior parameters (slightly optimistic)
    ALPHA_0 = 2.0
    BETA_0 = 2.0
    
    @staticmethod
    async def get_stats(db: AsyncSession, page_id: int) -> Dict:
        """Get rating stats for a page"""
        result = await db.execute(
            select(PageStats).where(PageStats.page_id == page_id)
        )
        stats = result.scalar_one_or_none()
        
        if stats:
            alpha = stats.alpha
            beta = stats.beta
            vote_count = stats.vote_count
        else:
            alpha = BayesianRatings.ALPHA_0
            beta = BayesianRatings.BETA_0
            vote_count = 0
        
        # Compute statistics
        expected = alpha / (alpha + beta)
        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        uncertainty = math.sqrt(variance)
        
        # Credible interval (approximation)
        z = 1.645  # 90% interval
        credible_low = max(0, expected - z * uncertainty)
        credible_high = min(1, expected + z * uncertainty)
        
        return {
            'alpha': alpha,
            'beta': beta,
            'expected': expected,
            'uncertainty': uncertainty,
            'credible_low': credible_low,
            'credible_high': credible_high,
            'total_votes': vote_count,
        }
    
    @staticmethod
    async def can_vote(db: AsyncSession, page_id: int, fingerprint: str) -> bool:
        """Check if user can vote (hasn't voted before)"""
        result = await db.execute(
            select(Rating)
            .where(Rating.page_id == page_id, Rating.fingerprint == fingerprint)
        )
        return result.scalar_one_or_none() is None
    
    @staticmethod
    async def record_vote(
        db: AsyncSession,
        page_id: int,
        is_good: bool,
        fingerprint: str,
        weight: float = 1.0
    ) -> Dict:
        """
        Record a vote.
        
        Returns:
            Dict with stats and 'recorded' bool indicating if vote was new
        """
        # Check if already voted
        if not await BayesianRatings.can_vote(db, page_id, fingerprint):
            stats = await BayesianRatings.get_stats(db, page_id)
            stats['recorded'] = False
            return stats
        
        try:
            # Record the vote
            rating = Rating(
                page_id=page_id,
                fingerprint=fingerprint,
                is_good=is_good,
                weight=weight
            )
            db.add(rating)
            
            # Update page stats using upsert
            alpha_add = weight if is_good else 0
            beta_add = weight if not is_good else 0
            
            stmt = insert(PageStats).values(
                page_id=page_id,
                alpha=BayesianRatings.ALPHA_0 + alpha_add,
                beta=BayesianRatings.BETA_0 + beta_add,
                vote_count=1
            ).on_conflict_do_update(
                index_elements=['page_id'],
                set_={
                    'alpha': PageStats.alpha + alpha_add,
                    'beta': PageStats.beta + beta_add,
                    'vote_count': PageStats.vote_count + 1
                }
            )
            await db.execute(stmt)
            await db.commit()
            
            stats = await BayesianRatings.get_stats(db, page_id)
            stats['recorded'] = True
            return stats
            
        except IntegrityError:
            # Race condition - another request already recorded this vote
            await db.rollback()
            stats = await BayesianRatings.get_stats(db, page_id)
            stats['recorded'] = False
            return stats
    
    @staticmethod
    def get_quality_score(alpha: float, beta: float) -> float:
        """
        Get quality score for search ranking.
        Uses lower bound of 90% credible interval (conservative).
        """
        expected = alpha / (alpha + beta)
        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        uncertainty = math.sqrt(variance)
        return max(0, expected - 1.645 * uncertainty)

# ============================================================
# PERSONALIZATION SIGNALS
# ============================================================
async def record_signal(
    db: AsyncSession,
    cluster_id: int,
    page_id: int,
    score: float,
    fingerprint: str
) -> None:
    """
    Record a personalization signal.
    
    Uses exponential moving average for signals.
    """
    from .database import ClusterSignal
    
    # EMA factor
    alpha = 0.3
    
    stmt = insert(ClusterSignal).values(
        cluster_id=cluster_id,
        page_id=page_id,
        score=score,
        signal_count=1
    ).on_conflict_do_update(
        index_elements=['cluster_id', 'page_id'],
        set_={
            'score': ClusterSignal.score * (1 - alpha) + score * alpha,
            'signal_count': ClusterSignal.signal_count + 1
        }
    )
    await db.execute(stmt)
    await db.commit()
