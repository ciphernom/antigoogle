"""
Database Models - PostgreSQL + pgvector
"""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, Boolean, Index,
    ForeignKey, UniqueConstraint, func, text
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from .config import get_settings

settings = get_settings()

class Base(DeclarativeBase):
    pass

class TrustHash(Base):
    """
    Trust DAG nodes - stores the Merkle DAG of domain trust.
    """
    __tablename__ = "trust_hashes"
    
    domain: Mapped[str] = mapped_column(String(255), primary_key=True)
    trust_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    parent_domains: Mapped[list] = mapped_column(ARRAY(String(255)), default=[])
    parent_hashes: Mapped[list] = mapped_column(ARRAY(String(64)), default=[])
    depth: Mapped[int] = mapped_column(Integer, default=0)
    trust_score: Mapped[float] = mapped_column(Float, default=0.0)

    __table_args__ = (
        Index('ix_trust_hash_lookup', 'trust_hash'),
        Index('ix_trust_depth', 'depth'),
    )

class Page(Base):
    """Indexed pages"""
    __tablename__ = "pages"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str] = mapped_column(String(2048), unique=True, index=True)
    url_hash: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    title_hash: Mapped[str] = mapped_column(String(32), index=True)  # For dedup
    
    title: Mapped[str] = mapped_column(String(500))
    description: Mapped[Optional[str]] = mapped_column(Text)
    domain: Mapped[str] = mapped_column(String(255), index=True)
    
    # Quality metrics
    quality_score: Mapped[float] = mapped_column(Float, default=0.5)
    slop_score: Mapped[float] = mapped_column(Float, default=0.0)
    spam_score: Mapped[float] = mapped_column(Float, default=0.0)
    domain_trust: Mapped[float] = mapped_column(Float, default=0.5)
    
    # Content info (not storing full content - just metadata)
    content_length: Mapped[int] = mapped_column(Integer, default=0)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Vector embedding (64 dimensions after PCA, if doing PCA)
    embedding: Mapped[List[float]] = mapped_column(Vector(384))
    
    # Tags/categories
    tags: Mapped[Optional[str]] = mapped_column(String(500))
    
    # Timestamps
    crawled_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    
    # Relationships
    #terms: Mapped[List["Term"]] = relationship(back_populates="page", cascade="all, delete-orphan")
    ratings: Mapped[List["Rating"]] = relationship(back_populates="page", cascade="all, delete-orphan")
    
    #MoE metadata
    expert_data = Column(JSONB, default={})

    # 1. ADD SEARCH VECTOR COLUMN
    search_vector = mapped_column(TSVECTOR)

    # Indexes for fast search
    __table_args__ = (
        Index('ix_pages_quality', 'quality_score'),
        Index('ix_pages_domain_quality', 'domain', 'quality_score'),
        Index('ix_pages_crawled', 'crawled_at'),
        # GIN INDEX FOR FULL-TEXT SEARCH
        Index('ix_pages_search_vector', 'search_vector', postgresql_using='gin'),
        # HNSW INDEX FOR VECTOR SEARCH (O(log n) instead of O(n))
        Index(
            'ix_pages_embedding_hnsw',
            'embedding',
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': 'vector_cosine_ops'}
        ),
    )


# 4. ADD VOCABULARY CLASS
class Vocabulary(Base):
    """
    Dictionary of known words for spell checking.
    Significantly smaller than the Inverted Index (Term) table.
    """
    __tablename__ = "vocabulary"
    
    word: Mapped[str] = mapped_column(String(100), primary_key=True)
    doc_count: Mapped[int] = mapped_column(Integer, default=1) 
    
    __table_args__ = (
        # Keep the trigram index for fuzzy matching
        Index('ix_vocab_trgm', 'word', postgresql_ops={'word': 'gin_trgm_ops'}, postgresql_using='gin'),
    )

class CrawlQueue(Base):
    """URL queue for crawling"""
    __tablename__ = "crawl_queue"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str] = mapped_column(String(2048), unique=True, index=True)
    priority: Mapped[float] = mapped_column(Float, default=0.5, index=True)
    source_page_id: Mapped[Optional[int]] = mapped_column(ForeignKey("pages.id", ondelete="SET NULL"))
    added_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_queue_priority_added', 'priority', 'added_at'),
    )

class Rating(Base):
    """User ratings for pages"""
    __tablename__ = "ratings"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    page_id: Mapped[int] = mapped_column(ForeignKey("pages.id", ondelete="CASCADE"), index=True)
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    is_good: Mapped[bool] = mapped_column(Boolean)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    page: Mapped["Page"] = relationship(back_populates="ratings")
    
    __table_args__ = (
        UniqueConstraint('page_id', 'fingerprint', name='uq_rating_page_fp'),
    )

class PageStats(Base):
    """Aggregated stats per page (Bayesian ratings)"""
    __tablename__ = "page_stats"
    
    page_id: Mapped[int] = mapped_column(ForeignKey("pages.id", ondelete="CASCADE"), primary_key=True)
    alpha: Mapped[float] = mapped_column(Float, default=2.0)  # Prior + positive votes
    beta: Mapped[float] = mapped_column(Float, default=2.0)   # Prior + negative votes
    vote_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ClusterSignal(Base):
    """Personalization cluster signals"""
    __tablename__ = "cluster_signals"
    
    cluster_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    page_id: Mapped[int] = mapped_column(ForeignKey("pages.id", ondelete="CASCADE"), primary_key=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    signal_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_cluster_signals_cluster', 'cluster_id'),
    )

class SpamTraining(Base):
    """Spam filter training data"""
    __tablename__ = "spam_training"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    token: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    spam_count: Mapped[int] = mapped_column(Integer, default=0)
    ham_count: Mapped[int] = mapped_column(Integer, default=0)

class PowChallenge(Base):
    """Active PoW challenges"""
    __tablename__ = "pow_challenges"
    
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    challenge: Mapped[str] = mapped_column(String(64))
    difficulty: Mapped[int] = mapped_column(Integer)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)


class DomainLink(Base):
    """
    Domain-to-domain link graph for topology analysis.
    Used by topology.py to detect indie vs corporate sites via clustering coefficient.
    """
    __tablename__ = "domain_links"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(255))  # Index defined in __table_args__
    target: Mapped[str] = mapped_column(String(255))  # Index defined in __table_args__
    link_count: Mapped[int] = mapped_column(Integer, default=1)
    first_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('source', 'target', name='uq_domain_link'),
        Index('ix_domain_links_source', 'source'),
        Index('ix_domain_links_target', 'target'),
    )

# Database connection management
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    echo=False,
)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    """Initialize database with pgvector and pg_trgm extensions"""
    async with engine.begin() as conn:
        # Create pgvector extension (ignore if exists)
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            # Enable trigram extension for spell checking
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        except Exception:
            pass  # Extension already exists
        # Create tables
        await conn.run_sync(Base.metadata.create_all)

async def get_db() -> AsyncSession:
    """Dependency for getting DB session"""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()
