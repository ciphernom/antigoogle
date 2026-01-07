"""
AntiGoogle - Decentralized Search Engine

A search engine with:
- Hybrid BM25 + semantic search
- Quality filtering (slop/spam detection)
- Bayesian ratings
- Privacy-preserving personalization via LSH
- Decentralized crawling via Nostr swarm

Modules:
- api: FastAPI web application
- crawler: Distributed crawler with VRF lottery
- database: PostgreSQL + pgvector models
- embedder: Sentence transformers + PCA
- filters: Spam, slop, quality detection
- nostr: Nostr protocol for swarm coordination
- ratings: Bayesian rating system
- search: Hybrid search implementation
- swarm: Event ingestion/publishing
- vdf: Verifiable Delay Function (anti-spam)
- vrf: Verifiable Random Function (domain lottery)
"""

__version__ = "2.0.0"
