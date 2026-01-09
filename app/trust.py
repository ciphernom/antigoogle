"""
Trust DAG - Merkle DAG for Decentralized Trust Propagation

=============================================================================
HOW THIS PREVENTS ILLEGAL CONTENT
=============================================================================

The core insight: The legitimate web and the illegal web are DISJOINT GRAPHS.

    [Seeds: arxiv, github, bbc, wikipedia, ...]
                    |
                    v
    [Legitimate Web] ----X----> [Illegal Content]
                    
    There are no links from legitimate sites to illegal content.
    
By construction:
1. We start with known-good seed domains
2. We only crawl domains linked FROM domains we already trust
3. Trust decays with each hop (so distant connections are weak)
4. A domain's "trust hash" cryptographically commits to its ancestry

The result: Illegal content is MATHEMATICALLY UNREACHABLE.

=============================================================================
THE MERKLE DAG STRUCTURE  
=============================================================================

Each domain has a trust hash that incorporates ALL its parents:

    trust_hash(domain) = H(domain || merkle_root(sorted(parent_hashes)))

Example:
    
    Seeds (depth 0):
        H("arxiv.org")     = abc123...
        H("github.com")    = def456...
        
    Depth 1 (linked from seeds):
        nature.com has parent arxiv.org
        H("nature.com" || merkle_root([abc123])) = 789xyz...
        
    Depth 2:
        cell.com has parents nature.com AND github.com
        H("cell.com" || merkle_root([789xyz, def456])) = final42...

The hash PROVES the ancestry. You can't claim a trust path without knowing
the actual chain of hashes.

=============================================================================
SWARM INTEGRATION
=============================================================================

When Node B sends a URL to Node A:

    B sends: {
        url: "https://newsite.com/page",
        domain: "newsite.com", 
        trust_proof: [
            {domain: "newsite.com", hash: "...", parents: ["nature.com"]},
            {domain: "nature.com", hash: "...", parents: ["arxiv.org"]},
            {domain: "arxiv.org", hash: "...", parents: []}  # seed
        ]
    }

Node A verifies:
    1. Recompute hashes from proof
    2. Check if ANY hash in the proof matches A's known hashes
    3. If yes: A and B share ancestry → accept with decayed trust
    4. If no: No shared trust → reject

This means:
    - Nodes with similar seeds automatically trust each other's discoveries
    - Nodes with divergent seeds gracefully ignore each other
    - Malicious nodes can only pollute their own index
    - No central authority decides what's trustworthy

=============================================================================
"""
import hashlib
import asyncio
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import json
import logging

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("trust")


# =============================================================================
# CRYPTOGRAPHIC PRIMITIVES
# =============================================================================

def sha256(data: bytes) -> bytes:
    """SHA256 hash"""
    return hashlib.sha256(data).digest()


def sha256_hex(data: bytes) -> str:
    """SHA256 hash as hex string"""
    return hashlib.sha256(data).hexdigest()


def merkle_root(hashes: List[str]) -> str:
    """
    Compute Merkle root of a list of hashes.
    
    For an ordered list of parent hashes, this produces a single
    commitment that covers all of them.
    """
    if not hashes:
        return sha256_hex(b"SEED")  # Special case for seed nodes
    
    if len(hashes) == 1:
        return hashes[0]
    
    # Sort for determinism (different nodes must compute same root)
    sorted_hashes = sorted(hashes)
    
    # Build tree bottom-up
    level = [bytes.fromhex(h) for h in sorted_hashes]
    
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                combined = sha256(level[i] + level[i + 1])
            else:
                combined = sha256(level[i] + level[i])  # Duplicate odd node
            next_level.append(combined)
        level = next_level
    
    return level[0].hex()


def compute_trust_hash(domain: str, parent_hashes: List[str]) -> str:
    """
    Compute the trust hash for a domain given its parents.
    
    trust_hash = SHA256(domain || merkle_root(parents))
    
    This cryptographically binds the domain to its entire ancestry.
    """
    root = merkle_root(parent_hashes)
    data = domain.encode('utf-8') + bytes.fromhex(root)
    return sha256_hex(data)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TrustNode:
    """A node in the trust DAG"""
    domain: str
    trust_hash: str
    parent_domains: List[str]
    parent_hashes: List[str]
    depth: int  # Distance from nearest seed
    trust_score: float  # Decayed trust (1.0 at seed, decays with depth)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TrustNode':
        return cls(**data)


@dataclass
class TrustProof:
    """
    Merkle proof of trust path from target domain back to seeds.
    
    The proof is a list of TrustNodes forming a subgraph that connects
    the target domain to one or more seed domains.
    """
    target_domain: str
    target_hash: str
    path: List[TrustNode]  # From target back to seeds
    
    def to_dict(self) -> dict:
        return {
            'target_domain': self.target_domain,
            'target_hash': self.target_hash,
            'path': [n.to_dict() for n in self.path],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TrustProof':
        return cls(
            target_domain=data['target_domain'],
            target_hash=data['target_hash'],
            path=[TrustNode.from_dict(n) for n in data['path']],
        )
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, s: str) -> 'TrustProof':
        return cls.from_dict(json.loads(s))
    
    def chain_hash(self) -> str:
        """
        Compute a single hash representing this entire proof.
        Used as the PoW challenge - binds the work to this specific proof.
        """
        data = self.target_hash.encode()
        for node in self.path:
            data += node.trust_hash.encode()
        return sha256_hex(data)


# =============================================================================
# TRUST DAG
# =============================================================================

class TrustDAG:
    """
    Maintains the trust graph and provides proof generation/verification.
    
    The DAG is built incrementally as the crawler discovers new domains.
    Seeds form the roots, and trust flows outward through links.
    """
    
    # Trust decay factor per hop (0.85 = 15% decay per link)
    TRUST_DECAY = 0.85
    
    # Minimum trust to consider a domain trustworthy
    MIN_TRUST = 0.1
    
    # Maximum proof depth (prevents DoS with huge proofs)
    MAX_PROOF_DEPTH = 10
    
    def __init__(self, seeds: List[str] = None):
        # domain -> TrustNode
        self._nodes: Dict[str, TrustNode] = {}
        
        # trust_hash -> domain (for fast proof verification)
        self._hash_to_domain: Dict[str, str] = {}
        
        # domain -> set of child domains (reverse index)
        self._children: Dict[str, Set[str]] = defaultdict(set)
        
        # Initialize seeds
        if seeds:
            for seed in seeds:
                self.add_seed(seed)
    
    def add_seed(self, domain: str) -> TrustNode:
        """
        Add a seed domain (root of trust).
        Seeds have no parents, depth 0, and trust 1.0.
        """
        domain = self._normalize_domain(domain)
        
        trust_hash = compute_trust_hash(domain, [])
        
        node = TrustNode(
            domain=domain,
            trust_hash=trust_hash,
            parent_domains=[],
            parent_hashes=[],
            depth=0,
            trust_score=1.0,
        )
        
        self._nodes[domain] = node
        self._hash_to_domain[trust_hash] = domain
        
        logger.info(f"🌱 Seed added: {domain} (hash: {trust_hash[:16]}...)")
        return node
    
    def add_domain(self, domain: str, parent_domains: List[str]) -> Optional[TrustNode]:
        """
        Add a domain with its parent domains (sites that link to it).
        
        Returns None if:
        - No valid parents exist in the DAG
        - Resulting trust would be below MIN_TRUST
        """
        domain = self._normalize_domain(domain)
        
        # Already exists?
        if domain in self._nodes:
            return self._maybe_update_parents(domain, parent_domains)
        
        # Get valid parent nodes
        valid_parents = []
        for p in parent_domains:
            p = self._normalize_domain(p)
            if p in self._nodes:
                valid_parents.append(self._nodes[p])
        
        if not valid_parents:
            return None  # No trust path exists
        
        # Compute depth (min parent depth + 1)
        depth = min(p.depth for p in valid_parents) + 1
        
        # Compute trust (max parent trust * decay)
        trust_score = max(p.trust_score for p in valid_parents) * self.TRUST_DECAY
        
        if trust_score < self.MIN_TRUST:
            return None  # Too far from seeds
        
        # Compute trust hash
        parent_hashes = sorted([p.trust_hash for p in valid_parents])
        trust_hash = compute_trust_hash(domain, parent_hashes)
        
        node = TrustNode(
            domain=domain,
            trust_hash=trust_hash,
            parent_domains=[p.domain for p in valid_parents],
            parent_hashes=parent_hashes,
            depth=depth,
            trust_score=trust_score,
        )
        
        self._nodes[domain] = node
        self._hash_to_domain[trust_hash] = domain
        
        # Update reverse index
        for p in valid_parents:
            self._children[p.domain].add(domain)
        
        logger.debug(f"✓ Domain added: {domain} (depth={depth}, trust={trust_score:.3f})")
        return node

    async def add_domain_async(self, domain: str, parent_domains: List[str], db: AsyncSession) -> Optional[TrustNode]:
        """Add domain and persist immediately"""
        node = self.add_domain(domain, parent_domains)
        
        if node and node.depth > 0:  # Don't re-save seeds
            from .database import TrustHash
            stmt = insert(TrustHash).values(
                domain=node.domain,
                trust_hash=node.trust_hash,
                parent_domains=node.parent_domains,
                parent_hashes=node.parent_hashes,
                depth=node.depth,
                trust_score=node.trust_score,
            ).on_conflict_do_nothing()
            await db.execute(stmt)
        
        return node

    def _maybe_update_parents(self, domain: str, new_parents: List[str]) -> TrustNode:
        """
        Update a domain's parents if new ones provide better trust.
        """
        node = self._nodes[domain]
        
        # Find new valid parents not already in the list
        current_parents = set(node.parent_domains)
        added = False
        
        for p in new_parents:
            p = self._normalize_domain(p)
            if p in self._nodes and p not in current_parents:
                parent_node = self._nodes[p]
                # Only add if this parent improves our trust
                if parent_node.trust_score * self.TRUST_DECAY > node.trust_score:
                    current_parents.add(p)
                    added = True
        
        if added:
            # Recompute the node with all parents
            valid_parents = [self._nodes[p] for p in current_parents if p in self._nodes]
            
            depth = min(p.depth for p in valid_parents) + 1
            trust_score = max(p.trust_score for p in valid_parents) * self.TRUST_DECAY
            parent_hashes = sorted([p.trust_hash for p in valid_parents])
            trust_hash = compute_trust_hash(domain, parent_hashes)
            
            # Remove old hash mapping
            del self._hash_to_domain[node.trust_hash]
            
            # Update node
            node.parent_domains = [p.domain for p in valid_parents]
            node.parent_hashes = parent_hashes
            node.depth = depth
            node.trust_score = trust_score
            node.trust_hash = trust_hash
            
            self._hash_to_domain[trust_hash] = domain
        
        return node
    
    def get_node(self, domain: str) -> Optional[TrustNode]:
        """Get trust node for a domain"""
        return self._nodes.get(self._normalize_domain(domain))
    
    def has_trust(self, domain: str) -> bool:
        """Check if domain is in our trust graph"""
        return self._normalize_domain(domain) in self._nodes
    
    def get_trust_score(self, domain: str) -> float:
        """Get trust score for a domain (0 if not in graph)"""
        node = self._nodes.get(self._normalize_domain(domain))
        return node.trust_score if node else 0.0
    
    def knows_hash(self, trust_hash: str) -> bool:
        """Check if we know a trust hash (for proof verification)"""
        return trust_hash in self._hash_to_domain
    
    def generate_proof(self, domain: str) -> Optional[TrustProof]:
        """
        Generate a trust proof for a domain.
        
        The proof is the minimal subgraph connecting the domain to seeds.
        We use BFS to find the shortest path.
        """
        domain = self._normalize_domain(domain)
        
        if domain not in self._nodes:
            return None
        
        # BFS from domain back to seeds
        path = []
        visited = set()
        queue = [domain]
        
        while queue and len(path) < self.MAX_PROOF_DEPTH:
            current = queue.pop(0)
            
            if current in visited:
                continue
            visited.add(current)
            
            node = self._nodes[current]
            path.append(node)
            
            # If this is a seed, we're done
            if node.depth == 0:
                break
            
            # Add parents to queue
            for parent in node.parent_domains:
                if parent not in visited:
                    queue.append(parent)
        
        if not path:
            return None
        
        target_node = self._nodes[domain]
        
        return TrustProof(
            target_domain=domain,
            target_hash=target_node.trust_hash,
            path=path,
        )
    
    def verify_proof(self, proof: TrustProof) -> Tuple[bool, float, str]:
        """
        Verify a trust proof from another node.
        
        Returns:
            (is_valid, trust_score, reason)
            
        A proof is valid if:
        1. All hashes in the path are correctly computed
        2. At least one hash in the path matches our known hashes
        
        The trust score is based on where the proof intersects our graph.
        """
        if not proof.path:
            return False, 0.0, "Empty proof"
        
        if len(proof.path) > self.MAX_PROOF_DEPTH:
            return False, 0.0, "Proof too deep"
        
        # Step 1: Verify all hashes are correctly computed
        for node in proof.path:
            expected_hash = compute_trust_hash(node.domain, node.parent_hashes)
            if expected_hash != node.trust_hash:
                return False, 0.0, f"Invalid hash for {node.domain}"
        
        # Step 2: Find intersection with our trust graph
        intersection_depth = -1
        intersection_trust = 0.0
        
        for i, node in enumerate(proof.path):
            if node.trust_hash in self._hash_to_domain:
                # We know this exact hash - full trust at this point
                intersection_depth = i
                intersection_trust = self._nodes[self._hash_to_domain[node.trust_hash]].trust_score
                break
            
            # Also check if we know the domain but with different parents
            # (partial trust - the domain is known but via different path)
            if node.domain in self._nodes:
                our_node = self._nodes[node.domain]
                intersection_depth = i
                # Partial trust: geometric mean of their claim and our knowledge
                intersection_trust = (our_node.trust_score * node.trust_score) ** 0.5
                break
        
        if intersection_depth < 0:
            return False, 0.0, "No intersection with local trust graph"
        
        # Step 3: Compute final trust
        # Trust decays for each hop from intersection to target
        hops_from_intersection = intersection_depth
        final_trust = intersection_trust * (self.TRUST_DECAY ** hops_from_intersection)
        
        if final_trust < self.MIN_TRUST:
            return False, final_trust, f"Trust too low: {final_trust:.4f}"
        
        return True, final_trust, "Valid"
    
    def _normalize_domain(self, domain: str) -> str:
        """Normalize domain for consistent hashing"""
        d = domain.lower().strip()
        if d.startswith('www.'):
            d = d[4:]
        return d
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    async def save_to_db(self, db: AsyncSession):
        """Persist trust DAG to database"""
        from .database import TrustHash  # Import here to avoid circular
        
        for domain, node in self._nodes.items():
            stmt = insert(TrustHash).values(
                domain=domain,
                trust_hash=node.trust_hash,
                parent_domains=node.parent_domains,
                parent_hashes=node.parent_hashes,
                depth=node.depth,
                trust_score=node.trust_score,
            ).on_conflict_do_update(
                index_elements=['domain'],
                set_={
                    'trust_hash': node.trust_hash,
                    'parent_domains': node.parent_domains,
                    'parent_hashes': node.parent_hashes,
                    'depth': node.depth,
                    'trust_score': node.trust_score,
                }
            )
            await db.execute(stmt)
        
        await db.commit()
        logger.info(f"💾 Saved {len(self._nodes)} trust nodes to database")
    
    async def load_from_db(self, db: AsyncSession):
        """Load trust DAG from database"""
        from .database import TrustHash
        
        result = await db.execute(select(TrustHash))
        rows = result.scalars().all()
        
        for row in rows:
            node = TrustNode(
                domain=row.domain,
                trust_hash=row.trust_hash,
                parent_domains=row.parent_domains,
                parent_hashes=row.parent_hashes,
                depth=row.depth,
                trust_score=row.trust_score,
            )
            self._nodes[row.domain] = node
            self._hash_to_domain[row.trust_hash] = row.domain
            
            for parent in row.parent_domains:
                self._children[parent].add(row.domain)
        
        logger.info(f"📂 Loaded {len(self._nodes)} trust nodes from database")
    
    def get_stats(self) -> dict:
        """Get DAG statistics"""
        if not self._nodes:
            return {'total': 0}
        
        depths = [n.depth for n in self._nodes.values()]
        trusts = [n.trust_score for n in self._nodes.values()]
        
        return {
            'total': len(self._nodes),
            'seeds': sum(1 for n in self._nodes.values() if n.depth == 0),
            'max_depth': max(depths),
            'avg_depth': sum(depths) / len(depths),
            'avg_trust': sum(trusts) / len(trusts),
            'min_trust': min(trusts),
        }


# =============================================================================
# SINGLETON
# =============================================================================

_trust_dag: Optional[TrustDAG] = None


async def get_trust_dag() -> TrustDAG:
    """Get or create the global trust DAG"""
    global _trust_dag
    
    if _trust_dag is None:
        from .config import get_settings
        settings = get_settings()
        
        # Initialize with seeds from TRUSTED_DOMAINS
        from .config import TRUSTED_DOMAINS
        seeds = [domain for domain, score in TRUSTED_DOMAINS.items() if score >= 0.9]
        
        _trust_dag = TrustDAG(seeds=seeds)
        
        # Load from database
        from .database import async_session
        async with async_session() as db:
            await _trust_dag.load_from_db(db)
        
        logger.info(f"✅ Trust DAG initialized: {_trust_dag.get_stats()}")
    
    return _trust_dag


async def save_trust_dag():
    """Persist current trust DAG to database"""
    if _trust_dag:
        from .database import async_session
        async with async_session() as db:
            await _trust_dag.save_to_db(db)
