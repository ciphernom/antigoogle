"""
Peer Discovery - Automatic swarm membership.

Protocol:
1. Node announces itself (Kind 4250)
2. Existing nodes send challenge (Kind 4251) 
3. New node solves PoW and responds (Kind 4252)
4. On valid response, add to trusted peers

No manual pubkey exchange needed.
"""
import secrets
import time
import logging
from typing import Dict, Set, Optional
from dataclasses import dataclass

from .nostr import (
    NostrService, EventKind,
    PeerAnnounceEvent, PeerChallengeEvent, PeerResponseEvent
)
from .config import get_settings

settings = get_settings()
logger = logging.getLogger("discovery")


def fnv1a_hash(challenge: str, nonce: int) -> int:
    """FNV-1a hash for PoW"""
    h = 2166136261
    for b in challenge.encode():
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    for i in range(4):
        h ^= (nonce >> (i * 8)) & 0xFF
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def count_leading_zeros(h: int) -> int:
    if h == 0:
        return 8
    z = 0
    while z < 8 and ((h >> (28 - z * 4)) & 0xF) == 0:
        z += 1
    return z


def solve_pow(challenge: str, difficulty: int) -> int:
    """Solve PoW challenge"""
    nonce = 0
    while True:
        if count_leading_zeros(fnv1a_hash(challenge, nonce)) >= difficulty:
            return nonce
        nonce += 1
        if nonce > 50_000_000:
            raise RuntimeError("PoW timeout")


def verify_pow(challenge: str, nonce: int, difficulty: int) -> bool:
    return count_leading_zeros(fnv1a_hash(challenge, nonce)) >= difficulty


@dataclass
class PendingChallenge:
    pubkey: str
    challenge: str
    difficulty: int
    created_at: float


class PeerDiscovery:
    """Manages automatic peer discovery."""
    
    def __init__(self, nostr: NostrService):
        self.nostr = nostr
        self.verified_peers: Set[str] = set()
        self.pending_challenges: Dict[str, PendingChallenge] = {}
        self.challenge_timeout = 60  # seconds
        self.announce_interval = 300  # 5 minutes
        self.last_announce = 0
        self.difficulty = settings.SWARM_POW_DIFFICULTY
    
    def register_handlers(self):
        """Register event handlers for discovery protocol."""
        
        @self.nostr.on_event(EventKind.PEER_ANNOUNCE)
        async def on_announce(event):
            await self._handle_announce(event)
        
        @self.nostr.on_event(EventKind.PEER_CHALLENGE)
        async def on_challenge(event):
            await self._handle_challenge(event)
        
        @self.nostr.on_event(EventKind.PEER_RESPONSE)
        async def on_response(event):
            await self._handle_response(event)
        
        logger.info("✅ Peer discovery handlers registered")
    
    async def announce(self, index_size: int = 0):
        """Announce our presence to the swarm."""
        event = PeerAnnounceEvent(
            version="1.0",
            capabilities=["crawl", "search", "vote"],
            index_size=index_size,
        )
        
        await self.nostr.publish(
            kind=EventKind.PEER_ANNOUNCE,
            content=event.to_content(),
            tags=[["t", "antigoogle-swarm"]],  # Hashtag for discovery
        )
        self.last_announce = time.time()
        logger.info("📢 Announced to swarm")
    
    async def _handle_announce(self, event: dict):
        """When we see a new peer announce, challenge them."""
        pubkey = event['pubkey']
        
        # Don't challenge ourselves or already verified peers
        if pubkey == self.nostr.public_key:
            return
        if pubkey in self.verified_peers:
            return
        
        # Don't re-challenge pending
        if pubkey in self.pending_challenges:
            pending = self.pending_challenges[pubkey]
            if time.time() - pending.created_at < self.challenge_timeout:
                return
        
        # Send challenge
        challenge = secrets.token_hex(16)
        self.pending_challenges[pubkey] = PendingChallenge(
            pubkey=pubkey,
            challenge=challenge,
            difficulty=self.difficulty,
            created_at=time.time(),
        )
        
        challenge_event = PeerChallengeEvent(
            target_pubkey=pubkey,
            challenge=challenge,
            difficulty=self.difficulty,
        )
        
        await self.nostr.publish(
            kind=EventKind.PEER_CHALLENGE,
            content=challenge_event.to_content(),
            tags=[["p", pubkey]],  # Tag target peer
        )
        logger.debug(f"🤝 Challenged peer {pubkey[:16]}...")
    
    async def _handle_challenge(self, event: dict):
        """When challenged, solve PoW and respond."""
        try:
            content = PeerChallengeEvent.from_content(event['content'])
        except Exception:
            return
        
        # Is this challenge for us?
        if content.target_pubkey != self.nostr.public_key:
            return
        
        challenger = event['pubkey']
        
        # Solve the PoW
        try:
            nonce = solve_pow(content.challenge, content.difficulty)
        except RuntimeError:
            logger.warning(f"Failed to solve challenge from {challenger[:16]}")
            return
        
        # Send response
        response = PeerResponseEvent(
            challenge=content.challenge,
            nonce=nonce,
            index_size=0,  # TODO: get actual index size
        )
        
        await self.nostr.publish(
            kind=EventKind.PEER_RESPONSE,
            content=response.to_content(),
            tags=[["p", challenger]],
        )
        logger.debug(f"✅ Responded to challenge from {challenger[:16]}")
    
    async def _handle_response(self, event: dict):
        """Verify challenge response and add peer."""
        pubkey = event['pubkey']
        
        # Check if we challenged this peer
        if pubkey not in self.pending_challenges:
            return
        
        pending = self.pending_challenges[pubkey]
        
        # Check timeout
        if time.time() - pending.created_at > self.challenge_timeout:
            del self.pending_challenges[pubkey]
            return
        
        try:
            content = PeerResponseEvent.from_content(event['content'])
        except Exception:
            return
        
        # Verify response
        if content.challenge != pending.challenge:
            return
        
        if not verify_pow(pending.challenge, content.nonce, pending.difficulty):
            logger.warning(f"Invalid PoW from {pubkey[:16]}")
            return
        
        # Success! Add to verified peers
        self.verified_peers.add(pubkey)
        del self.pending_challenges[pubkey]
        
        # Also add to nostr service trusted set
        self.nostr.trusted_pubkeys.add(pubkey)
        
        logger.info(f"✅ Verified peer: {pubkey[:16]}... (total: {len(self.verified_peers)})")
    
    def cleanup(self):
        """Remove expired pending challenges."""
        now = time.time()
        expired = [k for k, v in self.pending_challenges.items() 
                   if now - v.created_at > self.challenge_timeout]
        for k in expired:
            del self.pending_challenges[k]
    
    def get_stats(self) -> dict:
        return {
            "verified_peers": len(self.verified_peers),
            "pending_challenges": len(self.pending_challenges),
            "last_announce": self.last_announce,
        }


_discovery: Optional[PeerDiscovery] = None

def get_peer_discovery(nostr: NostrService = None) -> Optional[PeerDiscovery]:
    global _discovery
    if _discovery is None and nostr is not None:
        _discovery = PeerDiscovery(nostr)
    return _discovery


async def init_peer_discovery(nostr: NostrService, index_size: int = 0):
    """Initialize and start peer discovery."""
    global _discovery
    _discovery = PeerDiscovery(nostr)
    _discovery.register_handlers()
    await _discovery.announce(index_size)
    return _discovery
