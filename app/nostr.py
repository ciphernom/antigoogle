"""
Nostr Service - Decentralized Crawl Coordination

Event Kinds:
- 4242: URL Discovery (new URL found, requesting crawl)
- 4243: Crawl Result (page indexed, metadata + embedding)
- 4244: Vote Signal (user quality rating)
- 4245: Genesis Config (shared PCA/LSH matrices)
"""
import asyncio
import json
import time
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Callable, Set, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict, OrderedDict
from enum import IntEnum

import websockets
from websockets.exceptions import ConnectionClosed
from secp256k1 import PrivateKey, PublicKey

from .config import get_settings

settings = get_settings()
logger = logging.getLogger("nostr")

# ============================================================
# NOSTR EVENT KINDS
# ============================================================
class EventKind(IntEnum):
    URL_DISCOVERY = 4242
    CRAWL_RESULT = 4243
    VOTE_SIGNAL = 4244
    GENESIS_CONFIG = 4245
    # Peer discovery
    PEER_ANNOUNCE = 4250    # "I exist, here's my pubkey"
    PEER_CHALLENGE = 4251   # "Prove you're real with this PoW"
    PEER_RESPONSE = 4252    # "Here's my solved PoW"


# ============================================================
# NOSTR PRIMITIVES
# ============================================================
def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def compute_event_id(event: dict) -> str:
    """Compute event ID as sha256 of serialized event"""
    serialized = json.dumps([
        0,
        event['pubkey'],
        event['created_at'],
        event['kind'],
        event['tags'],
        event['content']
    ], separators=(',', ':'), ensure_ascii=False)
    return sha256(serialized.encode()).hex()


def sign_event(event: dict, private_key: PrivateKey) -> str:
    """Sign event with Schnorr signature"""
    event_id = bytes.fromhex(event['id'])
    sig = private_key.schnorr_sign(event_id, None, raw=True)
    return sig.hex()


def verify_signature(event: dict) -> bool:
    """Verify event signature"""
    try:
        pubkey = PublicKey(bytes.fromhex('02' + event['pubkey']), raw=True)
        sig = bytes.fromhex(event['sig'])
        event_id = bytes.fromhex(event['id'])
        return pubkey.schnorr_verify(event_id, sig, None, raw=True)
    except Exception as e:
        logger.warning(f"Signature verification failed: {e}")
        return False


# ============================================================
# EVENT DATA STRUCTURES
# ============================================================
@dataclass
class URLDiscoveryEvent:
    """Kind 4242: Request to crawl a URL with Trust Proof"""
    url: str
    domain: str
    priority: float
    
    # Trust proof: list of nodes from target back to seeds
    trust_proof: List[Dict] = field(default_factory=list)
    # Hash of the trust proof (binds PoW to this specific proof)
    proof_hash: str = ""
    # PoW solution for the proof_hash
    pow_nonce: Optional[int] = None
    
    # Legacy fields
    source_url: Optional[str] = None
    pow_challenge: Optional[str] = None 

    def to_content(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_content(cls, content: str) -> 'URLDiscoveryEvent':
        data = json.loads(content)
        data.setdefault('trust_proof', [])
        data.setdefault('proof_hash', '')
        return cls(**data)


@dataclass
class TrustProofNode:
    """A node in a trust proof (for serialization over Nostr)"""
    domain: str
    trust_hash: str
    parent_hashes: List[str]
    depth: int
    trust_score: float

@dataclass
class CrawlResultEvent:
    """Kind 4243: Crawl Result with Trust Proof"""
    url: str
    url_hash: str
    title: str
    description: str
    domain: str
    quality_score: float
    slop_score: float
    spam_score: float
    word_count: int
    tags: List[str]
    
    # Trust proof
    trust_proof: List[Dict] = field(default_factory=list)
    proof_hash: str = ""
    
    embedding: Optional[List[float]] = None
    experts: Dict[str, float] = field(default_factory=dict)
    vrf_proof: Optional[str] = None
    
    def to_content(self) -> str:
        data = asdict(self)
        data['description'] = data['description'][:200]
        data['tags'] = data['tags'][:5]
        return json.dumps(data)
    
    @classmethod
    def from_content(cls, content: str) -> 'CrawlResultEvent':
        data = json.loads(content)
        data.setdefault('trust_proof', [])
        data.setdefault('proof_hash', '')
        return cls(**data)


@dataclass
class VoteSignalEvent:
    """Kind 4244: User quality vote"""
    url_hash: str
    is_good: bool
    cluster_id: int  # LSH cluster for personalization
    pow_challenge: Optional[str] = None  # Anti-spam PoW challenge
    pow_nonce: Optional[int] = None      # Anti-spam PoW solution
    
    def to_content(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_content(cls, content: str) -> 'VoteSignalEvent':
        return cls(**json.loads(content))


@dataclass 
class GenesisConfigEvent:
    """Kind 4245: Shared configuration (PCA matrix, LSH planes)"""
    version: int
    pca_matrix: List[List[float]]
    lsh_l1_planes: List[List[float]]
    lsh_l2_planes: List[List[float]]
    vrf_epoch_seconds: int = 600  # 10 minutes
    
    def to_content(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_content(cls, content: str) -> 'GenesisConfigEvent':
        return cls(**json.loads(content))


@dataclass
class PeerAnnounceEvent:
    """Kind 4250: Announce presence to swarm"""
    version: str = "1.0"
    capabilities: List[str] = field(default_factory=lambda: ["crawl", "search", "vote"])
    index_size: int = 0  # How many pages indexed
    
    def to_content(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_content(cls, content: str) -> 'PeerAnnounceEvent':
        return cls(**json.loads(content))


@dataclass
class PeerChallengeEvent:
    """Kind 4251: Challenge a peer to prove legitimacy"""
    target_pubkey: str
    challenge: str  # Random hex string
    difficulty: int = 4  # Required leading zeros
    
    def to_content(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_content(cls, content: str) -> 'PeerChallengeEvent':
        return cls(**json.loads(content))


@dataclass
class PeerResponseEvent:
    """Kind 4252: Response to peer challenge"""
    challenge: str
    nonce: int
    index_size: int = 0
    
    def to_content(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_content(cls, content: str) -> 'PeerResponseEvent':
        return cls(**json.loads(content))


# ============================================================
# NOSTR SERVICE
# ============================================================
class NostrService:
    """
    Manages connections to Nostr relays and event pub/sub.
    
    Features:
    - Multi-relay connection pool
    - Web of trust (only accept events from trusted pubkeys)
    - Event deduplication
    - Automatic reconnection
    """
    
    def __init__(
        self,
        private_key_hex: str,
        relays: List[str],
        trusted_pubkeys: Set[str] = None,
    ):
        self.private_key = PrivateKey(bytes.fromhex(private_key_hex), raw=True)
        self.public_key = self.private_key.pubkey.serialize()[1:].hex()  # x-only
        self.relays = relays
        self.trusted_pubkeys = trusted_pubkeys or set()
        self.trusted_pubkeys.add(self.public_key)  # Always trust self
        
        self._connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        # Use OrderedDict for proper LRU eviction (sets are unordered!)
        self._seen_events: OrderedDict[str, float] = OrderedDict()
        self._max_seen_events = 10000
        self._handlers: Dict[int, List[Callable]] = defaultdict(list)
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # Rate limiting
        self._publish_times: List[float] = []
        self._max_publish_rate = 10  # per minute
    
    async def start(self):
        """Start the service and connect to relays"""
        if self._running:
            return
        
        self._running = True
        logger.info(f"🔌 Starting Nostr service (pubkey: {self.public_key[:16]}...)")
        
        # Connect to all relays
        for relay in self.relays:
            task = asyncio.create_task(self._relay_loop(relay))
            self._tasks.append(task)
        
        # Start subscription manager
        self._tasks.append(asyncio.create_task(self._subscription_manager()))
    
    async def stop(self):
        """Stop the service"""
        self._running = False
        for task in self._tasks:
            task.cancel()
        for ws in self._connections.values():
            await ws.close()
        self._connections.clear()
        logger.info("🔌 Nostr service stopped")
    
    async def _relay_loop(self, relay: str):
        """Maintain connection to a single relay with auto-reconnect"""
        backoff = 1
        
        while self._running:
            try:
                logger.info(f"📡 Connecting to {relay}")
                async with websockets.connect(relay, ping_interval=30) as ws:
                    self._connections[relay] = ws
                    backoff = 1  # Reset backoff on successful connect
                    logger.info(f"✅ Connected to {relay}")
                    
                    # Subscribe to our event kinds
                    await self._subscribe(ws)
                    
                    # Read messages
                    async for message in ws:
                        await self._handle_message(relay, message)
                        
            except ConnectionClosed:
                logger.warning(f"📡 Connection closed: {relay}")
            except Exception as e:
                logger.error(f"📡 Relay error {relay}: {e}")
            
            # Cleanup
            self._connections.pop(relay, None)
            
            if self._running:
                logger.info(f"📡 Reconnecting to {relay} in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
    
    async def _subscribe(self, ws: websockets.WebSocketClientProtocol):
        """Subscribe to relevant event kinds"""
        subscription = {
            "kinds": [EventKind.URL_DISCOVERY, EventKind.CRAWL_RESULT, 
                      EventKind.VOTE_SIGNAL, EventKind.GENESIS_CONFIG],
            "since": int(time.time()) - 3600,  # Last hour
        }
        
        # Only from trusted pubkeys if configured
        if self.trusted_pubkeys:
            subscription["authors"] = list(self.trusted_pubkeys)
        
        req = json.dumps(["REQ", "antigoogle", subscription])
        await ws.send(req)
        logger.debug(f"📬 Subscribed with filter: {subscription}")
    
    async def _subscription_manager(self):
        """Periodically refresh subscriptions"""
        while self._running:
            await asyncio.sleep(300)  # Every 5 minutes
            for relay, ws in list(self._connections.items()):
                try:
                    await self._subscribe(ws)
                except Exception as e:
                    logger.warning(f"Subscription refresh failed for {relay}: {e}")
    
    async def _handle_message(self, relay: str, raw: str):
        """Handle incoming relay message"""
        try:
            msg = json.loads(raw)
            
            if msg[0] == "EVENT":
                event = msg[2]
                await self._handle_event(event)
            elif msg[0] == "OK":
                event_id, success, message = msg[1], msg[2], msg[3] if len(msg) > 3 else ""
                if success:
                    logger.debug(f"✅ Event {event_id[:8]} accepted by {relay}")
                else:
                    logger.warning(f"❌ Event {event_id[:8]} rejected by {relay}: {message}")
            elif msg[0] == "NOTICE":
                logger.info(f"📢 Notice from {relay}: {msg[1]}")
                
        except Exception as e:
            logger.error(f"Message parse error: {e}")
    
    async def _handle_event(self, event: dict):
        """Process an incoming event"""
        event_id = event.get('id')
        
        # Deduplication using OrderedDict for LRU
        if event_id in self._seen_events:
            return
        
        # Add to seen events with timestamp
        self._seen_events[event_id] = time.time()
        
        # LRU eviction: remove oldest 20% when limit exceeded
        if len(self._seen_events) > self._max_seen_events:
            # Remove oldest entries (first items in OrderedDict)
            evict_count = self._max_seen_events // 5  # 20%
            for _ in range(evict_count):
                self._seen_events.popitem(last=False)
        
        # Verify event ID
        computed_id = compute_event_id(event)
        if computed_id != event_id:
            logger.warning(f"Invalid event ID: {event_id[:16]}")
            return
        
        # Web of Trust check
        pubkey = event.get('pubkey')
        if self.trusted_pubkeys and pubkey not in self.trusted_pubkeys:
            logger.debug(f"Ignoring event from untrusted pubkey: {pubkey[:16]}")
            return
        
        # Verify signature
        if not verify_signature(event):
            logger.warning(f"Invalid signature on event: {event_id[:16]}")
            return
        
        # Dispatch to handlers
        kind = event.get('kind')
        handlers = self._handlers.get(kind, [])
        
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Handler error for kind {kind}: {e}")
    
    def on_event(self, kind: int):
        """Decorator to register event handler"""
        def decorator(func: Callable):
            self._handlers[kind].append(func)
            return func
        return decorator
    
    def register_handler(self, kind: int, handler: Callable):
        """Register an event handler"""
        self._handlers[kind].append(handler)
    
    async def publish(self, kind: int, content: str, tags: List[List[str]] = None) -> Optional[str]:
        """
        Publish an event to all connected relays.
        
        Returns:
            Event ID if successful, None otherwise
        """
        # Rate limiting
        now = time.time()
        self._publish_times = [t for t in self._publish_times if t > now - 60]
        if len(self._publish_times) >= self._max_publish_rate:
            logger.warning("Rate limit exceeded, event not published")
            return None
        self._publish_times.append(now)
        
        # Build event
        event = {
            'pubkey': self.public_key,
            'created_at': int(now),
            'kind': kind,
            'tags': tags or [],
            'content': content,
        }
        event['id'] = compute_event_id(event)
        event['sig'] = sign_event(event, self.private_key)
        
        # Publish to all relays
        msg = json.dumps(["EVENT", event])
        success = False
        
        for relay, ws in list(self._connections.items()):
            try:
                await ws.send(msg)
                success = True
            except Exception as e:
                logger.warning(f"Publish to {relay} failed: {e}")
        
        if success:
            logger.info(f"📤 Published event {event['id'][:16]} (kind {kind})")
            return event['id']
        return None
    
    async def publish_discovery(self, event: URLDiscoveryEvent) -> Optional[str]:
        """Publish URL discovery event"""
        tags = [
            ["d", event.domain],  # Domain tag for filtering
            ["u", event.url],
        ]
        if event.source_url:
            tags.append(["source", event.source_url])
        return await self.publish(EventKind.URL_DISCOVERY, event.to_content(), tags)
    
    async def publish_result(self, event: CrawlResultEvent) -> Optional[str]:
        """Publish crawl result event"""
        tags = [
            ["d", event.domain],
            ["u", event.url],
            ["h", event.url_hash],
        ]
        for tag in event.tags:
            tags.append(["t", tag])
        return await self.publish(EventKind.CRAWL_RESULT, event.to_content(), tags)
    
    async def publish_vote(self, event: VoteSignalEvent) -> Optional[str]:
        """Publish vote signal event"""
        tags = [
            ["h", event.url_hash],
            ["c", str(event.cluster_id)],
        ]
        return await self.publish(EventKind.VOTE_SIGNAL, event.to_content(), tags)
    
    async def publish_genesis(self, event: GenesisConfigEvent) -> Optional[str]:
        """Publish genesis configuration (only leader should do this)"""
        tags = [["v", str(event.version)]]
        return await self.publish(EventKind.GENESIS_CONFIG, event.to_content(), tags)
    
    def get_stats(self) -> dict:
        """Get service statistics"""
        return {
            'public_key': self.public_key,
            'connected_relays': list(self._connections.keys()),
            'trusted_pubkeys': len(self.trusted_pubkeys),
            'seen_events': len(self._seen_events),
            'handlers': {k: len(v) for k, v in self._handlers.items()},
        }


# ============================================================
# SINGLETON
# ============================================================
_nostr_service: Optional[NostrService] = None


async def get_nostr_service() -> NostrService:
    """Get or create the Nostr service singleton"""
    global _nostr_service
    
    if _nostr_service is None:
        _nostr_service = NostrService(
            private_key_hex=settings.NOSTR_PRIVATE_KEY,
            relays=settings.NOSTR_RELAYS,
            trusted_pubkeys=set(settings.NOSTR_TRUSTED_PUBKEYS),
        )
        await _nostr_service.start()
    
    return _nostr_service


async def stop_nostr_service():
    """Stop the Nostr service"""
    global _nostr_service
    if _nostr_service:
        await _nostr_service.stop()
        _nostr_service = None
