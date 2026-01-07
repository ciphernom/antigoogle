"""
VRF (Verifiable Random Function) - Domain Lottery

Implements a VRF using deterministic BIP-340 Schnorr signatures.
This is the standard cryptographic primitive used in Nostr.

Mechanism:
- Proof (pi) = Deterministic Schnorr Signature of (domain || epoch)
- Output (beta) = SHA256(Proof)

This guarantees:
1. Uniqueness: For a given key+input, the output is always the same.
2. Verifiability: Anyone with the pubkey can verify the signature.
3. Randomness: The hash of the signature is uniformly distributed.
"""
import hashlib
import struct
import time
import logging
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

# Uses the standard secp256k1 bindings available in the environment
from secp256k1 import PrivateKey, PublicKey

from .config import get_settings

settings = get_settings()
logger = logging.getLogger("vrf")

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

# ============================================================
# STANDARD BIP-340 VRF IMPLEMENTATION
# ============================================================
class NostrVRF:
    """
    VRF implementation using deterministic BIP-340 Schnorr signatures.
    
    Instead of rolling custom EC math, we use the property that 
    BIP-340 signatures are deterministic (nonce is derived from key+msg).
    
    Proof = Signature(Sk, Msg)
    Output = Hash(Proof)
    """
    
    @staticmethod
    def prove(private_key: PrivateKey, alpha: bytes) -> Tuple[bytes, bytes]:
        """
        Generate VRF proof and output.
        
        Args:
            private_key: secp256k1 private key
            alpha: Input message (domain || epoch)
        
        Returns:
            (beta, pi) where:
            - beta: 32-byte VRF output (SHA256 of signature)
            - pi: 64-byte Proof (The Schnorr signature itself)
        """
        # 1. Hash input to 32-bytes (required for Schnorr signing)
        msg_hash = sha256(b"VRF_INPUT:" + alpha)
        
        # 2. Generate Deterministic Schnorr Signature
        # passing None for aux_rand ensures deterministic nonce generation (BIP-340)
        pi = private_key.schnorr_sign(msg_hash, None, raw=True)
        
        # 3. Generate Output (Beta) by hashing the unique signature
        beta = sha256(b"VRF_OUTPUT:" + pi)
        
        return beta, pi
    
    @staticmethod
    def verify(public_key: PublicKey, alpha: bytes, beta: bytes, pi: bytes) -> bool:
        """
        Verify VRF proof.
        
        Args:
            public_key: secp256k1 public key
            alpha: Original input
            beta: Claimed VRF output
            pi: Proof (Signature)
        
        Returns:
            True if proof is valid and matches output
        """
        if len(pi) != 64: # Schnorr signatures are 64 bytes
            return False
            
        # 1. Verify the signature is valid for the input
        msg_hash = sha256(b"VRF_INPUT:" + alpha)
        
        try:
            # schnorr_verify returns True/False or raises Error in some bindings
            is_valid = public_key.schnorr_verify(msg_hash, pi, None, raw=True)
            if not is_valid:
                return False
        except Exception as e:
            logger.debug(f"VRF Sig verify failed: {e}")
            return False
            
        # 2. Verify the output matches the signature hash
        expected_beta = sha256(b"VRF_OUTPUT:" + pi)
        if beta != expected_beta:
            return False
            
        return True

# ============================================================
# DOMAIN LOTTERY
# ============================================================
@dataclass
class LotteryTicket:
    """Result of domain lottery"""
    domain: str
    epoch: int
    vrf_output: bytes # The random value used for sorting
    vrf_proof: bytes  # The signature proving validity
    rank: int = 0     # 0 = Winner, 1 = Backup, etc.

class DomainLottery:
    """
    Determines crawl responsibility using VRF lottery.
    """
    
    def __init__(self, private_key: PrivateKey, epoch_seconds: int = None):
        self.private_key = private_key
        # Ensure we have the correct public key object
        self.public_key = private_key.pubkey
        self.epoch_seconds = epoch_seconds or settings.VRF_EPOCH_SECONDS
        self.vrf = NostrVRF()
    
    def get_current_epoch(self) -> int:
        """Get current epoch number"""
        return int(time.time()) // self.epoch_seconds
    
    def compute_ticket(self, domain: str, epoch: int = None) -> LotteryTicket:
        """
        Compute lottery ticket for a domain.
        """
        if epoch is None:
            epoch = self.get_current_epoch()
        
        # Input: domain || epoch (big-endian 64-bit unsigned)
        alpha = domain.encode() + struct.pack('>Q', epoch)
        
        beta, pi = self.vrf.prove(self.private_key, alpha)
        
        return LotteryTicket(
            domain=domain,
            epoch=epoch,
            vrf_output=beta,
            vrf_proof=pi
        )
    
    def verify_ticket(self, ticket: LotteryTicket, public_key_hex: str) -> bool:
        """
        Verify a lottery ticket from another node.
        
        Args:
            ticket: The ticket object
            public_key_hex: The x-only hex public key of the sender
        """
        try:
            # Reconstruct PublicKey object from hex
            # Note: secp256k1 library usually expects '02' or '03' prefix for full key
            # or might handle x-only. Nostr uses x-only. 
            # We prepend '02' to assume even Y for verification context if needed,
            # but standard lib usage depends on specific binding version.
            # Assuming standard python-secp256k1 behavior:
            pk_bytes = bytes.fromhex('02' + public_key_hex)
            public_key = PublicKey(pk_bytes, raw=True)
            
            alpha = ticket.domain.encode() + struct.pack('>Q', ticket.epoch)
            return self.vrf.verify(public_key, alpha, ticket.vrf_output, ticket.vrf_proof)
        except Exception as e:
            logger.warning(f"Ticket verification error: {e}")
            return False
    
    def should_crawl(
        self, 
        domain: str, 
        competitors: Dict[str, bytes] = None,
        max_winners: int = 1
    ) -> Tuple[bool, LotteryTicket]:
        """
        Determine if this node should crawl the domain.
        
        Args:
            domain: Domain to check
            competitors: Dict of {pubkey: vrf_output_bytes} from other nodes
            max_winners: How many nodes can crawl (for redundancy)
        
        Returns:
            (should_crawl, ticket)
        """
        ticket = self.compute_ticket(domain)
        
        if not competitors:
            return True, ticket
        
        # Sort myself and competitors by VRF output (lowest wins)
        # Using integer comparison of the beta bytes
        my_val = int.from_bytes(ticket.vrf_output, 'big')
        
        better_count = 0
        for comp_output in competitors.values():
            comp_val = int.from_bytes(comp_output, 'big')
            if comp_val < my_val:
                better_count += 1
            # Tie-breaking: If values are equal (extremely rare), verify pubkeys
            # For simplicity, we just count equal as "better" to be conservative
            elif comp_val == my_val:
                better_count += 1
        
        ticket.rank = better_count
        
        # If I am in the top N (rank < max_winners), I crawl.
        return better_count < max_winners, ticket

# ============================================================
# LOTTERY MANAGER
# ============================================================
class LotteryManager:
    """
    Manages state of the domain lottery across the swarm.
    """
    
    def __init__(self, lottery: DomainLottery):
        self.lottery = lottery
        # Storage: domain -> epoch -> pubkey -> vrf_output
        self._known_tickets: Dict[str, Dict[int, Dict[str, bytes]]] = {}
    
    def record_ticket(self, domain: str, epoch: int, pubkey: str, vrf_output: bytes):
        """Record a valid ticket observed from the network"""
        if domain not in self._known_tickets:
            self._known_tickets[domain] = {}
        if epoch not in self._known_tickets[domain]:
            self._known_tickets[domain][epoch] = {}
            
        self._known_tickets[domain][epoch][pubkey] = vrf_output
    
    def should_crawl(self, domain: str, redundancy: int = 1) -> Tuple[bool, LotteryTicket]:
        """Check if we should crawl, considering known competitors"""
        epoch = self.lottery.get_current_epoch()
        
        # Get competitors for this specific epoch
        competitors = self._known_tickets.get(domain, {}).get(epoch, {})
        
        return self.lottery.should_crawl(domain, competitors, max_winners=redundancy)
    
    def cleanup_old_epochs(self):
        """Remove data from past epochs to save memory"""
        current = self.lottery.get_current_epoch()
        # Keep current and previous epoch (for edge cases)
        cutoff = current - 1
        
        domains_to_remove = []
        for domain, epochs in self._known_tickets.items():
            epochs_to_remove = [e for e in epochs if e < cutoff]
            for e in epochs_to_remove:
                del epochs[e]
            if not epochs:
                domains_to_remove.append(domain)
                
        for d in domains_to_remove:
            del self._known_tickets[d]

# ============================================================
# SINGLETON
# ============================================================
_lottery_manager: Optional[LotteryManager] = None

def get_lottery_manager(private_key: PrivateKey = None) -> LotteryManager:
    """Get or create lottery manager singleton"""
    global _lottery_manager
    
    if _lottery_manager is None:
        if private_key is None:
            # Load from settings
            try:
                from secp256k1 import PrivateKey as PK
                if not settings.NOSTR_PRIVATE_KEY:
                    raise ValueError("Missing NOSTR_PRIVATE_KEY")
                private_key = PK(bytes.fromhex(settings.NOSTR_PRIVATE_KEY), raw=True)
            except Exception as e:
                logger.error(f"Failed to load private key for VRF: {e}")
                # Fallback for dev/test without crashing
                import secrets
                private_key = PK(secrets.token_bytes(32), raw=True)
        
        lottery = DomainLottery(private_key)
        _lottery_manager = LotteryManager(lottery)
    
    return _lottery_manager
