"""
Embedding Service - Sentence Transformers + PCA
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import pickle
from pathlib import Path
from typing import Union, List
import asyncio
from functools import lru_cache

from .config import get_settings

settings = get_settings()

class EmbeddingService:
    """
    Handles text embedding with dimension reduction.
    Uses PCA to reduce 384-dim embeddings to 64-dim for storage efficiency.
    """
    
    def __init__(self):
        self.model = None
        self.pca = None
        self.lsh_l1 = None
        self.lsh_l2 = None
        self._initialized = False
    
    async def initialize(self):
        """Load models asynchronously"""
        if self._initialized:
            return
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_models)
        self._initialized = True
    
    def _load_models(self):
        """Load sentence transformer and PCA"""
        print("Loading embedding model...")
        self.model = SentenceTransformer(settings.EMBED_MODEL)
        
        # Load or create PCA
        pca_path = Path("data/pca_model.pkl")
        loaded = False
        
        if pca_path.exists():
            try:
                with open(pca_path, "rb") as f:
                    self.pca = pickle.load(f)
                print("Loaded PCA from disk")
                loaded = True
            except Exception as e:
                print(f"⚠️ Failed to load PCA: {e}")
        
        if not loaded:
            self._generate_pca(pca_path)
        
        # Generate LSH hyperplanes
        self._init_lsh()
        print("✅ Embedding service ready")

    def _generate_pca(self, pca_path):
        """Generate and attempt to save PCA matrix"""
        print("Creating PCA projection...")
        np.random.seed(42)
        # Random orthogonal projection
        random_matrix = np.random.randn(settings.EMBED_DIM, settings.STORED_DIM).astype(np.float32)
        q, _ = np.linalg.qr(random_matrix)
        self.pca = q
        
        # Save for consistency, but don't crash if we can't
        try:
            pca_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pca_path, "wb") as f:
                pickle.dump(self.pca, f)
        except Exception as e:
            print(f"⚠️ Could not save PCA model (using in-memory): {e}")

    
    def _init_lsh(self):
        """Initialize LSH hyperplanes for personalization"""
        np.random.seed(settings.LSH_SEED_L1)
        self.lsh_l1 = np.random.randn(settings.L1_PLANES, settings.STORED_DIM).astype(np.float32)
        self.lsh_l1 /= np.linalg.norm(self.lsh_l1, axis=1, keepdims=True)
        
        np.random.seed(settings.LSH_SEED_L2)
        self.lsh_l2 = np.random.randn(settings.L2_PLANES, settings.STORED_DIM).astype(np.float32)
        self.lsh_l2 /= np.linalg.norm(self.lsh_l2, axis=1, keepdims=True)
    
    def set_lsh_planes(self, l1_planes: List[List[float]], l2_planes: List[List[float]]):
        """
        Dynamically update LSH planes from Swarm Genesis Config.
        This ensures all nodes use identical bucketing for personalization.
        """
        try:
            new_l1 = np.array(l1_planes, dtype=np.float32)
            new_l2 = np.array(l2_planes, dtype=np.float32)
            
            # Basic shape validation
            if new_l1.shape[1] != settings.STORED_DIM or new_l2.shape[1] != settings.STORED_DIM:
                print(f"⚠️ LSH Update rejected: Dimension mismatch. Expected {settings.STORED_DIM}")
                return

            self.lsh_l1 = new_l1
            self.lsh_l2 = new_l2
            print("✅ LSH planes updated from Swarm Genesis")
            
        except Exception as e:
            print(f"❌ Failed to update LSH planes: {e}")
    
    def encode(self, text: Union[str, List[str]], reduce: bool = True) -> np.ndarray:
        """
        Encode text to embedding vector.
        
        Args:
            text: Single string or list of strings
            reduce: If True, apply PCA reduction (64-dim), else return full (384-dim)
        
        Returns:
            Normalized embedding vector(s)
        """
        if not self._initialized:
            raise RuntimeError("EmbeddingService not initialized. Call initialize() first.")
        
        # Get full embeddings
        embeddings = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        # --- START CHANGE ---
        # Optimization: If stored dim matches model dim, skip PCA entirely
        if settings.STORED_DIM == settings.EMBED_DIM:
            return embeddings
        # --- END CHANGE ---
        if not reduce:
            return embeddings
        
        # Apply PCA reduction
        if embeddings.ndim == 1:
            reduced = embeddings @ self.pca
            return reduced / (np.linalg.norm(reduced) + 1e-8)
        else:
            reduced = embeddings @ self.pca
            norms = np.linalg.norm(reduced, axis=1, keepdims=True) + 1e-8
            return reduced / norms
    
    async def encode_async(self, text: Union[str, List[str]], reduce: bool = True) -> np.ndarray:
        """
        Async version of encode - runs in thread pool to avoid blocking event loop.
        Use this in async request handlers.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.encode, text, reduce)
    
    def compute_lsh(self, embedding: np.ndarray) -> tuple[int, int, int]:
        """
        Compute LSH hash for embedding.
        
        Returns:
            (l1_hash, l2_hash, combined_cluster_id)
        """
        if embedding.shape[-1] != settings.STORED_DIM:
            raise ValueError(f"Expected {settings.STORED_DIM}-dim embedding, got {embedding.shape[-1]}")
        
        l1 = 0
        for i, plane in enumerate(self.lsh_l1):
            if np.dot(embedding, plane) > 0:
                l1 |= (1 << i)
        
        l2 = 0
        for i, plane in enumerate(self.lsh_l2):
            if np.dot(embedding, plane) > 0:
                l2 |= (1 << i)
        
        combined = l1 * settings.NUM_L2 + l2
        return l1, l2, combined
    
    def get_lsh_planes(self) -> dict:
        """Get LSH planes for client-side computation"""
        return {
            "l1_planes": self.lsh_l1.tolist(),
            "l2_planes": self.lsh_l2.tolist(),
        }
    
    def get_projection_matrix(self) -> dict:
        """Get PCA projection matrix for client-side computation"""
        return {
            "matrix": self.pca.tolist(),
            "input_dim": settings.EMBED_DIM,
            "output_dim": settings.STORED_DIM,
        }

# Global singleton
_embedder: EmbeddingService = None

async def get_embedder() -> EmbeddingService:
    """Get or create embedder singleton"""
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingService()
        await _embedder.initialize()
    return _embedder
