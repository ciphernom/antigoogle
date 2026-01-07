#!/usr/bin/env python3
"""
Generate LSH hyperplane data for WASM module.
Must use same seed as server to produce matching hashes.
"""

import numpy as np

EMBED_DIM = 384
L1_PLANES = 6
L2_PLANES = 6
SEED_L1 = 42
SEED_L2 = 43  # seed + 1

def generate_planes(n_planes: int, dim: int, seed: int) -> np.ndarray:
    """Generate normalized random hyperplanes."""
    np.random.seed(seed)
    planes = np.random.randn(n_planes, dim).astype(np.float32)
    # Normalize each plane
    norms = np.linalg.norm(planes, axis=1, keepdims=True)
    planes = planes / norms
    return planes

def format_rust_array(planes: np.ndarray) -> str:
    """Format numpy array as Rust array literal."""
    lines = ["["]
    for plane in planes:
        values = ", ".join(f"{v:.8}f32" for v in plane)
        lines.append(f"    [{values}],")
    lines.append("]")
    return "\n".join(lines)

def main():
    # Generate L1 planes
    l1_planes = generate_planes(L1_PLANES, EMBED_DIM, SEED_L1)
    l1_rust = format_rust_array(l1_planes)
    
    with open("src/l1_planes.rs", "w") as f:
        f.write(f"// Auto-generated LSH hyperplanes (seed={SEED_L1})\n")
        f.write(f"// {L1_PLANES} planes × {EMBED_DIM} dimensions\n")
        f.write(l1_rust)
    
    print(f"Generated src/l1_planes.rs ({L1_PLANES}x{EMBED_DIM})")
    
    # Generate L2 planes
    l2_planes = generate_planes(L2_PLANES, EMBED_DIM, SEED_L2)
    l2_rust = format_rust_array(l2_planes)
    
    with open("src/l2_planes.rs", "w") as f:
        f.write(f"// Auto-generated LSH hyperplanes (seed={SEED_L2})\n")
        f.write(f"// {L2_PLANES} planes × {EMBED_DIM} dimensions\n")
        f.write(l2_rust)
    
    print(f"Generated src/l2_planes.rs ({L2_PLANES}x{EMBED_DIM})")
    
    # Verify they match what server would generate
    print("\nVerification:")
    print(f"L1[0][0:5] = {l1_planes[0][:5]}")
    print(f"L2[0][0:5] = {l2_planes[0][:5]}")

if __name__ == "__main__":
    main()
