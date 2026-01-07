# AntiGoogle WASM Module

Fast client-side vector operations for privacy-preserving search.

## Why WASM?

| Operation | Pure JS | WASM | Speedup |
|-----------|---------|------|---------|
| Dot product (64-dim) | ~500ns | ~50ns | 10x |
| LSH hash (12 planes) | ~6μs | ~0.5μs | 12x |
| Embedding update | ~10μs | ~1μs | 10x |
| PoW (difficulty 4) | ~100ms | ~5ms | 20x |
| Mouse entropy | ~50μs | ~5μs | 10x |

On mobile: JS math can freeze UI. WASM runs in linear memory, doesn't block.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        BROWSER                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐     ┌─────────────────────────────┐   │
│  │  antigoogle.js  │────▶│     antigoogle.wasm         │   │
│  │  (JS wrapper)   │     │     (~15KB)                 │   │
│  └────────┬────────┘     │                             │   │
│           │              │  • dot_product()            │   │
│           │              │  • compute_lsh()            │   │
│           │              │  • update_embedding()       │   │
│           │              │  • compute_engagement()     │   │
│           │              │  • solve_pow()              │   │
│           │              │  • hash_fingerprint()       │   │
│           ▼              │                             │   │
│  ┌─────────────────┐     │  Linear Memory:             │   │
│  │  localStorage   │     │  • user embedding (256B)    │   │
│  │  (persistence)  │     │  • LSH planes (3KB)         │   │
│  └─────────────────┘     └─────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Files

```
wasm/
├── Cargo.toml           # Rust project config
├── build.sh             # Build script
├── generate_planes.py   # Generate LSH hyperplanes
├── src/
│   ├── lib.rs           # Main WASM code
│   ├── l1_planes.rs     # L1 hyperplanes (generated)
│   └── l2_planes.rs     # L2 hyperplanes (generated)
├── pkg/                  # Build output (after wasm-pack build)
│   ├── antigoogle_wasm.js
│   ├── antigoogle_wasm_bg.wasm
│   └── ...
└── antigoogle_client.js  # Full JS wrapper with fallback
```

## Building

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install wasm-pack
cargo install wasm-pack

# Add WASM target
rustup target add wasm32-unknown-unknown
```

### Build

```bash
cd wasm

# Generate hyperplanes (must match server seeds!)
python3 generate_planes.py

# Build WASM
wasm-pack build --target web --release

# Output in pkg/
ls -la pkg/
```

### Output Size

```
antigoogle_wasm_bg.wasm    ~15KB (gzipped: ~6KB)
antigoogle_wasm.js         ~3KB
```

## Usage

### With WASM (recommended)

```html
<script type="module">
import init, { 
    compute_lsh, 
    update_embedding,
    compute_engagement,
    solve_pow 
} from './pkg/antigoogle_wasm.js';

async function main() {
    await init();  // Load WASM
    
    // Initialize embedding
    init_random(Date.now());
    
    // Compute LSH
    const [l1, l2, combined] = compute_lsh();
    console.log(`Cluster: L1=${l1}, L2=${l2}`);
    
    // Compute engagement from behavioral signals
    const score = compute_engagement(
        30.0,   // dwell_time (seconds)
        0.8,    // scroll_depth (0-1)
        2.5,    // mouse_entropy (bits)
        3.0,    // click_position (1-indexed)
        0.0     // refined (0 or 1)
    );
    console.log(`Engagement: ${score}`);
}

main();
</script>
```

### With Fallback (works everywhere)

```html
<script src="antigoogle_client.js" type="module"></script>
<script>
// Auto-initializes, falls back to pure JS if WASM unavailable
window.AG.init().then(client => {
    console.log('Info:', client.getInfo());
    // { wasmEnabled: true/false, embedDim: 64, lsh: {...}, ... }
});
</script>
```

### Inline (no external files)

```html
<script>
// Paste contents of antigoogle_inline.js here
// (~4KB minified, ~1.5KB gzipped)
</script>
```

## API Reference

### Embedding Management

```rust
// Initialize from existing data
fn init_embedding(data: &[f32])

// Initialize random (seeded)
fn init_random(seed: u32)

// Get current embedding
fn get_embedding() -> Vec<f32>

// Update based on interaction
fn update_embedding(page_vec: &[f32], signal: f32, learning_rate: f32)

// Batch update
fn batch_update(data: &[f32], learning_rate: f32)
```

### LSH Hashing

```rust
// Hash current user embedding
fn compute_lsh() -> Vec<u32>  // [l1, l2, combined]

// Hash arbitrary vector
fn compute_lsh_for(vec: &[f32]) -> Vec<u32>
```

### Behavioral Signals

```rust
// Compute engagement score (0-1)
fn compute_engagement(
    dwell_time: f32,      // seconds (0-300)
    scroll_depth: f32,    // 0-1
    mouse_entropy: f32,   // bits (0-3)
    click_position: f32,  // 1-20
    refined: f32          // 0 or 1
) -> f32

// Compute entropy from mouse velocities
fn compute_mouse_entropy(velocities: &[f32]) -> f32
```

### Similarity

```rust
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32
fn hamming_distance(a: &[u32], b: &[u32]) -> u32
fn quantize_binary(vec: &[f32]) -> Vec<u32>
```

### Security

```rust
// Hash browser fingerprint
fn hash_fingerprint(
    screen: &str, timezone: &str, language: &str,
    cores: u32, memory: f32, touch: u32, canvas: &str
) -> String

// Solve proof-of-work
fn solve_pow(challenge: &str, difficulty: u32) -> u32

// Verify PoW
fn verify_pow(challenge: &str, nonce: u32, difficulty: u32) -> bool
```

## LSH Seed Consistency

**Critical:** Server and WASM must use same seeds!

```python
# Server (Python)
SEED_L1 = 42
SEED_L2 = 43

np.random.seed(SEED_L1)
l1_planes = np.random.randn(6, 64)
```

```rust
// WASM (Rust) - planes generated with same seeds
static L1_HYPERPLANES: [[f32; 64]; 6] = include!("l1_planes.rs");
```

```javascript
// Client (JS) - fetches from server
const params = await fetch('/api/lsh').then(r => r.json());
// params.l1_planes, params.l2_planes
```

Verification:
```
L1[0][0:3] should equal: [0.068, -0.019, 0.089]
L2[0][0:3] should equal: [0.033, -0.116, -0.048]
```

## Performance Tips

1. **Batch updates**: Use `batch_update()` instead of multiple `update_embedding()` calls

2. **Reuse vectors**: WASM copies data across boundary. Minimize crossings.

3. **Pre-compute LSH**: Cache result, only recompute after embedding update.

4. **Typed arrays**: Always pass `Float32Array` not regular arrays.

```javascript
// Bad
update_embedding([0.1, 0.2, ...], 0.5, 0.05);

// Good
const pageVec = new Float32Array([0.1, 0.2, ...]);
update_embedding(pageVec, 0.5, 0.05);
```

## Fallback Strategy

```javascript
class AntiGoogleClient {
    async init() {
        try {
            // Try WASM first
            const wasm = await import('./pkg/antigoogle_wasm.js');
            await wasm.default();
            this.impl = wasm;
            this.fast = true;
        } catch (e) {
            // Fall back to pure JS
            this.impl = PureJS;
            this.fast = false;
        }
    }
    
    computeLSH() {
        if (this.fast) {
            return this.impl.compute_lsh();
        }
        return PureJS.computeLSH(this.embedding, L1, L2);
    }
}
```

## Memory Layout

WASM linear memory usage:

```
Offset    Size    Contents
0x0000    256B    User embedding (64 × f32)
0x0100    1536B   L1 hyperplanes (6 × 64 × f32)
0x0700    1536B   L2 hyperplanes (6 × 64 × f32)
0x0D00    ...     Stack/heap

Total: ~4KB static + dynamic allocations
```

## Security Notes

1. **No secrets in WASM**: All code is visible. LSH planes are public parameters.

2. **Fingerprint is commitment**: Used for rate limiting, not identification.

3. **PoW is client-side**: Server verifies but doesn't compute.

4. **Embedding never leaves device**: Only LSH hash sent to server.
