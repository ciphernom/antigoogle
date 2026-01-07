//! AntiGoogle WASM Module
//! Fast client-side vector operations for privacy-preserving search
//! 
//! Compiles to ~15KB WASM, runs 10-50x faster than JS for vector ops

use wasm_bindgen::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ============================================================
// CONSTANTS (must match server)
// ============================================================
const EMBED_DIM: usize = 384;
const L1_PLANES: usize = 6;
const L2_PLANES: usize = 6;
const NUM_L2: usize = 64;

// ============================================================
// LSH HYPERPLANES (embedded at compile time for speed)
// Generated with seed=42 to match server
// ============================================================
static L1_HYPERPLANES: [[f32; EMBED_DIM]; L1_PLANES] = include!("l1_planes.rs");
static L2_HYPERPLANES: [[f32; EMBED_DIM]; L2_PLANES] = include!("l2_planes.rs");

// ============================================================
// CORE MATH OPERATIONS
// ============================================================

/// Fast dot product using SIMD-friendly loop
#[inline(always)]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    
    // Unroll by 4 for better WASM performance
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    
    let chunks = a.len() / 4;
    for i in 0..chunks {
        let idx = i * 4;
        sum0 += a[idx] * b[idx];
        sum1 += a[idx + 1] * b[idx + 1];
        sum2 += a[idx + 2] * b[idx + 2];
        sum3 += a[idx + 3] * b[idx + 3];
    }
    
    // Handle remainder
    for i in (chunks * 4)..a.len() {
        sum0 += a[i] * b[i];
    }
    
    sum0 + sum1 + sum2 + sum3
}

/// Vector norm (L2)
#[inline(always)]
fn norm(v: &[f32]) -> f32 {
    dot_product(v, v).sqrt()
}

/// Normalize vector in place
fn normalize(v: &mut [f32]) {
    let n = norm(v);
    if n > 1e-8 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

// ============================================================
// USER EMBEDDING (stored in WASM linear memory)
// ============================================================

static mut USER_EMBEDDING: [f32; EMBED_DIM] = [0.0; EMBED_DIM];
static mut INITIALIZED: bool = false;

/// Initialize user embedding from JS Float32Array
#[wasm_bindgen]
pub fn init_embedding(data: &[f32]) {
    unsafe {
        if data.len() == EMBED_DIM {
            USER_EMBEDDING.copy_from_slice(data);
            INITIALIZED = true;
        }
    }
}

/// Get current embedding as Float32Array
#[wasm_bindgen]
pub fn get_embedding() -> Vec<f32> {
    unsafe { USER_EMBEDDING.to_vec() }
}

/// Initialize with random values (seeded by browser fingerprint hash)
#[wasm_bindgen]
pub fn init_random(seed: u32) {
    unsafe {
        // Simple PRNG (xorshift32)
        let mut state = seed;
        for i in 0..EMBED_DIM {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            // Map to [-0.05, 0.05]
            USER_EMBEDDING[i] = ((state as f32) / (u32::MAX as f32) - 0.5) * 0.1;
        }
        normalize(&mut USER_EMBEDDING);
        INITIALIZED = true;
    }
}

// ============================================================
// LSH HASHING
// ============================================================

/// Compute LSH hash for current user embedding
/// Returns: (l1_cluster, l2_cluster, combined_cluster)
#[wasm_bindgen]
pub fn compute_lsh() -> Vec<u32> {
    unsafe {
        if !INITIALIZED {
            return vec![0, 0, 0];
        }
        
        // Level 1 hash
        let mut l1: u32 = 0;
        for (i, plane) in L1_HYPERPLANES.iter().enumerate() {
            if dot_product(&USER_EMBEDDING, plane) > 0.0 {
                l1 |= 1 << i;
            }
        }
        
        // Level 2 hash
        let mut l2: u32 = 0;
        for (i, plane) in L2_HYPERPLANES.iter().enumerate() {
            if dot_product(&USER_EMBEDDING, plane) > 0.0 {
                l2 |= 1 << i;
            }
        }
        
        let combined = l1 * (NUM_L2 as u32) + l2;
        
        vec![l1, l2, combined]
    }
}

/// Compute LSH for arbitrary vector (for page embeddings)
#[wasm_bindgen]
pub fn compute_lsh_for(vec: &[f32]) -> Vec<u32> {
    if vec.len() != EMBED_DIM {
        return vec![0, 0, 0];
    }
    
    let mut l1: u32 = 0;
    for (i, plane) in L1_HYPERPLANES.iter().enumerate() {
        if dot_product(vec, plane) > 0.0 {
            l1 |= 1 << i;
        }
    }
    
    let mut l2: u32 = 0;
    for (i, plane) in L2_HYPERPLANES.iter().enumerate() {
        if dot_product(vec, plane) > 0.0 {
            l2 |= 1 << i;
        }
    }
    
    vec![l1, l2, l1 * (NUM_L2 as u32) + l2]
}

// ============================================================
// EMBEDDING UPDATES
// ============================================================

/// Update user embedding based on page interaction
/// page_vec: the page's embedding
/// signal: engagement score [-0.5, 0.5] centered at 0
/// learning_rate: typically 0.05
#[wasm_bindgen]
pub fn update_embedding(page_vec: &[f32], signal: f32, learning_rate: f32) {
    unsafe {
        if !INITIALIZED || page_vec.len() != EMBED_DIM {
            return;
        }
        
        let lr = learning_rate * signal;
        
        // Move toward page embedding proportional to signal
        for i in 0..EMBED_DIM {
            USER_EMBEDDING[i] += lr * (page_vec[i] - USER_EMBEDDING[i]);
        }
        
        // Re-normalize
        normalize(&mut USER_EMBEDDING);
    }
}

/// Batch update from multiple interactions
/// Format: [page1_vec..., signal1, page2_vec..., signal2, ...]
#[wasm_bindgen]
pub fn batch_update(data: &[f32], learning_rate: f32) {
    let record_size = EMBED_DIM + 1; // vec + signal
    let n_records = data.len() / record_size;
    
    for i in 0..n_records {
        let offset = i * record_size;
        let page_vec = &data[offset..offset + EMBED_DIM];
        let signal = data[offset + EMBED_DIM];
        update_embedding(page_vec, signal, learning_rate);
    }
}

// ============================================================
// BEHAVIORAL SIGNAL COMPUTATION
// ============================================================

/// Compute engagement score from behavioral signals
/// dwell_time: seconds on page (0-300)
/// scroll_depth: 0-1
/// mouse_entropy: 0-3 (bits)
/// click_position: 1-20
/// refined: did they search again (0 or 1)
#[wasm_bindgen]
pub fn compute_engagement(
    dwell_time: f32,
    scroll_depth: f32,
    mouse_entropy: f32,
    click_position: f32,
    refined: f32,
) -> f32 {
    let mut score = 0.5f32;
    
    // Dwell time: log scale, 30s is neutral
    let dt = dwell_time.min(300.0);
    let dt_score = ((dt + 1.0).ln() - 30.0f32.ln()) / 10.0f32.ln();
    score += 0.3 * dt_score;
    
    // Scroll depth: linear
    score += 0.2 * (scroll_depth - 0.5);
    
    // Mouse entropy: higher = reading
    score += 0.1 * (mouse_entropy / 3.0 - 0.3);
    
    // Click position: earlier = better
    let pos_score = 1.0 - (click_position + 1.0).ln() / 21.0f32.ln();
    score += 0.15 * pos_score;
    
    // Refinement: negative
    if refined > 0.5 {
        score -= 0.25;
    }
    
    score.max(0.0).min(1.0)
}

// ============================================================
// MOUSE ENTROPY CALCULATION
// ============================================================

/// Compute entropy from mouse velocity distribution
/// velocities: array of velocity magnitudes
#[wasm_bindgen]
pub fn compute_mouse_entropy(velocities: &[f32]) -> f32 {
    if velocities.len() < 10 {
        return 0.0;
    }
    
    // Bin velocities (8 bins)
    let mut bins = [0u32; 8];
    for &v in velocities {
        let bin = ((v / 50.0) as usize).min(7);
        bins[bin] += 1;
    }
    
    // Compute entropy
    let total = velocities.len() as f32;
    let mut entropy = 0.0f32;
    
    for &count in &bins {
        if count > 0 {
            let p = (count as f32) / total;
            entropy -= p * p.log2();
        }
    }
    
    entropy
}

// ============================================================
// SIMILARITY COMPUTATIONS
// ============================================================

/// Cosine similarity between two vectors
#[wasm_bindgen]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    
    let dot = dot_product(a, b);
    let norm_a = norm(a);
    let norm_b = norm(b);
    
    if norm_a < 1e-8 || norm_b < 1e-8 {
        return 0.0;
    }
    
    dot / (norm_a * norm_b)
}

/// Hamming distance between binary vectors (packed as u32 array)
#[wasm_bindgen]
pub fn hamming_distance(a: &[u32], b: &[u32]) -> u32 {
    if a.len() != b.len() {
        return u32::MAX;
    }
    
    let mut dist = 0u32;
    for i in 0..a.len() {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

/// Binary quantize a float vector
#[wasm_bindgen]
pub fn quantize_binary(vec: &[f32]) -> Vec<u32> {
    let n_u32 = (vec.len() + 31) / 32;
    let mut result = vec![0u32; n_u32];
    
    for (i, &v) in vec.iter().enumerate() {
        if v > 0.0 {
            result[i / 32] |= 1 << (i % 32);
        }
    }
    
    result
}

// ============================================================
// FINGERPRINT HASHING
// ============================================================

/// Hash browser fingerprint data to pseudonymous ID
#[wasm_bindgen]
pub fn hash_fingerprint(
    screen: &str,
    timezone: &str,
    language: &str,
    cores: u32,
    memory: f32,
    touch: u32,
    canvas: &str,
) -> String {
    let mut hasher = DefaultHasher::new();
    
    screen.hash(&mut hasher);
    timezone.hash(&mut hasher);
    language.hash(&mut hasher);
    cores.hash(&mut hasher);
    ((memory * 1000.0) as u32).hash(&mut hasher);
    touch.hash(&mut hasher);
    canvas.hash(&mut hasher);
    
    let hash = hasher.finish();
    format!("{:016x}", hash)
}

// ============================================================
// PROOF OF WORK
// ============================================================

/// Solve proof-of-work challenge
/// Returns nonce that produces hash with `difficulty` leading zero bits
#[wasm_bindgen]
pub fn solve_pow(challenge: &str, difficulty: u32) -> u32 {
    let target_zeros = difficulty as usize;
    let challenge_bytes = challenge.as_bytes();
    
    for nonce in 0..u32::MAX {
        // Simple hash: FNV-1a
        let mut hash = 2166136261u32;
        
        for &byte in challenge_bytes {
            hash ^= byte as u32;
            hash = hash.wrapping_mul(16777619);
        }
        
        // Add nonce bytes
        for i in 0..4 {
            let byte = ((nonce >> (i * 8)) & 0xFF) as u32;
            hash ^= byte;
            hash = hash.wrapping_mul(16777619);
        }
        
        // Check leading zeros (in hex)
        let leading_zeros = hash.leading_zeros() / 4;
        if leading_zeros >= target_zeros as u32 {
            return nonce;
        }
    }
    
    0
}

/// Verify proof-of-work
#[wasm_bindgen]
pub fn verify_pow(challenge: &str, nonce: u32, difficulty: u32) -> bool {
    let challenge_bytes = challenge.as_bytes();
    
    let mut hash = 2166136261u32;
    
    for &byte in challenge_bytes {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(16777619);
    }
    
    for i in 0..4 {
        let byte = ((nonce >> (i * 8)) & 0xFF) as u32;
        hash ^= byte;
        hash = hash.wrapping_mul(16777619);
    }
    
    let leading_zeros = hash.leading_zeros() / 4;
    leading_zeros >= difficulty
}

// ============================================================
// MEMORY INFO (for debugging)
// ============================================================

#[wasm_bindgen]
pub fn get_memory_usage() -> usize {
    std::mem::size_of::<[f32; EMBED_DIM]>() * 2 + // User embedding + temp
    std::mem::size_of::<[[f32; EMBED_DIM]; L1_PLANES]>() + // L1 planes
    std::mem::size_of::<[[f32; EMBED_DIM]; L2_PLANES]>()   // L2 planes
}

#[wasm_bindgen]
pub fn get_embed_dim() -> usize {
    EMBED_DIM
}
