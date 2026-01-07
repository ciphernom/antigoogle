/* tslint:disable */
/* eslint-disable */

/**
 * Batch update from multiple interactions
 * Format: [page1_vec..., signal1, page2_vec..., signal2, ...]
 */
export function batch_update(data: Float32Array, learning_rate: number): void;

/**
 * Compute engagement score from behavioral signals
 * dwell_time: seconds on page (0-300)
 * scroll_depth: 0-1
 * mouse_entropy: 0-3 (bits)
 * click_position: 1-20
 * refined: did they search again (0 or 1)
 */
export function compute_engagement(dwell_time: number, scroll_depth: number, mouse_entropy: number, click_position: number, refined: number): number;

/**
 * Compute LSH hash for current user embedding
 * Returns: (l1_cluster, l2_cluster, combined_cluster)
 */
export function compute_lsh(): Uint32Array;

/**
 * Compute LSH for arbitrary vector (for page embeddings)
 */
export function compute_lsh_for(vec: Float32Array): Uint32Array;

/**
 * Compute entropy from mouse velocity distribution
 * velocities: array of velocity magnitudes
 */
export function compute_mouse_entropy(velocities: Float32Array): number;

/**
 * Cosine similarity between two vectors
 */
export function cosine_similarity(a: Float32Array, b: Float32Array): number;

export function get_embed_dim(): number;

/**
 * Get current embedding as Float32Array
 */
export function get_embedding(): Float32Array;

export function get_memory_usage(): number;

/**
 * Hamming distance between binary vectors (packed as u32 array)
 */
export function hamming_distance(a: Uint32Array, b: Uint32Array): number;

/**
 * Hash browser fingerprint data to pseudonymous ID
 */
export function hash_fingerprint(screen: string, timezone: string, language: string, cores: number, memory: number, touch: number, canvas: string): string;

/**
 * Initialize user embedding from JS Float32Array
 */
export function init_embedding(data: Float32Array): void;

/**
 * Initialize with random values (seeded by browser fingerprint hash)
 */
export function init_random(seed: number): void;

/**
 * Binary quantize a float vector
 */
export function quantize_binary(vec: Float32Array): Uint32Array;

/**
 * Solve proof-of-work challenge
 * Returns nonce that produces hash with `difficulty` leading zero bits
 */
export function solve_pow(challenge: string, difficulty: number): number;

/**
 * Update user embedding based on page interaction
 * page_vec: the page's embedding
 * signal: engagement score [-0.5, 0.5] centered at 0
 * learning_rate: typically 0.05
 */
export function update_embedding(page_vec: Float32Array, signal: number, learning_rate: number): void;

/**
 * Verify proof-of-work
 */
export function verify_pow(challenge: string, nonce: number, difficulty: number): boolean;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly batch_update: (a: number, b: number, c: number) => void;
  readonly compute_engagement: (a: number, b: number, c: number, d: number, e: number) => number;
  readonly compute_lsh: () => [number, number];
  readonly compute_lsh_for: (a: number, b: number) => [number, number];
  readonly compute_mouse_entropy: (a: number, b: number) => number;
  readonly cosine_similarity: (a: number, b: number, c: number, d: number) => number;
  readonly get_embed_dim: () => number;
  readonly get_embedding: () => [number, number];
  readonly get_memory_usage: () => number;
  readonly hamming_distance: (a: number, b: number, c: number, d: number) => number;
  readonly hash_fingerprint: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
  readonly init_embedding: (a: number, b: number) => void;
  readonly quantize_binary: (a: number, b: number) => [number, number];
  readonly solve_pow: (a: number, b: number, c: number) => number;
  readonly update_embedding: (a: number, b: number, c: number, d: number) => void;
  readonly verify_pow: (a: number, b: number, c: number, d: number) => number;
  readonly init_random: (a: number) => void;
  readonly __wbindgen_externrefs: WebAssembly.Table;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
