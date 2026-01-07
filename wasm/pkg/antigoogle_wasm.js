let wasm;

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    }
}

let WASM_VECTOR_LEN = 0;

/**
 * Batch update from multiple interactions
 * Format: [page1_vec..., signal1, page2_vec..., signal2, ...]
 * @param {Float32Array} data
 * @param {number} learning_rate
 */
export function batch_update(data, learning_rate) {
    const ptr0 = passArrayF32ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    wasm.batch_update(ptr0, len0, learning_rate);
}

/**
 * Compute engagement score from behavioral signals
 * dwell_time: seconds on page (0-300)
 * scroll_depth: 0-1
 * mouse_entropy: 0-3 (bits)
 * click_position: 1-20
 * refined: did they search again (0 or 1)
 * @param {number} dwell_time
 * @param {number} scroll_depth
 * @param {number} mouse_entropy
 * @param {number} click_position
 * @param {number} refined
 * @returns {number}
 */
export function compute_engagement(dwell_time, scroll_depth, mouse_entropy, click_position, refined) {
    const ret = wasm.compute_engagement(dwell_time, scroll_depth, mouse_entropy, click_position, refined);
    return ret;
}

/**
 * Compute LSH hash for current user embedding
 * Returns: (l1_cluster, l2_cluster, combined_cluster)
 * @returns {Uint32Array}
 */
export function compute_lsh() {
    const ret = wasm.compute_lsh();
    var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
}

/**
 * Compute LSH for arbitrary vector (for page embeddings)
 * @param {Float32Array} vec
 * @returns {Uint32Array}
 */
export function compute_lsh_for(vec) {
    const ptr0 = passArrayF32ToWasm0(vec, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.compute_lsh_for(ptr0, len0);
    var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v2;
}

/**
 * Compute entropy from mouse velocity distribution
 * velocities: array of velocity magnitudes
 * @param {Float32Array} velocities
 * @returns {number}
 */
export function compute_mouse_entropy(velocities) {
    const ptr0 = passArrayF32ToWasm0(velocities, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.compute_mouse_entropy(ptr0, len0);
    return ret;
}

/**
 * Cosine similarity between two vectors
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @returns {number}
 */
export function cosine_similarity(a, b) {
    const ptr0 = passArrayF32ToWasm0(a, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(b, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.cosine_similarity(ptr0, len0, ptr1, len1);
    return ret;
}

/**
 * @returns {number}
 */
export function get_embed_dim() {
    const ret = wasm.get_embed_dim();
    return ret >>> 0;
}

/**
 * Get current embedding as Float32Array
 * @returns {Float32Array}
 */
export function get_embedding() {
    const ret = wasm.get_embedding();
    var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
}

/**
 * @returns {number}
 */
export function get_memory_usage() {
    const ret = wasm.get_memory_usage();
    return ret >>> 0;
}

/**
 * Hamming distance between binary vectors (packed as u32 array)
 * @param {Uint32Array} a
 * @param {Uint32Array} b
 * @returns {number}
 */
export function hamming_distance(a, b) {
    const ptr0 = passArray32ToWasm0(a, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray32ToWasm0(b, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.hamming_distance(ptr0, len0, ptr1, len1);
    return ret >>> 0;
}

/**
 * Hash browser fingerprint data to pseudonymous ID
 * @param {string} screen
 * @param {string} timezone
 * @param {string} language
 * @param {number} cores
 * @param {number} memory
 * @param {number} touch
 * @param {string} canvas
 * @returns {string}
 */
export function hash_fingerprint(screen, timezone, language, cores, memory, touch, canvas) {
    let deferred5_0;
    let deferred5_1;
    try {
        const ptr0 = passStringToWasm0(screen, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(timezone, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(language, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passStringToWasm0(canvas, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len3 = WASM_VECTOR_LEN;
        const ret = wasm.hash_fingerprint(ptr0, len0, ptr1, len1, ptr2, len2, cores, memory, touch, ptr3, len3);
        deferred5_0 = ret[0];
        deferred5_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred5_0, deferred5_1, 1);
    }
}

/**
 * Initialize user embedding from JS Float32Array
 * @param {Float32Array} data
 */
export function init_embedding(data) {
    const ptr0 = passArrayF32ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    wasm.init_embedding(ptr0, len0);
}

/**
 * Initialize with random values (seeded by browser fingerprint hash)
 * @param {number} seed
 */
export function init_random(seed) {
    wasm.init_random(seed);
}

/**
 * Binary quantize a float vector
 * @param {Float32Array} vec
 * @returns {Uint32Array}
 */
export function quantize_binary(vec) {
    const ptr0 = passArrayF32ToWasm0(vec, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.quantize_binary(ptr0, len0);
    var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v2;
}

/**
 * Solve proof-of-work challenge
 * Returns nonce that produces hash with `difficulty` leading zero bits
 * @param {string} challenge
 * @param {number} difficulty
 * @returns {number}
 */
export function solve_pow(challenge, difficulty) {
    const ptr0 = passStringToWasm0(challenge, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.solve_pow(ptr0, len0, difficulty);
    return ret >>> 0;
}

/**
 * Update user embedding based on page interaction
 * page_vec: the page's embedding
 * signal: engagement score [-0.5, 0.5] centered at 0
 * learning_rate: typically 0.05
 * @param {Float32Array} page_vec
 * @param {number} signal
 * @param {number} learning_rate
 */
export function update_embedding(page_vec, signal, learning_rate) {
    const ptr0 = passArrayF32ToWasm0(page_vec, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    wasm.update_embedding(ptr0, len0, signal, learning_rate);
}

/**
 * Verify proof-of-work
 * @param {string} challenge
 * @param {number} nonce
 * @param {number} difficulty
 * @returns {boolean}
 */
export function verify_pow(challenge, nonce, difficulty) {
    const ptr0 = passStringToWasm0(challenge, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.verify_pow(ptr0, len0, nonce, difficulty);
    return ret !== 0;
}

const EXPECTED_RESPONSE_TYPES = new Set(['basic', 'cors', 'default']);

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && EXPECTED_RESPONSE_TYPES.has(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else {
                    throw e;
                }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }
}

function __wbg_get_imports() {
    const imports = {};
    imports.wbg = {};
    imports.wbg.__wbindgen_init_externref_table = function() {
        const table = wasm.__wbindgen_externrefs;
        const offset = table.grow(4);
        table.set(0, undefined);
        table.set(offset + 0, undefined);
        table.set(offset + 1, null);
        table.set(offset + 2, true);
        table.set(offset + 3, false);
    };

    return imports;
}

function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    __wbg_init.__wbindgen_wasm_module = module;
    cachedFloat32ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;


    wasm.__wbindgen_start();
    return wasm;
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (typeof module !== 'undefined') {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (typeof module_or_path !== 'undefined') {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (typeof module_or_path === 'undefined') {
        module_or_path = new URL('antigoogle_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
