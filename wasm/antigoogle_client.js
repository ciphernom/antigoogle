/**
 * AntiGoogle Client Library
 * 
 * Uses WASM for fast vector operations with pure-JS fallback.
 * WASM is ~10-50x faster for the math-heavy operations.
 */

// ============================================================
// CONFIGURATION
// ============================================================
const CONFIG = {
    EMBED_DIM: 384,
    L1_PLANES: 6,
    L2_PLANES: 6,
    NUM_L2: 64,
    STORAGE_KEY: 'ag_emb',
    HISTORY_KEY: 'ag_hist',
    WASM_PATH: '/wasm/antigoogle_wasm.js',
};

// ============================================================
// WASM MODULE STATE
// ============================================================
let wasmModule = null;
let wasmReady = false;
let wasmInitPromise = null;  // Guard against double init
let lshParams = null;

// ============================================================
// PURE-JS FALLBACK IMPLEMENTATIONS
// ============================================================

const PureJS = {
    // Dot product
    dot(a, b) {
        let sum = 0;
        for (let i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    },
    
    // Vector norm
    norm(v) {
        return Math.sqrt(this.dot(v, v));
    },
    
    // Normalize in place
    normalize(v) {
        const n = this.norm(v);
        if (n > 1e-8) {
            for (let i = 0; i < v.length; i++) {
                v[i] /= n;
            }
        }
        return v;
    },
    
    // Compute LSH hash
    computeLSH(embedding, l1Planes, l2Planes) {
        let l1 = 0;
        for (let i = 0; i < l1Planes.length; i++) {
            if (this.dot(embedding, l1Planes[i]) > 0) {
                l1 |= (1 << i);
            }
        }
        
        let l2 = 0;
        for (let i = 0; i < l2Planes.length; i++) {
            if (this.dot(embedding, l2Planes[i]) > 0) {
                l2 |= (1 << i);
            }
        }
        
        return {
            l1,
            l2,
            combined: l1 * CONFIG.NUM_L2 + l2
        };
    },
    
    // Update embedding
    updateEmbedding(embedding, pageVec, signal, lr = 0.05) {
        const scaledLr = lr * signal;
        for (let i = 0; i < embedding.length; i++) {
            embedding[i] += scaledLr * (pageVec[i] - embedding[i]);
        }
        return this.normalize(embedding);
    },
    
    // Compute engagement score
    computeEngagement(dwellTime, scrollDepth, mouseEntropy, clickPosition, refined) {
        let score = 0.5;
        
        // Dwell time (log scale)
        const dt = Math.min(dwellTime, 300);
        const dtScore = (Math.log(dt + 1) - Math.log(30)) / Math.log(10);
        score += 0.3 * dtScore;
        
        // Scroll depth
        score += 0.2 * (scrollDepth - 0.5);
        
        // Mouse entropy
        score += 0.1 * (mouseEntropy / 3 - 0.3);
        
        // Click position
        const posScore = 1.0 - Math.log(clickPosition + 1) / Math.log(21);
        score += 0.15 * posScore;
        
        // Refinement
        if (refined) {
            score -= 0.25;
        }
        
        return Math.max(0, Math.min(1, score));
    },
    
    // Mouse entropy calculation
    computeMouseEntropy(velocities) {
        if (velocities.length < 10) return 0;
        
        const bins = new Array(8).fill(0);
        for (const v of velocities) {
            const bin = Math.min(7, Math.floor(v / 50));
            bins[bin]++;
        }
        
        let entropy = 0;
        const total = velocities.length;
        for (const count of bins) {
            if (count > 0) {
                const p = count / total;
                entropy -= p * Math.log2(p);
            }
        }
        
        return entropy;
    },
    
    // Cosine similarity
    cosineSimilarity(a, b) {
        const dot = this.dot(a, b);
        const normA = this.norm(a);
        const normB = this.norm(b);
        if (normA < 1e-8 || normB < 1e-8) return 0;
        return dot / (normA * normB);
    },
    
    // Simple fingerprint hash (FNV-1a, no crypto.subtle needed)
    async hashFingerprint(fp) {
        const data = JSON.stringify(fp, Object.keys(fp).sort());
        // FNV-1a hash
        let h = 2166136261;
        for (let i = 0; i < data.length; i++) {
            h ^= data.charCodeAt(i);
            h = Math.imul(h, 16777619) >>> 0;
        }
        // Return as hex string
        return h.toString(16).padStart(8, '0') + 
               (h ^ (h >>> 16)).toString(16).padStart(8, '0') +
               (h ^ (h >>> 8)).toString(16).padStart(8, '0') +
               (h ^ (h >>> 4)).toString(16).padStart(8, '0');
    },
    
    // Solve PoW (slower than WASM but works)
    async solvePoW(challenge, difficulty) {
        function countLeadingZeroHex(h) {
            if (h === 0) return 8;
            let z = 0;
            while (z < 8 && ((h >>> (28 - z * 4)) & 0xF) === 0) z++;
            return z;
        }
        
        for (let nonce = 0; nonce < 0xFFFFFFFF; nonce++) {
            // FNV-1a hash (simple, fast)
            let hash = 2166136261;
            for (let i = 0; i < challenge.length; i++) {
                hash ^= challenge.charCodeAt(i);
                hash = Math.imul(hash, 16777619) >>> 0;
            }
            for (let i = 0; i < 4; i++) {
                hash ^= (nonce >> (i * 8)) & 0xFF;
                hash = Math.imul(hash, 16777619) >>> 0;
            }
            
            // Check leading zeros
            if (countLeadingZeroHex(hash) >= difficulty) {
                return nonce;
            }
            
            // Yield to browser every 50K iterations
            if (nonce % 50000 === 0 && nonce > 0) {
                await new Promise(r => setTimeout(r, 0));
            }
        }
        return 0;
    }
};

// ============================================================
// WASM-ACCELERATED INTERFACE
// ============================================================

class AntiGoogleClient {
    constructor() {
        this.embedding = null;
        this.lshResult = null;
        this.behavior = {
            start: Date.now(),
            maxScroll: 0,
            mousePositions: [],
            mouseVelocities: [],
        };
        this.pageId = null;
        this.clickPosition = 1;
    }
    
    // Initialize (load WASM, fetch LSH params, load/create embedding)
async init() {
        console.group("🚀 AntiGoogle WASM Boot Sequence");
        
        try {
            // Check if WASM already initialized globally
            if (wasmReady && wasmModule) {
                console.log("[SKIP] WASM already initialized, reusing");
                this.wasm = wasmModule;
            } else if (wasmInitPromise) {
                console.log("[WAIT] WASM init in progress, waiting...");
                await wasmInitPromise;
                this.wasm = wasmModule;
            } else {
                // First init - set up the promise
                wasmInitPromise = (async () => {
                    // STEP 1: LOAD JS GLUE
                    const path = CONFIG.WASM_PATH;
                    console.log(`[1/4] Attempting to import JS glue from: ${path}`);
                    const wasm = await import(path);
                    console.log("[2/4] JS Glue imported successfully. Object:", wasm);

                    // STEP 2: LOAD WASM BINARY
                    console.log("[3/4] Initializing WASM binary (fetching .wasm file)...");
                    await wasm.default(); 
                    
                    // STEP 3: SUCCESS
                    console.log("[4/4] WASM Binary compiled and memory allocated!");
                    wasmModule = wasm;
                    wasmReady = true;
                    return wasm;
                })();
                
                this.wasm = await wasmInitPromise;
            }
            
            console.log("✅ SYSTEM READY: Running in WASM Mode");

        } catch (e) {
            console.error("❌ CRITICAL WASM FAILURE");
            console.error("Error Name:", e.name);
            console.error("Error Message:", e.message);
            

            console.log("⚠️ Falling back to Pure JS implementation");
            this.wasm = null;
            wasmInitPromise = null;  // Allow retry
        }
        
        console.groupEnd();
        
        // Fetch LSH parameters from server
        try {
            const resp = await fetch('/api/lsh');
            lshParams = await resp.json();
        } catch (e) {
            console.warn('Could not fetch LSH params');
        }
        
        // Load or create embedding
        this.loadEmbedding();
        
        // Compute initial LSH
        this.computeLSH();
        
        // Setup behavior tracking
        this.setupBehaviorTracking();
        
        return this;
    }
    
    // Load embedding from localStorage or create new
    loadEmbedding() {
        const stored = localStorage.getItem(CONFIG.STORAGE_KEY);
        
        if (stored) {
            this.embedding = new Float32Array(JSON.parse(stored));
        } else {
            // Initialize random
            this.embedding = new Float32Array(CONFIG.EMBED_DIM);
            for (let i = 0; i < CONFIG.EMBED_DIM; i++) {
                this.embedding[i] = (Math.random() - 0.5) * 0.1;
            }
            PureJS.normalize(this.embedding);
            this.saveEmbedding();
        }
        
        // If WASM available, sync to WASM memory
        if (this.wasm) {
            this.wasm.init_embedding(this.embedding);
        }
    }
    
    // Save embedding to localStorage
    saveEmbedding() {
        localStorage.setItem(CONFIG.STORAGE_KEY, JSON.stringify(Array.from(this.embedding)));
    }
    
    // Compute LSH hash
    computeLSH() {
        if (!lshParams) {
            this.lshResult = { l1: 0, l2: 0, combined: 0 };
            return this.lshResult;
        }
        
        if (this.wasm) {
            // Use WASM
            const result = this.wasm.compute_lsh();
            this.lshResult = {
                l1: result[0],
                l2: result[1],
                combined: result[2]
            };
        } else {
            // Use pure JS
            this.lshResult = PureJS.computeLSH(
                this.embedding,
                lshParams.l1_planes,
                lshParams.l2_planes
            );
        }
        
        // Expose globally for form submission
        window.agL1 = this.lshResult.l1;
        window.agL2 = this.lshResult.l2;
        
        return this.lshResult;
    }
    
    // Update embedding from page interaction
    updateEmbedding(pageEmbedding, signal) {
        if (this.wasm) {
            this.wasm.update_embedding(new Float32Array(pageEmbedding), signal, 0.05);
            this.embedding = new Float32Array(this.wasm.get_embedding());
        } else {
            PureJS.updateEmbedding(this.embedding, pageEmbedding, signal, 0.05);
        }
        
        this.saveEmbedding();
        this.computeLSH();
    }
    
    // Setup behavior tracking
    setupBehaviorTracking() {
        // Scroll tracking
        window.addEventListener('scroll', () => {
            const scrollable = document.body.scrollHeight - window.innerHeight;
            if (scrollable > 0) {
                const depth = window.scrollY / scrollable;
                this.behavior.maxScroll = Math.max(this.behavior.maxScroll, depth);
            }
        });
        
        // Mouse tracking (sampled)
        let lastMouseTime = 0;
        document.addEventListener('mousemove', (e) => {
            const now = Date.now();
            if (now - lastMouseTime > 100) {
                const pos = { x: e.clientX, y: e.clientY, t: now };
                
                // Compute velocity if we have previous position
                if (this.behavior.mousePositions.length > 0) {
                    const prev = this.behavior.mousePositions[this.behavior.mousePositions.length - 1];
                    const dx = pos.x - prev.x;
                    const dy = pos.y - prev.y;
                    const dt = pos.t - prev.t;
                    if (dt > 0) {
                        this.behavior.mouseVelocities.push(Math.sqrt(dx*dx + dy*dy) / dt * 100);
                        if (this.behavior.mouseVelocities.length > 100) {
                            this.behavior.mouseVelocities.shift();
                        }
                    }
                }
                
                this.behavior.mousePositions.push(pos);
                if (this.behavior.mousePositions.length > 50) {
                    this.behavior.mousePositions.shift();
                }
                
                lastMouseTime = now;
            }
        });
    }
    
    // Compute mouse entropy
    getMouseEntropy() {
        if (this.wasm) {
            return this.wasm.compute_mouse_entropy(new Float32Array(this.behavior.mouseVelocities));
        }
        return PureJS.computeMouseEntropy(this.behavior.mouseVelocities);
    }
    
    // Compute engagement score
    computeEngagement() {
        const dwellTime = (Date.now() - this.behavior.start) / 1000;
        const scrollDepth = this.behavior.maxScroll;
        const mouseEntropy = this.getMouseEntropy();
        const clickPosition = this.clickPosition;
        const refined = false; // Would track search refinements
        
        if (this.wasm) {
            return this.wasm.compute_engagement(
                dwellTime, scrollDepth, mouseEntropy, clickPosition, refined ? 1 : 0
            );
        }
        return PureJS.computeEngagement(dwellTime, scrollDepth, mouseEntropy, clickPosition, refined);
    }
    
    // Get browser fingerprint
    async getFingerprint() {
        const fp = {
            screen: `${screen.width}x${screen.height}`,
            tz: Intl.DateTimeFormat().resolvedOptions().timeZone,
            lang: navigator.language,
            cores: navigator.hardwareConcurrency || 0,
            memory: navigator.deviceMemory || 0,
            touch: navigator.maxTouchPoints || 0,
        };
        
        // Canvas fingerprint
        try {
            const c = document.createElement('canvas');
            const ctx = c.getContext('2d');
            ctx.textBaseline = 'top';
            ctx.font = '14px Arial';
            ctx.fillText('antigoogle', 2, 2);
            fp.canvas = c.toDataURL().slice(-50);
        } catch (e) {
            fp.canvas = '';
        }
        
        // Hash it
        if (this.wasm) {
            return this.wasm.hash_fingerprint(
                fp.screen, fp.tz, fp.lang, fp.cores, fp.memory, fp.touch, fp.canvas
            );
        }
        return PureJS.hashFingerprint(fp);
    }
    
    // Track result click
    trackClick(position) {
        this.clickPosition = position;
    }
    
    // Click handler - call this when user clicks a search result
    // Sends signal to server with engagement data
    click(position, pageId) {
        this.clickPosition = position;
        
        // Send signal async (don't block navigation)
        if (pageId && this.lshResult) {
            const score = this.computeEngagement();
            const data = JSON.stringify({
                l1: this.lshResult.l1,
                l2: this.lshResult.l2,
                page_id: pageId,
                score: score
            });
            
            // Use sendBeacon for reliability during navigation
            if (navigator.sendBeacon) {
                const blob = new Blob([data], { type: 'application/json' });
                navigator.sendBeacon('/api/signal', blob);
            } else {
                // Fallback to fetch (may not complete before navigation)
                fetch('/api/signal', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: data
                }).catch(() => {});
            }
        }
    }
    
    // Set page ID (called when viewing a result)
    setPageId(pageId) {
        this.pageId = pageId;
        this.behavior.start = Date.now();
        this.behavior.maxScroll = 0;
    }
    
    // Send signal to server
    async sendSignal(pageId, pageEmbedding = null) {
        if (!this.lshResult) return;
        
        const score = this.computeEngagement();
        const fpHash = await this.getFingerprint();
        
        // Update local embedding if page embedding provided
        if (pageEmbedding) {
            this.updateEmbedding(pageEmbedding, score - 0.5);
        }
        
        try {
            await fetch('/api/signal', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    l1: this.lshResult.l1,
                    l2: this.lshResult.l2,
                    page_id: pageId,
                    score: score,
                    fp_hash: fpHash
                })
            });
        } catch (e) {
            // Silent fail
        }
    }
    
    // Solve PoW
    async solvePoW(challenge, difficulty) {
        if (this.wasm) {
            return this.wasm.solve_pow(challenge, difficulty);
        }
        return PureJS.solvePoW(challenge, difficulty);
    }
    
    // Get performance info
    getInfo() {
        return {
            wasmEnabled: !!this.wasm,
            embedDim: CONFIG.EMBED_DIM,
            lsh: this.lshResult,
            behaviorTime: Date.now() - this.behavior.start,
            maxScroll: this.behavior.maxScroll,
            mouseEntropy: this.getMouseEntropy(),
        };
    }
}

// ============================================================
// GLOBAL INSTANCE & EXPORTS
// ============================================================

let client = null;
let initPromise = null;

async function init() {
    // Singleton guard - prevent double initialization
    if (client) {
        console.log('AG: Already initialized, returning existing client');
        return client;
    }
    if (initPromise) {
        console.log('AG: Init in progress, waiting...');
        return initPromise;
    }
    
    initPromise = (async () => {
        client = new AntiGoogleClient();
        await client.init();
        window.AG = client;
        return client;
    })();
    
    return initPromise;
}

// Auto-init on load (guarded)
if (typeof window !== 'undefined' && !window.AG && !window._agInitStarted) {
    window._agInitStarted = true;
    init().catch(console.error);
}

// Export for modules
export { AntiGoogleClient, init, PureJS, CONFIG };
export default init;
