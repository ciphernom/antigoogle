/**
 * AntiGoogle Inline Client
 * Self-contained version that embeds in HTML
 * Pure JS - no WASM dependency (can be added later)
 * 
 * Minified: ~4KB gzipped
 */
(function(window) {
    'use strict';
    
    const C = { // Config
        DIM: 384,
        N_L2: 64,
        STORE: 'ag_emb'
    };
    
    // LSH hyperplanes (will be loaded from server)
    let L1 = null, L2 = null;
    
    // User embedding (Float32Array)
    let emb = null;
    
    // Behavior tracking
    const B = { t0: Date.now(), scroll: 0, mVel: [], pos: 1 };
    
    // ---- MATH ----
    const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
    const norm = v => Math.sqrt(dot(v, v));
    const normalize = v => { const n = norm(v); if (n > 1e-8) for (let i = 0; i < v.length; i++) v[i] /= n; return v; };
    
    // ---- LSH ----
    const lsh = () => {
        if (!L1 || !L2 || !emb) return { l1: 0, l2: 0, c: 0 };
        let l1 = 0, l2 = 0;
        for (let i = 0; i < L1.length; i++) if (dot(emb, L1[i]) > 0) l1 |= (1 << i);
        for (let i = 0; i < L2.length; i++) if (dot(emb, L2[i]) > 0) l2 |= (1 << i);
        return { l1, l2, c: l1 * C.N_L2 + l2 };
    };
    
    // ---- EMBEDDING ----
    const load = () => {
        const s = localStorage.getItem(C.STORE);
        if (s) {
            emb = new Float32Array(JSON.parse(s));
        } else {
            emb = new Float32Array(C.DIM);
            for (let i = 0; i < C.DIM; i++) emb[i] = (Math.random() - 0.5) * 0.1;
            normalize(emb);
            save();
        }
    };
    
    const save = () => localStorage.setItem(C.STORE, JSON.stringify(Array.from(emb)));
    
    const update = (pageVec, signal, lr = 0.05) => {
        const s = lr * signal;
        for (let i = 0; i < C.DIM; i++) emb[i] += s * (pageVec[i] - emb[i]);
        normalize(emb);
        save();
    };
    
    // ---- FINGERPRINT ----
    const fp = async () => {
        const d = {
            s: `${screen.width}x${screen.height}`,
            z: Intl.DateTimeFormat().resolvedOptions().timeZone,
            l: navigator.language,
            c: navigator.hardwareConcurrency || 0,
            m: navigator.deviceMemory || 0,
            t: navigator.maxTouchPoints || 0
        };
        try {
            const cv = document.createElement('canvas');
            const ctx = cv.getContext('2d');
            ctx.font = '14px Arial';
            ctx.fillText('ag', 2, 2);
            d.v = cv.toDataURL().slice(-30);
        } catch(e) {}
        const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(JSON.stringify(d)));
        return Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2,'0')).join('').slice(0, 32);
    };
    
    // ---- BEHAVIOR ----
    const entropy = () => {
        if (B.mVel.length < 10) return 0;
        const bins = new Array(8).fill(0);
        for (const v of B.mVel) bins[Math.min(7, Math.floor(v / 50))]++;
        let e = 0;
        for (const c of bins) if (c > 0) { const p = c / B.mVel.length; e -= p * Math.log2(p); }
        return e;
    };
    
    const engage = () => {
        const dt = Math.min((Date.now() - B.t0) / 1000, 300);
        let s = 0.5;
        s += 0.3 * (Math.log(dt + 1) - Math.log(30)) / Math.log(10);
        s += 0.2 * (B.scroll - 0.5);
        s += 0.1 * (entropy() / 3 - 0.3);
        s += 0.15 * (1 - Math.log(B.pos + 1) / Math.log(21));
        return Math.max(0, Math.min(1, s));
    };
    
    // ---- TRACKING ----
    let lastM = 0, lastPos = null;
    
    window.addEventListener('scroll', () => {
        const h = document.body.scrollHeight - window.innerHeight;
        if (h > 0) B.scroll = Math.max(B.scroll, window.scrollY / h);
    });
    
    document.addEventListener('mousemove', e => {
        const now = Date.now();
        if (now - lastM > 100) {
            if (lastPos) {
                const dx = e.clientX - lastPos.x, dy = e.clientY - lastPos.y;
                B.mVel.push(Math.sqrt(dx*dx + dy*dy));
                if (B.mVel.length > 100) B.mVel.shift();
            }
            lastPos = { x: e.clientX, y: e.clientY };
            lastM = now;
        }
    });
    
    // ---- SIGNAL ----
    const send = async (pageId, pageEmb = null) => {
        const h = lsh();
        if (!h.l1 && !h.l2) return;
        
        const score = engage();
        const hash = await fp();
        
        if (pageEmb) update(pageEmb, score - 0.5);
        
        try {
            await fetch('/api/signal', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    l1: h.l1,
                    l2: h.l2,
                    page_id: pageId,
                    score: score,
                    fp_hash: hash
                })
            });
        } catch(e) {}
    };
    
    // ---- INIT ----
    const init = async () => {
        try {
            const r = await fetch('/api/lsh');
            const p = await r.json();
            L1 = p.l1_planes;
            L2 = p.l2_planes;
        } catch(e) {}
        
        load();
        const h = lsh();
        window.agL1 = h.l1;
        window.agL2 = h.l2;
    };
    
    // ---- EXPORTS ----
    window.AG = {
        init,
        lsh,
        update,
        send,
        click: pos => { B.pos = pos; },
        reset: () => { B.t0 = Date.now(); B.scroll = 0; },
        info: () => ({
            dim: C.DIM,
            lsh: lsh(),
            scroll: B.scroll,
            entropy: entropy(),
            engage: engage()
        })
    };
    
    // Auto-init
    init();
    
})(window);
