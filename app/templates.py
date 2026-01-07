"""
HTML Templates - CSS and JavaScript
"""

CSS = """<style>
:root {
    /* Default (Dark) */
    --bg: #111111;
    --fg: #d0d0d0;
    --dim: #666666;
    --border: #333333;
    --link: #ffffff;
}

[data-theme="light"] {
    /* Light Mode Overrides */
    --bg: #ffffff;
    --fg: #111111;
    --dim: #888888;
    --border: #cccccc;
    --link: #000000;
}

* { box-sizing: border-box; }

body {
    background: var(--bg);
    color: var(--fg);
    font-family: 'Courier New', monospace;
    font-size: 14px;
    line-height: 1.4;
    margin: 0;
    padding: 20px;
}

.container { max-width: 800px; margin: 0 auto; }

/* Header */
.header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 10px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
}

.logo {
    font-weight: bold;
    color: var(--fg);
    text-decoration: none;
    border: 1px solid var(--fg);
    padding: 2px 8px;
}

.nav { display: flex; gap: 15px; }
.nav a { color: var(--dim); text-decoration: none; }
.nav a:hover { color: var(--fg); text-decoration: underline; }

/* Search Input */
input[type=text], input[type=number] {
    width: 100%;
    background: var(--bg);
    color: var(--fg);
    border: 1px solid var(--border);
    padding: 10px;
    font-family: inherit;
    font-size: 16px;
    outline: none;
    border-radius: 0;
    margin-bottom: 20px;
}

input:focus { border-color: var(--fg); }

/* Results */
.stats { color: var(--dim); font-size: 12px; margin-bottom: 20px; }

.correction {
    color: var(--dim);
    margin-bottom: 20px;
    font-style: italic;
}

.result {
    margin-bottom: 25px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
}

.result-title {
    display: block;
    font-size: 16px;
    font-weight: bold;
    color: var(--link);
    text-decoration: none;
    margin-bottom: 5px;
}

.result-title:hover { text-decoration: underline; }

.result-url {
    font-size: 12px;
    color: var(--dim);
    display: block;
    margin-bottom: 5px;
}

.result-desc { color: var(--fg); }

.result-meta { margin-top: 5px; font-size: 12px; color: var(--dim); }

.tag {
    border: 1px solid var(--border);
    padding: 0 4px;
    font-size: 10px;
    margin-left: 8px;
    color: var(--dim);
}

/* Quality Indicators */
.quality-high { color: var(--fg); font-weight: bold; }
.quality-med { color: var(--dim); }
.quality-low { color: var(--border); }

/* Add URL Page */
.box { border: 1px solid var(--border); padding: 20px; margin-top: 20px; }
label { display: block; margin-bottom: 5px; color: var(--dim); }

button {
    background: var(--bg);
    color: var(--fg);
    border: 1px solid var(--border);
    padding: 8px 16px;
    font-family: inherit;
    cursor: pointer;
    border-radius: 0;
}

button:hover { border-color: var(--fg); background: var(--border); }
button:disabled { border-color: var(--border); color: var(--dim); cursor: wait; }

.pow-status {
    margin-top: 10px;
    font-size: 12px;
    color: var(--dim);
}

/* Subtle Theme Toggle */
#theme-toggle {
    position: fixed;
    top: 15px;
    right: 15px;
    color: var(--dim);
    text-decoration: none;
    font-family: monospace;
    font-size: 12px;
    opacity: 0.5;
    z-index: 9999;
}
#theme-toggle:hover { opacity: 1; color: var(--fg); }
</style>
"""

JS = """<script type="module">
import('/wasm/antigoogle_client.js').catch(e => {
    console.warn('WASM client failed, using inline fallback:', e);
    const C={DIM:64,STORE:'ag_emb'};let L1=null,L2=null,emb=null;
    const dot=(a,b)=>{let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s};
    const norm=v=>Math.sqrt(dot(v,v));
    const normalize=v=>{const n=norm(v);if(n>1e-8)for(let i=0;i<v.length;i++)v[i]/=n;return v};
    const lsh=()=>{if(!L1||!L2||!emb)return{l1:0,l2:0};let l1=0,l2=0;
    for(let i=0;i<L1.length;i++)if(dot(emb,L1[i])>0)l1|=(1<<i);
    for(let i=0;i<L2.length;i++)if(dot(emb,L2[i])>0)l2|=(1<<i);return{l1,l2}};
    const load=()=>{const s=localStorage.getItem(C.STORE);
    if(s)emb=new Float32Array(JSON.parse(s));
    else{emb=new Float32Array(C.DIM);for(let i=0;i<C.DIM;i++)emb[i]=(Math.random()-.5)*.1;normalize(emb);save()}};
    const save=()=>localStorage.setItem(C.STORE,JSON.stringify(Array.from(emb)));
    const click=(pos,pageId)=>{if(!pageId)return;const h=lsh();const d=JSON.stringify({l1:h.l1,l2:h.l2,page_id:pageId,score:0.5});
    if(navigator.sendBeacon){const b=new Blob([d],{type:'application/json'});navigator.sendBeacon('/api/signal',b)}
    else fetch('/api/signal',{method:'POST',headers:{'Content-Type':'application/json'},body:d}).catch(()=>{})};
    (async()=>{
    try{const r=await fetch('/api/lsh');const p=await r.json();L1=p.l1_planes;L2=p.l2_planes}catch(e){}
    load();const h=lsh();window.agL1=h.l1;window.agL2=h.l2;
    window.AG={lsh,click,getInfo:()=>({wasmEnabled:false,lsh:h})}})();
});
</script>
<script>
(function() {
    // 1. Initialize Theme
    const saved = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);

    // 2. Inject Toggle Button
    document.addEventListener('DOMContentLoaded', () => {
        const btn = document.createElement('a');
        btn.id = 'theme-toggle';
        btn.href = '#';
        btn.innerText = '[+/-]';
        btn.onclick = (e) => {
            e.preventDefault();
            const current = document.documentElement.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            localStorage.setItem('theme', next);
        };
        document.body.appendChild(btn);
    });
})();
</script>
"""

INFINITE_SCROLL_JS = """<script>
(function() {
    let loading = false;
    let offset = 20;
    let hasMore = true;
    const limit = 20;
    
    function getQuery() {
        const params = new URLSearchParams(window.location.search);
        return params.get('q') || '';
    }
    
    function createResultHTML(r) {
        // Match Python logic for quality classes
        let qClass = 'quality-low';
        if (r.quality_score > 0.7) qClass = 'quality-high';
        else if (r.quality_score > 0.4) qClass = 'quality-med';
        
        // Format rating if exists
        let ratingHtml = '';
        if (r.vote_count > 0 && r.user_rating != null) {
            const pct = Math.round(r.user_rating * 100);
            ratingHtml = `<span class="tag">${pct}% (${r.vote_count})</span>`;
        }
        
        // Format tags
        let tagsHtml = '';
        if (r.tags) {
            const tags = r.tags.split(',').slice(0, 3);
            tags.forEach(t => {
                if(t) tagsHtml += `<span class="tag">${escapeHtml(t)}</span>`;
            });
        }

        // Use CSS classes from styles (result-title, result-url, etc)
        return `
        <div class="result">
            <a href="${escapeHtml(r.url)}" class="result-title" onclick="AG?.click?.(0,${r.id})">${escapeHtml(r.title)}</a>
            <span class="result-url">${escapeHtml(r.domain)}</span>
            <div class="result-desc">${escapeHtml(r.description || '')}</div>
            <div class="result-meta">
                <span class="${qClass}">Q:${Math.round(r.quality_score * 100)}%</span>
                ${ratingHtml}
                ${tagsHtml}
                <a href="/rate/${r.id}" style="margin-left:10px;color:var(--dim)">[rate]</a>
            </div>
        </div>`;
    }
    
    function escapeHtml(str) {
        if (!str) return '';
        return str.replace(/[&<>"']/g, m => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', 
            '"': '&quot;', "'": '&#39;'
        })[m]);
    }
    
    async function loadMore() {
        if (loading || !hasMore) return;
        
        const query = getQuery();
        if (!query) return;
        
        loading = true;
        
        // Show loading indicator
        const results = document.getElementById('results');
        const loader = document.createElement('div');
        loader.className = 'loading';
        loader.style.padding = '20px';
        loader.style.color = 'var(--dim)';
        loader.style.textAlign = 'center';
        loader.textContent = 'Loading more...';
        results.appendChild(loader);
        
        try {
            const l1 = window.agL1 || '';
            const l2 = window.agL2 || '';
            const url = `/api/search?q=${encodeURIComponent(query)}&offset=${offset}&limit=${limit}&l1=${l1}&l2=${l2}`;
            
            const resp = await fetch(url);
            const data = await resp.json();
            
            loader.remove();
            
            if (data.results && data.results.length > 0) {
                const html = data.results.map(createResultHTML).join('');
                results.insertAdjacentHTML('beforeend', html);
                offset += data.results.length;
                hasMore = data.has_more;
            } else {
                hasMore = false;
            }
            
            if (!hasMore) {
                // Check if end message already exists to avoid dupes
                if (!document.querySelector('.end-of-results')) {
                    const end = document.createElement('div');
                    end.className = 'end-of-results';
                    end.style.padding = '20px';
                    end.style.color = 'var(--dim)';
                    end.style.textAlign = 'center';
                    end.textContent = '— End of results —';
                    results.appendChild(end);
                }
            }
        } catch (e) {
            console.error('Load more failed:', e);
            loader.textContent = 'Failed to load more';
        } finally {
            loading = false;
        }
    }
    
    function setupObserver() {
        // Remove existing sentinel if any
        const existing = document.getElementById('scroll-sentinel');
        if (existing) existing.remove();

        const sentinel = document.createElement('div');
        sentinel.id = 'scroll-sentinel';
        sentinel.style.height = '1px';
        
        const results = document.getElementById('results');
        if (!results) return;
        
        results.appendChild(sentinel);
        
        const observer = new IntersectionObserver((entries) => {
            if (entries[0].isIntersecting) {
                loadMore();
            }
        }, { rootMargin: '200px' });
        
        observer.observe(sentinel);
    }
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupObserver);
    } else {
        setupObserver();
    }
})();
</script>
"""

POW_JS = """<script>
let wasmPoW = null;

// Helper to show logs on screen and in console
function debugLog(msg) {
    console.log('[PoW] ' + msg);
    const st = document.getElementById('pow-status');
    if(st) st.innerText = msg;
}

// JS fallback implementation
function fnv1a(challenge, nonce) {
    let h = 2166136261;
    for (let i = 0; i < challenge.length; i++) {
        h ^= challenge.charCodeAt(i);
        h = Math.imul(h, 16777619) >>> 0;
    }
    for (let i = 0; i < 4; i++) {
        h ^= (nonce >>> (i * 8)) & 0xFF;
        h = Math.imul(h, 16777619) >>> 0;
    }
    return h;
}
function countLeadingZeroHex(h) {
    if (h === 0) return 8;
    let z = 0;
    while (z < 8 && ((h >>> (28 - z * 4)) & 0xF) === 0) {
        z++;
    }
    return z;
}

// Try to load WASM
(async()=>{try{
const w=await import('/wasm/antigoogle_wasm.js');
await w.default();
wasmPoW=w;
console.log('WASM PoW loaded')}catch(e){console.log('Using JS PoW fallback')}})();



async function solvePoW(ch, diff) {
    debugLog(`Starting solver. Diff: ${diff}`);
    const start = Date.now();

    // Use WASM if available
    if (wasmPoW && wasmPoW.solve_pow) {
        debugLog('Running WASM solver...');
        try {
            const n = wasmPoW.solve_pow(ch, diff).toString();
            debugLog(`WASM solved in ${Date.now() - start}ms`);
            return n;
        } catch (e) {
            console.error(e);
            debugLog('WASM failed, switching to JS...');
        }
    }

    // JS Loop
    let n = 0;
    while (true) {
        const h = fnv1a(ch, n);
        if (countLeadingZeroHex(h) >= diff) {
            debugLog(`JS solved in ${Date.now() - start}ms`);
            return n.toString();
        }
        n++;
        if (n % 50000 === 0) {
            debugLog(`Scanning... ${n.toLocaleString()} ops`);
            await new Promise(r => setTimeout(r, 0));
        }
    }
}

async function submitWithPoW(e) {
    e.preventDefault();
    const form = e.target;
    const btn = form.querySelector('button');
    btn.disabled = true;
    
    try {
        debugLog('Fetching challenge...');
        const cr = await fetch('/api/pow');
        if(!cr.ok) throw new Error('API Error: ' + cr.status);
        
        const cd = await cr.json();
        debugLog(`Challenge received: ${cd.challenge_id.substr(0,8)}...`);

        const nonce = await solvePoW(cd.challenge, cd.difficulty);
        
        debugLog('Solution found. Submitting form...');
        document.getElementById('nonce').value = nonce;
        form.querySelector('[name=challenge_id]').value = cd.challenge_id;
        
        // Brief pause so user sees the success message
        await new Promise(r => setTimeout(r, 200));
        form.submit();
        
    } catch (err) {
        console.error(err);
        debugLog('ERROR: ' + err.message);
        btn.disabled = false;
    }
}
</script>"""
