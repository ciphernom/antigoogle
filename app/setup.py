"""
Setup & Settings UI for Nostr Swarm Configuration

Provides:
- First-run setup wizard (generate keys, configure relays)
- Settings page (modify config after setup)
- Key management (view pubkey, regenerate)
- Relay management (add/remove relays)
- Peer trust management (add/remove trusted pubkeys)
"""
import os
import secrets
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from .config import get_settings

settings = get_settings()

# Config file path
CONFIG_DIR = Path(os.environ.get("ANTIGOOGLE_CONFIG_DIR", "./data"))
CONFIG_FILE = CONFIG_DIR / "swarm_config.json"


@dataclass
class SwarmConfig:
    """Persisted swarm configuration"""
    private_key: str
    relays: list[str]
    trusted_pubkeys: list[str]
    publish_results: bool = True
    publish_discovery: bool = True
    accept_external: bool = True
    pow_difficulty: int = 4
    vrf_epoch_seconds: int = 600
    vrf_redundancy: int = 2
    
    def save(self):
        """Save config to file"""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls) -> Optional['SwarmConfig']:
        """Load config from file"""
        if not CONFIG_FILE.exists():
            return None
        try:
            with open(CONFIG_FILE) as f:
                data = json.load(f)
            return cls(**data)
        except Exception:
            return None
    
    @classmethod
    def exists(cls) -> bool:
        """Check if config file exists"""
        return CONFIG_FILE.exists()


def generate_keypair() -> tuple[str, str]:
    """
    Generate a new Nostr keypair.
    
    Returns:
        (private_key_hex, public_key_hex)
    """
    from secp256k1 import PrivateKey
    
    # Generate random 32 bytes
    private_key_bytes = secrets.token_bytes(32)
    private_key = PrivateKey(private_key_bytes, raw=True)
    
    # Get x-only public key (32 bytes, no prefix)
    public_key_bytes = private_key.pubkey.serialize()[1:]  # Remove 02/03 prefix
    
    return private_key_bytes.hex(), public_key_bytes.hex()


def get_pubkey_from_private(private_key_hex: str) -> str:
    """Derive public key from private key"""
    from secp256k1 import PrivateKey
    
    private_key = PrivateKey(bytes.fromhex(private_key_hex), raw=True)
    return private_key.pubkey.serialize()[1:].hex()


def validate_private_key(key: str) -> tuple[bool, str]:
    """Validate a private key"""
    if not key:
        return False, "Key is empty"
    
    if len(key) != 64:
        return False, f"Key must be 64 hex characters (got {len(key)})"
    
    try:
        bytes.fromhex(key)
    except ValueError:
        return False, "Key must be valid hexadecimal"
    
    try:
        from secp256k1 import PrivateKey
        PrivateKey(bytes.fromhex(key), raw=True)
    except Exception as e:
        return False, f"Invalid secp256k1 key: {e}"
    
    return True, "Valid"


def validate_relay_url(url: str) -> tuple[bool, str]:
    """Validate a relay URL"""
    if not url:
        return False, "URL is empty"
    
    if not url.startswith("wss://") and not url.startswith("ws://"):
        return False, "URL must start with wss:// or ws://"
    
    return True, "Valid"


def validate_pubkey(key: str) -> tuple[bool, str]:
    """Validate a public key"""
    if not key:
        return False, "Key is empty"
    
    if len(key) != 64:
        return False, f"Key must be 64 hex characters (got {len(key)})"
    
    try:
        bytes.fromhex(key)
    except ValueError:
        return False, "Key must be valid hexadecimal"
    
    return True, "Valid"


# Default relays
DEFAULT_RELAYS = [
    "wss://relay.damus.io",
    "wss://nos.lol",
    "wss://nostr.wine",
]


def get_setup_html(error: str = None, success: str = None) -> str:
    """Generate setup wizard HTML"""
    from .templates import CSS, JS
    
    error_html = f'<div class="box" style="border-color:#c33;color:#c33">{error}</div>' if error else ''
    success_html = f'<div class="box" style="border-color:#3c3;color:#3c3">{success}</div>' if success else ''
    
    relays_default = "\n".join(DEFAULT_RELAYS)
    
    return f"""<!DOCTYPE html><html><head>
<title>Setup - AntiGoogle Swarm</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
{CSS}{JS}
<style>
.setup-section {{ margin: 25px 0; padding: 20px; border: 1px solid var(--dim); }}
.setup-section h3 {{ margin-top: 0; border-bottom: 1px solid var(--dim); padding-bottom: 10px; }}

textarea {{
    width: 100%;
    min-height: 120px;
    font-family: monospace;
    font-size: 12px;
    background-color: var(--bg);
    color: var(--fg);
    border: 1px solid var(--dim);
    padding: 8px;
    box-sizing: border-box;
    resize: vertical;
}}

textarea::placeholder {{
    color: var(--dim);
    opacity: 1;
}}

.key-display {{
    font-family: monospace;
    font-size: 11px;
    word-break: break-all;
    background: var(--bg);
    padding: 10px;
    border: 1px solid var(--dim);
    margin: 10px 0;
}}

.generate-btn {{
    background: var(--fg);
    color: var(--bg);
    border: none;
    padding: 8px 16px;
    cursor: pointer;
}}

.radio-group {{ margin: 15px 0; }}
.radio-group label {{ display: block; margin: 8px 0; cursor: pointer; }}
.help-text {{ font-size: 12px; color: var(--dim); margin-top: 5px; }}
</style>
<script>
async function generateKey() {{
    const resp = await fetch('/setup/generate-key');
    const data = await resp.json();
    document.getElementById('private_key').value = data.private_key;
    document.getElementById('pubkey-display').textContent = data.public_key;
    document.getElementById('pubkey-display').style.display = 'block';
}}

function updatePubkeyDisplay() {{
    const pk = document.getElementById('private_key').value.trim();
    if (pk.length === 64) {{
        fetch('/setup/derive-pubkey?key=' + pk)
            .then(r => r.json())
            .then(data => {{
                if (data.public_key) {{
                    document.getElementById('pubkey-display').textContent = data.public_key;
                    document.getElementById('pubkey-display').style.display = 'block';
                }}
            }});
    }}
}}
</script>
</head><body>
<div class="container">
<h1 style="text-align:center;border:none;margin-bottom:30px">[s] AntiGoogle Swarm Setup</h1>

{error_html}
{success_html}

<form action="/setup" method="post">

<div class="setup-section">
<h3>1. Identity (Required)</h3>
<p>Your node needs a keypair to sign events and prove identity in the swarm.</p>

<div class="radio-group">
<label><input type="radio" name="key_mode" value="generate" checked onclick="document.getElementById('key-input').style.display='none'"> 
Generate new keypair (recommended for new nodes)</label>
<label><input type="radio" name="key_mode" value="import" onclick="document.getElementById('key-input').style.display='block'"> 
Import existing private key</label>
</div>

<div id="key-input" style="display:none">
<label>Private Key (64 hex characters):</label>
<input type="text" name="private_key" id="private_key" placeholder="Enter your existing private key..." 
    maxlength="64" onchange="updatePubkeyDisplay()">
<p class="help-text">⚠️ Never share your private key. It controls your node's identity.</p>
</div>

<div id="pubkey-display" class="key-display" style="display:none"></div>
</div>

<div class="setup-section">
<h3>2. Relays</h3>
<p>Relays are servers that route messages between nodes. Add one per line.</p>
<textarea name="relays" placeholder="wss://relay.example.com">{relays_default}</textarea>
<p class="help-text">Default relays are pre-filled. Add your own relay for better reliability.</p>
</div>

<div class="setup-section">
<h3>3. Trusted Peers (Optional)</h3>
<p>Only accept crawl results from these public keys. Leave empty to trust all peers (not recommended for production).</p>
<textarea name="trusted_pubkeys" placeholder="pubkey1 (one per line)&#10;pubkey2"></textarea>
<p class="help-text">Get public keys from other node operators you trust.</p>
</div>

<div class="setup-section">
<h3>4. Behavior</h3>

<label style="display:block;margin:10px 0">
<input type="checkbox" name="publish_results" checked> 
Share my crawl results with the swarm
</label>

<label style="display:block;margin:10px 0">
<input type="checkbox" name="publish_discovery" checked> 
Share discovered URLs with the swarm
</label>

<label style="display:block;margin:10px 0">
<input type="checkbox" name="accept_external" checked> 
Accept URLs from other nodes
</label>
</div>

<div class="setup-section">
<h3>5. Advanced</h3>

<label>PoW Difficulty (1-8):</label>
<input type="number" name="pow_difficulty" value="4" min="1" max="8" style="width:80px">
<p class="help-text">Higher = more spam protection but slower publishing</p>

<label style="margin-top:15px;display:block">VRF Epoch (seconds):</label>
<input type="number" name="vrf_epoch_seconds" value="600" min="60" max="3600" style="width:100px">
<p class="help-text">How often domain assignments rotate (600 = 10 minutes)</p>

<label style="margin-top:15px;display:block">VRF Redundancy:</label>
<input type="number" name="vrf_redundancy" value="2" min="1" max="10" style="width:80px">
<p class="help-text">How many nodes can crawl each domain (higher = more reliability)</p>
</div>

<div style="text-align:center;margin:30px 0">
<button type="submit" style="font-size:16px;padding:12px 40px">Complete Setup</button>
</div>

</form>
</div></body></html>"""


def get_settings_html(config: SwarmConfig, error: str = None, success: str = None) -> str:
    """Generate settings page HTML"""
    from .templates import CSS, JS
    
    error_html = f'<div class="box" style="border-color:#c33;color:#c33">{error}</div>' if error else ''
    success_html = f'<div class="box" style="border-color:#3c3;color:#3c3">{success}</div>' if success else ''
    
    pubkey = get_pubkey_from_private(config.private_key)
    relays_text = "\n".join(config.relays)
    trusted_text = "\n".join(config.trusted_pubkeys)
    
    publish_results_checked = "checked" if config.publish_results else ""
    publish_discovery_checked = "checked" if config.publish_discovery else ""
    accept_external_checked = "checked" if config.accept_external else ""
    
    return f"""<!DOCTYPE html><html><head>
<title>Settings - AntiGoogle Swarm</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
{CSS}{JS}
<style>
.settings-section {{ margin: 25px 0; padding: 20px; border: 1px solid var(--dim); }}
.settings-section h3 {{ margin-top: 0; border-bottom: 1px solid var(--dim); padding-bottom: 10px; }}

textarea {{
    width: 100%;
    min-height: 100px;
    font-family: monospace;
    font-size: 12px;
    background-color: var(--bg);
    color: var(--fg);
    border: 1px solid var(--dim);
    padding: 8px;
    box-sizing: border-box;
    resize: vertical;
}}

textarea::placeholder {{
    color: var(--dim);
    opacity: 1;
}}

.key-display {{
    font-family: monospace;
    font-size: 11px;
    word-break: break-all;
    background: var(--bg);
    padding: 10px;
    border: 1px solid var(--dim);
    margin: 10px 0;
}}

.help-text {{ font-size: 12px; color: var(--dim); margin-top: 5px; }}
.copy-btn {{ font-size: 11px; padding: 2px 8px; cursor: pointer; }}
</style>
<script>
function copyPubkey() {{
    navigator.clipboard.writeText('{pubkey}');
    document.getElementById('copy-btn').textContent = 'Copied!';
    setTimeout(() => document.getElementById('copy-btn').textContent = 'Copy', 2000);
}}
</script>
</head><body>
<div class="container">
<div class="header">
<a href="/" class="logo">AntiGoogle</a>
<div class="nav"><a href="/swarm">Swarm Status</a></div>
</div>

<h2>Swarm Settings</h2>

{error_html}
{success_html}

<div class="settings-section">
<h3>Your Identity</h3>
<p><b>Public Key</b> (share this with peers):</p>
<div class="key-display">
{pubkey}
<button class="copy-btn" id="copy-btn" onclick="copyPubkey()">Copy</button>
</div>
<p class="help-text">Other node operators need this to add you as a trusted peer.</p>
</div>

<form action="/settings" method="post">

<div class="settings-section">
<h3>Relays</h3>
<textarea name="relays">{relays_text}</textarea>
<p class="help-text">One relay URL per line (wss://...)</p>
</div>

<div class="settings-section">
<h3>Trusted Peers</h3>
<textarea name="trusted_pubkeys">{trusted_text}</textarea>
<p class="help-text">Public keys of nodes you trust (one per line). Empty = trust all.</p>
</div>

<div class="settings-section">
<h3>Behavior</h3>

<label style="display:block;margin:10px 0">
<input type="checkbox" name="publish_results" {publish_results_checked}> 
Share crawl results with swarm
</label>

<label style="display:block;margin:10px 0">
<input type="checkbox" name="publish_discovery" {publish_discovery_checked}> 
Share discovered URLs with swarm
</label>

<label style="display:block;margin:10px 0">
<input type="checkbox" name="accept_external" {accept_external_checked}> 
Accept URLs from other nodes
</label>
</div>

<div class="settings-section">
<h3>Advanced</h3>

<label>PoW Difficulty:</label>
<input type="number" name="pow_difficulty" value="{config.pow_difficulty}" min="1" max="8" style="width:80px">

<label style="margin-top:15px;display:block">VRF Epoch (seconds):</label>
<input type="number" name="vrf_epoch_seconds" value="{config.vrf_epoch_seconds}" min="60" max="3600" style="width:100px">

<label style="margin-top:15px;display:block">VRF Redundancy:</label>
<input type="number" name="vrf_redundancy" value="{config.vrf_redundancy}" min="1" max="10" style="width:80px">
</div>

<div style="margin:30px 0">
<button type="submit">Save Settings</button>
</div>

</form>

<div class="settings-section" style="border-color:#c33">
<h3 style="color:#c33">Danger Zone</h3>
<p>Regenerate your keypair. This will change your node's identity. Other nodes will need your new public key.</p>
<form action="/settings/regenerate-key" method="post" onsubmit="return confirm('Are you sure? This cannot be undone.')">
<button type="submit" style="background:#c33;border-color:#c33">Regenerate Keypair</button>
</form>
</div>

</div></body></html>"""
