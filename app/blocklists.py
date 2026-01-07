import requests
import logging

logger = logging.getLogger(__name__)

# StevenBlack Unified Hosts (Adware + Malware + Adult + Gambling)
BLOCKLIST_URL = "https://raw.githubusercontent.com/StevenBlack/hosts/master/alternates/gambling-porn/hosts"

def fetch_blocklist():
    """Returns a set of blocked domains."""
    blocked = set()
    try:
        logger.info(f"🛡️ Downloading blocklist from {BLOCKLIST_URL}...")
        r = requests.get(BLOCKLIST_URL, timeout=10)
        if r.status_code == 200:
            for line in r.text.splitlines():
                if line.startswith("0.0.0.0"):
                    parts = line.split()
                    if len(parts) >= 2:
                        blocked.add(parts[1])
            logger.info(f"✅ Loaded {len(blocked)} blocked domains.")
        else:
            logger.warning("⚠️ Failed to download blocklist.")
    except Exception as e:
        logger.error(f"❌ Error fetching blocklist: {e}")
    return blocked
