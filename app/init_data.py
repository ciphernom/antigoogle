import asyncio
import logging
from app.database import init_db

# Configure logging so we see output in Docker
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("init_data")

if __name__ == "__main__":
    logger.info("⏳ Initializing database tables...")
    try:
        asyncio.run(init_db())
        logger.info("✅ Database tables ready.")
    except Exception as e:
        error_msg = str(e).lower()
        # Ignore "already exists" errors - another container already created it
        if "already exists" in error_msg or "duplicate" in error_msg:
            logger.info("✅ Database tables already exist (created by another service).")
        else:
            logger.error(f"🔥 Database initialization failed: {e}")
            exit(1)
