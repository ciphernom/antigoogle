#!/bin/bash
set -e

# Initialize database tables using existing init_data module (with retry)
MAX_RETRIES=10
RETRY_COUNT=0

until python -m app.init_data; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "❌ Database init failed after $MAX_RETRIES attempts"
        exit 1
    fi
    echo "⏳ Waiting for database... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

# Execute the main command
exec "$@"
