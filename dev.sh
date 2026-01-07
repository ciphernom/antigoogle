#!/bin/bash
# Local development runner for AntiGoogle
# Requires: PostgreSQL with pgvector, Redis

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🔍 AntiGoogle Development Server${NC}"
echo ""

# Check dependencies
check_service() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}❌ $1 not found. Please install it.${NC}"
        exit 1
    fi
}

check_service python3
check_service pip

# Check PostgreSQL
if ! pg_isready -h localhost -p 5432 &> /dev/null; then
    echo -e "${YELLOW}⚠️  PostgreSQL not running. Starting with Docker...${NC}"
    docker run -d --name antigoogle-postgres \
        -e POSTGRES_USER=antigoogle \
        -e POSTGRES_PASSWORD=antigoogle \
        -e POSTGRES_DB=antigoogle \
        -p 5432:5432 \
        pgvector/pgvector:pg16
    sleep 3
fi

# Check Redis
if ! redis-cli ping &> /dev/null; then
    echo -e "${YELLOW}⚠️  Redis not running. Starting with Docker...${NC}"
    docker run -d --name antigoogle-redis \
        -p 6379:6379 \
        redis:7-alpine
    sleep 2
fi

# Create virtual environment if needed
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -q -r requirements.txt

# Create data directory
mkdir -p data

# Initialize database
echo -e "${YELLOW}Initializing database...${NC}"
python3 -c "import asyncio; from app.database import init_db; asyncio.run(init_db())"

# Seed queue
echo -e "${YELLOW}Seeding crawler queue...${NC}"
python3 -c "import asyncio; from app.crawler import seed_queue; asyncio.run(seed_queue())"

echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "To run:"
echo -e "  ${YELLOW}# Terminal 1 - API${NC}"
echo "  source venv/bin/activate"
echo "  uvicorn app.api:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo -e "  ${YELLOW}# Terminal 2 - Crawler${NC}"
echo "  source venv/bin/activate"
echo "  celery -A app.crawler worker -Q crawler -c 2 --loglevel=info"
echo ""
echo -e "  ${YELLOW}# Terminal 3 - Scheduler${NC}"
echo "  source venv/bin/activate"
echo "  celery -A app.crawler beat --loglevel=info"
echo ""
echo -e "Then open ${GREEN}http://localhost:8000${NC}"
