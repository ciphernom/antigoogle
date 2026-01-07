# 🔍 AntiGoogle - Production

A quality-first search engine that filters out AI slop, SEO spam, and garbage content.

## Features

- **Hybrid Search**: BM25 keyword + semantic vector search
- **Quality Filtering**: Spam detector, AI slop detector, quality analyzer
- **User Ratings**: Bayesian (Beta-Binomial) rating system
- **Personalization**: Privacy-preserving LSH-based personalization (client-side)
- **Anti-Abuse**: Proof-of-work challenges, rate limiting
- **Scalable**: PostgreSQL + pgvector, Redis, Celery workers

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Nginx     │────▶│  FastAPI    │────▶│ PostgreSQL  │
│  (reverse   │     │  (API)      │     │ + pgvector  │
│   proxy)    │     └─────────────┘     └─────────────┘
└─────────────┘            │
                           ▼
                    ┌─────────────┐     ┌─────────────┐
                    │   Redis     │◀────│  Celery     │
                    │  (cache/    │     │  (crawler   │
                    │   queue)    │     │   workers)  │
                    └─────────────┘     └─────────────┘
```

## Quick Start

### Development (Local)

```bash
# Start PostgreSQL and Redis
docker-compose up -d postgres redis

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "import asyncio; from app.database import init_db; asyncio.run(init_db())"

# Run API
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

# Run crawler (separate terminal)
celery -A app.crawler worker -Q crawler -c 2 --loglevel=info

# Run scheduler (separate terminal)
celery -A app.crawler beat --loglevel=info
```

### Production (Docker)

```bash
# Build and start all services
docker-compose up -d

# With nginx (production profile)
docker-compose --profile production up -d

# View logs
docker-compose logs -f api crawler

# Scale workers
docker-compose up -d --scale crawler=4
```

## Configuration

Environment variables (see `app/config.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `MIN_QUALITY` | `0.3` | Minimum quality threshold |
| `SLOP_THRESHOLD` | `0.5` | Maximum slop score |
| `SPAM_THRESHOLD` | `0.7` | Spam classification threshold |

## Storage Requirements

| Pages | PostgreSQL | Total |
|-------|-----------|-------|
| 10K | ~100 MB | ~150 MB |
| 100K | ~1 GB | ~1.5 GB |
| 1M | ~10 GB | ~15 GB |

Vectors are stored efficiently at 64 dimensions (256 bytes/page).

## Scaling

### To 100K pages, 10K users/day:
- Single server: 4 CPU, 8GB RAM
- 2 API workers, 2 crawler workers

### To 1M pages, 100K users/day:
- 2+ API servers behind load balancer
- 4+ crawler workers
- PostgreSQL with read replicas
- Redis cluster

### To 10M+ pages:
- Consider Qdrant/Milvus for vectors (faster than pgvector at scale)
- Elasticsearch for full-text (faster than BM25 at scale)
- Kubernetes for orchestration

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/search?q=...` | GET | Search results |
| `/trending` | GET | Trending pages |
| `/add` | GET/POST | Add URL |
| `/rate/{id}` | GET | Rate page |
| `/stats` | GET | Index stats |
| `/api/pow` | GET | Get PoW challenge |
| `/api/rate` | POST | Submit rating |
| `/api/signal` | POST | Personalization signal |
| `/api/lsh` | GET | LSH planes for client |
| `/health` | GET | Health check |

## WASM Client

Copy your WASM files to `wasm/pkg/`:
- `antigoogle_wasm.js`
- `antigoogle_wasm_bg.wasm`
- `antigoogle_client.js`

The client handles:
- LSH computation for personalization
- PoW solving
- Embedding updates
- Behavior tracking

## Monitoring

Add Prometheus metrics:

```python
# In app/api.py
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

Key metrics to watch:
- Request latency (p50, p95, p99)
- Search latency
- Crawler throughput
- Queue size
- Error rates

## License

GNU GPL v3
