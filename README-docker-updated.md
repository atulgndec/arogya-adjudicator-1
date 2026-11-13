# Arogya Sanjeevani Claims Adjudication: Docker Guide

How to build, run and access the application (FastAPI API, Prometheus, Grafana) using Docker. The language-model steps use **Google Gemini** (`gemini-3.6-flash`).

## 0. Links

| What | Where |
|---|---|
| Source code | https://github.com/suhanijain/arogya-adjudicator-1 |
| Live deployment (Render) | https://arogya-adjudicator.onrender.com/ |
| Live API docs | https://arogya-adjudicator.onrender.com/docs |
| Gemini API keys | https://aistudio.google.com/app/apikey |
| Gemini API endpoint | `https://generativelanguage.googleapis.com/v1beta` |

Get the code:
```bash
git clone https://github.com/suhanijain/arogya-adjudicator-1.git
cd arogya-adjudicator-1
```

Tip: the Render URL is a quick way to see the finished application before running anything locally. On Render's free tier the first request after a period of inactivity can take a minute while the service wakes up.

## 1. Prerequisites

- **Docker Desktop** (Mac/Windows) or Docker Engine + Compose plugin (Linux)
- Docker must be **running** before any command below. On Mac: `open -a Docker`, then wait for "Docker Desktop is running". Confirm with:
  ```bash
  docker info
  ```
- At least **6 GB RAM** allocated to Docker (Docker Desktop → Settings → Resources)
- Free ports: **8000** (API), **9090** (Prometheus), **3000** (Grafana)
- A **Gemini API key** from [Google AI Studio](https://aistudio.google.com/app/apikey)
- Outbound internet access from the container to `generativelanguage.googleapis.com`

Without a Gemini key, the model steps use deterministic text and everything else still works.

## 2. Project files used by Docker

| File | Purpose |
|---|---|
| `Dockerfile` | Two-stage build: installs `requirements.txt` into a virtualenv, then copies it into a slim runtime image (non-root user, health check) |
| `.dockerignore` | Keeps `.git`, caches, `.env` and `.vectordb/` out of the image |
| `requirements.txt` | Python libraries installed **at build time** (not when the container starts) |
| `docker-compose.yml` | Starts the API + Prometheus + Grafana together (Gemini settings, `vectordb` volume; see section 6) |
| `ops/` | Prometheus config and Grafana dashboard ("Arogya claims") |

The Python version in both `FROM` lines of the `Dockerfile` must be compatible with the pins in `requirements.txt` (this project was built with Python 3.14, so use `python:3.14-slim`).

## 3. Configure Gemini

`lib/llm.py` already uses Gemini and reads the key from the environment. No further code change is needed for the model calls.

### How the code reads its settings

| Variable | Where to set it | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | `.env` (secret) | Google AI Studio key. `GOOGLE_API_KEY` is also accepted |
| `GEMINI_MODEL` | `docker-compose.yml` (`environment:`) | Model name, default `gemini-3.6-flash` |
| `ADJUDICATOR_KEY` | `.env` | Key used by adjudicator views (default `demo-key`). **Set your own** |
| `LANGSMITH_TRACING=true` + LangSmith API key | `.env` | *(Optional)* tracing; health details are masked before leaving |
| `PORT` | `.env` or compose | Port inside the container (default 8000) |

Create the `.env` file in the project root (it is excluded by `.dockerignore`; also add it to `.gitignore` so the key is never committed):

```bash
GEMINI_API_KEY=your-key-here
ADJUDICATOR_KEY=change-me
```
Write the key with no quotes and no trailing space. Get a key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### What the Gemini version of `lib/llm.py` does

- Uses `ChatGoogleGenerativeAI` (package `langchain-google-genai`) for the LangChain chains, and `dspy.LM("gemini/<model>")` for DSPy.
- Sets no `temperature`, `top_p` or `top_k`: Google has deprecated them for Gemini 3.6 and later (ignored now, an HTTP 400 error in future generations).
- Uses `max_output_tokens=2048`, because Gemini 3.x models spend part of the limit on internal reasoning.
- Falls back to deterministic text when there is no key or a call fails, and writes the real error to the container logs.

### Requirements

`requirements.txt` must contain:
```
langchain-google-genai==4.4.2
```
If `pip` reports a conflict with your pinned `langchain-core`, upgrade that pin.

### Changing the model

Edit `GEMINI_MODEL` in `docker-compose.yml`, then run `docker compose up -d`. No rebuild or code change is needed. Google's documentation describes `gemini-3.6-flash` as a previous-generation model that remains available for production use, and lists `gemini-3.8-flash` as the newest Flash model.

### Verify it works

Process a claim; the analysis result should show `"model": "gemini-3.6-flash"` and `"is_ai": true`. If it shows `deterministic-rules (model call failed: ...)`, read the reason in the logs:
```bash
docker compose logs api | grep "Gemini call failed"
```
If it shows plain `deterministic-rules`, no key reached the container; check `GEMINI_API_KEY` in `.env` and run `docker compose up -d` again.


### Embeddings (semantic search)

Policy clauses and the plain-language glossary are embedded and stored in a persistent Chroma collection (`.vectordb/`, kept in the `vectordb` volume).

What this means in practice:
- **It works out of the box.** Search results are combined with BM25 as before (`hybrid_clauses`). The UI sidebar shows which embedder is active, and it should say the offline one.
- **No data leaves your server for search.** Only the model calls in `lib/llm.py` (analysis and settlement notes) go to Google.
- **The retrieval figures in the project README were measured on this fallback only.** The neural embedder was never evaluated, and neither was Gemini. Hybrid search did not beat vector search alone in those tests, so treat the weights as a judgement.
- **First start builds the index** (about 8 seconds). The collection name carries a fingerprint of the content and embedder, so changed clauses or a changed embedder rebuild it automatically. To force a rebuild, run `docker compose down` and `docker volume rm <project>_vectordb`, then start again.


## 4. Build the image

From the project root (the folder containing the `Dockerfile`):

```bash
docker build -t arogya-claims:1.0 .
```

Check it exists:
```bash
docker images | grep arogya-claims
```

To build for a different CPU than your machine (for example, from an Apple Silicon Mac for an x86 Linux server):
```bash
docker build --platform linux/amd64 -t arogya-claims:1.0 .
```

For full build output when troubleshooting:
```bash
docker build --progress=plain --no-cache -t arogya-claims:1.0 . 2>&1 | tail -60
```

## 5. Run the API alone (`docker run`)

```bash
docker run -d --name arogya -p 8000:8000 \
  --env-file .env \
  -v arogya_vectordb:/app/.vectordb \
  arogya-claims:1.0
```

| Option | Meaning |
|---|---|
| `-d` | Run in the background |
| `-p 8000:8000` | Host port 8000 → container port 8000 |
| `--env-file .env` | Loads `GEMINI_API_KEY`, `GEMINI_MODEL`, `ADJUDICATOR_KEY` from your `.env` file |
| `-v arogya_vectordb:/app/.vectordb` | Persists the vector index between restarts |

Useful commands:
```bash
docker ps                  # running containers
docker logs -f arogya      # follow logs
docker stop arogya         # stop
docker start arogya        # start again
docker rm -f arogya        # remove
```

The first start takes about 8 seconds while the vector index is built.

## 6. Run everything (Docker Compose)

Before the first run, check:
1. `.env` exists in the project root with `GEMINI_API_KEY=...` (no quotes, no trailing space) and `ADJUDICATOR_KEY=...`.
2. `requirements.txt` contains `langchain-google-genai==4.4.2` and no `langchain-ollama` (section 3).
3. `ops/prometheus.yml` scrapes `api:8000`, not `localhost:8000`. Inside Compose, `localhost` means the Prometheus container itself.

Then start everything:

```bash
docker compose up --build -d
```

- `up`: create and start all services in `docker-compose.yml`
- `--build`: rebuild images first (use after code or Dockerfile changes)
- `-d`: run in the background

### The compose file (`docker-compose.yml`)

```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    env_file: [.env]                    # GEMINI_API_KEY, ADJUDICATOR_KEY (keep secrets here)
    environment:
      GEMINI_MODEL: "gemini-3.6-flash"  # replaces OLLAMA_BASE_URL / OLLAMA_MODEL
    volumes:
      - vectordb:/app/.vectordb         # keeps the vector index between restarts

  prometheus:
    image: prom/prometheus
    volumes: ["./ops/prometheus.yml:/etc/prometheus/prometheus.yml"]
    ports: ["9090:9090"]
    depends_on: [api]

  grafana:
    image: grafana/grafana
    ports: ["3000:3000"]
    volumes:
      - ./ops/grafana/provisioning:/etc/grafana/provisioning
      - ./ops/grafana/dashboards:/var/lib/grafana/dashboards
    depends_on: [prometheus]

volumes:
  vectordb:
```

| Service | Image | Port | What it does |
|---|---|---|---|
| `api` | Built from your `Dockerfile` | 8000 | The FastAPI app and web console; calls Gemini; exposes metrics |
| `prometheus` | `prom/prometheus` | 9090 | Scrapes the API's metrics using `ops/prometheus.yml` |
| `grafana` | `grafana/grafana` | 3000 | Shows the "Arogya claims" dashboard, loaded from the mounted `ops/grafana` folders |

Notes:
- **Secrets vs settings:** `environment:` values override the same names in `.env`. Keep secrets in `.env` and fixed settings (the model name) in `environment:`.
- **Changing the model:** edit `GEMINI_MODEL` in the compose file (for example to `gemini-3.8-flash`), then run `docker compose up -d`. No rebuild is needed.
- **Start order:** `depends_on` only controls start order, not readiness, so Prometheus may show the API as `DOWN` for a few seconds at startup.
- **`vectordb` volume:** stores the search index so it is not rebuilt on each restart. `docker compose down -v` deletes it.
- **Changed `.env`:** run `docker compose up -d` again so the container is recreated with the new values.

### Check that it is working

```bash
docker compose ps                                    # all three services should be running
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/docs     # expect 200
```
- Prometheus: http://localhost:9090 → **Status → Targets**, the API target should be `UP`.
- Gemini: process a claim; if the result shows `deterministic-rules (model call failed: ...)`, run `docker compose logs api | grep "Gemini call failed"`.

### Manage the stack

```bash
docker compose ps                # status and ports
docker compose logs -f api       # follow the API logs
docker compose up -d             # start again without rebuilding
docker compose restart api       # restart only the API
docker compose down              # stop and remove containers
docker compose down -v           # also delete volumes (erases Prometheus/Grafana/vector data)
```

Run these from the folder containing `docker-compose.yml`.

## 7. Accessing the application

| Component | Local URL | Notes |
|---|---|---|
| **Web console** (Patient, Network hospital, Adjudicator views) | http://localhost:8000 | Adjudicator needs name, level and key (`ADJUDICATOR_KEY`) in the top bar |
| **API docs (Swagger UI)** | http://localhost:8000/docs | Try endpoints in the browser |
| **Prometheus** | http://localhost:9090 | Check **Status → Targets**: the API target should be `UP` |
| **Grafana** | http://localhost:3000 | Open the dashboard **"Arogya claims"** (Dashboards → Browse) |

The deployed version of the console and API docs is at https://arogya-adjudicator.onrender.com/ and https://arogya-adjudicator.onrender.com/docs.

Grafana login: `admin` / `admin` by default (it asks you to change the password), unless your `docker-compose.yml` sets other credentials.

### Quick API checks
```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/docs     # expect 200
curl -s http://localhost:8000/metrics | head                            # Prometheus metrics, if exposed at /metrics
```

### Checking the Gemini connection

Test your key directly from your terminal:
```bash
curl -s "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent" \
  -H "x-goog-api-key: $GEMINI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"parts":[{"text":"Reply with OK"}]}]}'
```
A JSON reply containing text means the key and model work. A 400/403 usually means a missing or invalid key.

### Running a claim through the workflow
```bash
# 1. Run the claim through the graph (it pauses for the human adjudicator)
curl -X POST http://localhost:8000/api/claims/CLM-004/run

# 2. Record the adjudicator's decision
curl -X POST http://localhost:8000/api/claims/CLM-004/decision \
  -H "Content-Type: application/json" \
  -d '{"action":"approve","approver":"Your Name","level":"senior"}'
```
Replace `CLM-004` with a claim ID from the app. Actions are `approve`, `edit` or `reject`.

### Seeing data in Grafana
The dashboard fills as claims are processed. If panels show "No data", process a few claims in the console or via the API above, wait about 15 to 30 seconds for Prometheus to scrape, and refresh.

## 8. Sharing the application

| Method | Steps |
|---|---|
| **Git** (simplest) | Share https://github.com/suhanijain/arogya-adjudicator-1; the receiver clones it, adds their own `.env`, and runs `docker compose up --build -d` |
| **Container registry** (recommended for servers) | `docker login` → `docker tag arogya-claims:1.0 <user>/arogya-claims:1.0` → `docker push <user>/arogya-claims:1.0`. Others run `docker pull <user>/arogya-claims:1.0` |
| **Image file** (no registry) | `docker save arogya-claims:1.0 \| gzip > arogya-claims-1.0.tar.gz`; receiver runs `gunzip -c arogya-claims-1.0.tar.gz \| docker load` (file is large, likely 1 GB+) |

Notes:
- The image contains your code. Use a private registry if it is confidential.
- Never commit `.env` or put the Gemini key in the image or in Git. Each person or environment supplies its own key at run time.
- For compose, also share `docker-compose.yml` and the `ops/` folder. Prometheus and Grafana images are pulled automatically.

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `failed to connect to the docker API ... docker.sock` | Docker Desktop is not running. Run `open -a Docker`, wait, then retry. Also check `docker context use desktop-linux` and `unset DOCKER_HOST` |
| `No matching distribution found for numpy==...` during build | The image's Python is older than your `requirements.txt` needs. Match the `FROM python:X.Y-slim` lines to your local `python --version` |
| Build killed or exit code 137 | Out of memory. Raise Docker's memory to 6 GB or more |
| `pip install` fails on a package needing a compiler | Keep the `build-essential` step in the builder stage of the `Dockerfile` |
| `port is already allocated` | Another program uses 8000, 9090 or 3000. Stop it or change the left side of the `ports:` mapping |
| Model steps return plain `deterministic-rules` | No key reached the container. Check `GEMINI_API_KEY` in `.env` (no quotes or trailing space), then `docker compose up -d` to recreate the container. Confirm with `docker compose exec api env \| grep GEMINI` |
| Gemini errors 400/403 | Missing, invalid or restricted API key. Test with the curl command in section 7 |
| Gemini error 429 | Rate or quota limit reached on your key; wait or raise the quota |
| Gemini error 404 for the model | Check the model name is exactly `gemini-3.6-flash` |
| `model call failed: GoogleInvalidRequestError` | Google rejected the request with a 400 error. Read the full message: `docker compose logs api \| grep "Gemini call failed"`. Common causes: (1) invalid or malformed API key (Google reports this as a 400 "API key not valid"): check `.env` has no quotes or trailing space, and that it is a Google AI Studio key; (2) deprecated parameters such as `temperature` sent to `gemini-3.6-flash`: make sure `lib/llm.py` does not set them; (3) a wrong model name. Check the key reached the container: `docker compose exec api python -c "import os; k=os.environ.get('GEMINI_API_KEY',''); print(len(k), k[:4])"` (about 39 characters, starting `AIza`) |
| Grafana dashboard empty | Process some claims, wait for a scrape, check Prometheus **Status → Targets** |
| Container exits immediately | `docker logs arogya` (or `docker compose logs api`). Check the start command in the last line of the `Dockerfile` matches your `Makefile` `serve` target |

## 10. Important notes

- **Data privacy:** with Gemini, the structured claim fields sent to the model leave your server and go to Google. The project design sends only structured fields and policy clauses, never raw documents, but check this against your data-protection obligations before using real patient data.
- Demo-level access control only: put real authentication (OTP for patients, SSO for staff) in front before any real use.
- Requests are held in memory and reset when the container restarts. Run a single replica.
- Automatic approval limits (`fast-track-policy-v1`) need compliance sign-off before real use.
