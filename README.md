# Vector Databases: From Embeddings to Production

The code for the course. Every example uses the data of **Larkspan**, a company that sells project-management and helpdesk software.

## Setup (once)

You need **uv**, **Docker Desktop** and **git**.

```bash
git clone <this repo's link> larkspan
cd larkspan
uv sync                          # installs Python 3.12 and every library at the tested versions
cp .env.example .env             # then paste your API keys into .env
docker compose up -d             # starts Qdrant and Postgres
uv run python tools/check_setup.py
```

On Windows (PowerShell), use `copy .env.example .env` instead of `cp`.

`check_setup.py` makes no API calls. Add `--live` to also test your keys with one tiny request each (a fraction of a cent).

**Only have one API key?**
- No Anthropic key: set `LLM_PROVIDER=openai` in `.env`.
- No OpenAI key: set `EMBED_PROVIDER=local` in `.env`. Embeddings then run on your machine for free (a model of about 67 MB downloads the first time).

## Code for each section

Each section of the course that has code has its own branch. A branch holds the code as it is at the end of that section, with one commit per lecture.

```bash
git checkout get-set-up          # the code at the end of the "Get set up" section
git checkout main                # the finished course code
```

Stuck halfway through a section? Look at the commits on that section's branch (`git log --oneline`). Each commit is one lecture, named after it.

| Udemy section | Section | Branch |
|---|---|---|
| 2 | Get set up | `get-set-up` |
| 3 | LLM essentials crash course | `llm-essentials-crash-course` |
| 4 | Embeddings in depth | `embeddings-in-depth` |
| 5 | Your first vector database: Chroma | `your-first-vector-database-chroma` |
| 6 | Qdrant: a production vector database | `qdrant-a-production-vector-database` |
| 7 | Indexes: making search fast | `indexes-making-search-fast` |
| 8 | Search that works in the real world | `search-that-works-in-the-real-world` |
| 9 | Already on Postgres? pgvector | `already-on-postgres-pgvector` |
| 10 | Production and choosing a database | `production-and-choosing-a-database` |
| 11 | Project: Larkspan semantic search engine | `project-larkspan-semantic-search-engine` |
| | The finished course code | `main` |

## The search API (the project at the end of the course)

```bash
uv run fastapi dev               # starts the API: the search page is at http://127.0.0.1:8000, each endpoint at /docs
```

The API searches `tickets_live`, so run the production section's `migrate_model.py` first.

## What's where

| Folder | What it is |
|---|---|
| `data/larkspan/` | 50 customers, 5,000 support tickets and error codes |
| `eval/` | The right answers we use later to score search quality |
| `settings.py` | Reads your `.env` file: models, database addresses |
| `tools/` | The setup check and the dataset generator (a later section adds the golden-set builder) |
| `tests/` | `uv run pytest` runs them (free, no API calls) |

## Costs

Prices are per 1 million tokens, checked on 2026-11-03, for prompts up to 100K tokens (claude-haiku-5-5 costs more above that, gpt-6-luna above 272K). The models are set in `.env`.

| Model | Used for | Input | Output |
|---|---|---|---|
| claude-haiku-5-5 | chat (default) | $0.10 | $0.50 |
| gpt-6-luna | chat (alternative) | $0.10 | $0.50 |
| text-embedding-3-small | embeddings | $0.02 | n/a |
| text-embedding-3-large | embeddings, from the production section on | $0.13 | n/a |

Embedding all 5,000 tickets used 125,122 tokens on a real run (each different text is sent once), which is about a quarter of a cent with text-embedding-3-small and about 1.6 cents with text-embedding-3-large.

## Troubleshooting

| Problem | Fix |
|---|---|
| `uv: command not found` | Open a new terminal after installing uv. |
| `Cannot connect to the Docker daemon` | Start Docker Desktop and wait until it says it's running. |
| Windows: `WSL 2 installation is incomplete` | Run `wsl --install` in PowerShell as Administrator, restart, then start Docker Desktop. |
| `FATAL: role "larkspan" does not exist` | Another Postgres on your computer (for example Postgres.app) answers on port 5432 before ours. Quit it, or change `"5432:5432"` to `"5433:5432"` in docker-compose.yml and the port in `DATABASE_URL` in .env. `uv run python tools/check_setup.py` tells you which case you're in. |
| `port is already allocated` (5432) | Another Postgres is running. Stop it, or change `"5432:5432"` to `"5433:5432"` in docker-compose.yml and update `DATABASE_URL` in .env. |
| `.env` keys not found | The file must be named exactly `.env` (not `.env.txt`) in the repo's top folder. |
| `AuthenticationError` / 401 | Copy the key again into .env, with no quotes or spaces. |
| `git checkout` refuses because of local changes | Run `git stash` to put your changes aside, then check out the branch. |
