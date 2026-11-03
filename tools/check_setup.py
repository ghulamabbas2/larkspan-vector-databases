"""Check that everything the course needs is installed and running.

    uv run python tools/check_setup.py          # free, makes no API calls
    uv run python tools/check_setup.py --live   # also sends one tiny request per API key (a fraction of a cent)
"""

import os
import sys
import urllib.request
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import settings  # noqa: E402

console = Console()
problems = 0


def check(label: str, ok: bool, fix: str = "") -> None:
    global problems
    if ok:
        console.print(f"[green]✓[/] {label}")
    else:
        problems += 1
        console.print(f"[red]✗[/] {label}" + (f"\n    [dim]→ {fix}[/]" if fix else ""))


def postgres_problem() -> str:
    """Log in to the course's Postgres and look for pgvector. Returns what's wrong, or "" if all is well."""
    import psycopg

    try:
        with psycopg.connect(settings.DATABASE_URL, connect_timeout=3) as conn:
            if conn.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'").fetchone() is None:
                return "This Postgres has no pgvector. Is another Postgres answering on port 5432? Quit it, or change the port."
    except psycopg.OperationalError as e:
        if "does not exist" in str(e) or "authentication failed" in str(e):  # it answered, but it isn't ours
            return ("Another Postgres on your computer is answering on port 5432. Quit it, or change the left port "
                    "in docker-compose.yml to 5433 and DATABASE_URL in .env to match.")
        return "Run: docker compose up -d"
    return ""


def qdrant_ok() -> bool:
    try:
        with urllib.request.urlopen(f"{settings.QDRANT_URL}/healthz", timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def live_checks(has_openai: bool, has_anthropic: bool) -> None:
    """One tiny real request per key. Shows the provider's own error if one fails."""
    if has_openai:
        from openai import OpenAI

        try:
            r = OpenAI().embeddings.create(model=settings.OPENAI_EMBED_MODEL, input="hello")
            check(f"OpenAI key works (embedding with {len(r.data[0].embedding)} numbers)", True)
        except Exception as e:
            check("OpenAI key works", False, str(e)[:200])

    if has_anthropic:
        from anthropic import Anthropic

        try:
            r = Anthropic().messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=20,
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": "Reply with the single word: ready"}],
            )
            check(f"Anthropic key works ({r.model} said {r.content[0].text.strip()!r})", True)
        except Exception as e:
            check("Anthropic key works", False, str(e)[:200])


def main() -> None:
    console.print("\n[bold]Larkspan setup check[/]\n")
    check(f"Python {sys.version.split()[0]}", sys.version_info[:2] == (3, 12), "Run: uv sync")
    check("Larkspan data", (settings.DATA_DIR / "tickets.jsonl").exists(), "Run: uv run python tools/generate_data.py")
    check(".env file", (settings.ROOT / ".env").exists(), "Run: cp .env.example .env  (then add your keys)")

    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    check("OPENAI_API_KEY set", has_openai, "Add it to .env (needed for embeddings), or set EMBED_PROVIDER=local")
    check("ANTHROPIC_API_KEY set", has_anthropic, "Add it to .env, or set LLM_PROVIDER=openai")
    check("Qdrant running (localhost:6333)", qdrant_ok(), "Run: docker compose up -d")
    problem = postgres_problem()
    check("Postgres with pgvector running (localhost:5432)", not problem, problem)

    if "--live" in sys.argv:
        live_checks(has_openai, has_anthropic)

    console.print()
    if problems:
        console.print(f"[red bold]{problems} problem(s).[/] Fix the items marked ✗ and run this again.\n")
        sys.exit(1)
    console.print("[green bold]All good. You're ready.[/]\n")


if __name__ == "__main__":
    main()
