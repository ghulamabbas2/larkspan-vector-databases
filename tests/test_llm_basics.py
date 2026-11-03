import json
import py_compile
import subprocess
import sys
import typing

import pytest

import settings

SCRIPTS = ["first_call", "streaming", "structured_output"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_scripts_compile(name):
    py_compile.compile(str(settings.ROOT / "llm_basics" / f"{name}.py"), doraise=True)


def test_triage_areas_match_the_dataset():
    source = (settings.ROOT / "llm_basics" / "structured_output.py").read_text()
    namespace = {}
    exec(source.split("with (settings.DATA_DIR")[0], namespace)   # the definitions only, no API call
    areas = set(typing.get_args(namespace["ProductArea"]))
    with (settings.DATA_DIR / "tickets.jsonl").open() as f:
        assert areas == {json.loads(line)["product_area"] for line in f}


@pytest.mark.live
@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.parametrize("name", SCRIPTS)
def test_scripts_run(name, provider):
    result = subprocess.run([sys.executable, "-m", f"llm_basics.{name}", provider],
                            cwd=settings.ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
