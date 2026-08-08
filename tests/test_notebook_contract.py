from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/cs_kaggle_all.ipynb"
PLACEHOLDER = "REPLACE_WITH_EXACT_V2_RUNTIME_COMMIT_SHA"


def test_kaggle_notebook_is_cleared_stage_oriented_cli_frontend():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    text = "\n".join(sources)

    assert notebook["nbformat"] == 4
    assert text.count(PLACEHOLDER) >= 2
    assert "checkout\", \"--detach" in text
    assert "requirements-kaggle.lock" in text
    assert '"-r"' in text
    assert '"-U"' not in text and "--upgrade" not in text
    assert "Save Version" in text and "Save & Run All (Commit)" in text
    assert "/kaggle/working/sanna-v2" in text
    assert "/kaggle/input" in text and "PREVIOUS_RUN_DIR" in text
    assert "run.identity.json" in text and "sha256_file" in text
    assert "sanna_tagging.cli" in text
    assert "AUTHORIZE_FINAL_TEST is not True" in text
    for stage in ("prepare", "budget", "adapt", "report-draft", "finalize-test"):
        assert stage in text
    for legacy_module in ("import prep", "import budget", "import selftrain", "report_gen.main"):
        assert legacy_module not in text
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []
