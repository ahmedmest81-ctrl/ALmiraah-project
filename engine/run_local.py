"""Start the public research server from a repository checkout.

After installing requirements, run ``python engine/run_local.py`` from any
working directory. The public query dataset and model are fetched on first use.
No Hugging Face write token is required for read-only local use.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine"
SPACE_ID = "AhmedMSLTI/almiraah_transformer"


def prepare() -> None:
    for source, name in (
        (ROOT / "data/paper_b/basis_99_v3.json", "all_99_corrected.json"),
        (ROOT / "data/paper_b/poincare_data_v3.json", "poincare_data_v3.json"),
    ):
        if not source.is_file():
            raise FileNotFoundError(f"Required released basis asset missing: {source}")
        destination = ENGINE / name
        if not destination.exists():
            shutil.copyfile(source, destination)
    # The public Space owns the browser UI; use a local fallback when offline.
    if not (ENGINE / "ui.html").exists():
        try:
            downloaded = hf_hub_download(SPACE_ID, "ui.html", repo_type="space")
            shutil.copyfile(downloaded, ENGINE / "ui.html")
        except Exception as exc:  # noqa: BLE001 - optional public asset; keep offline fallback
            logger.warning("Could not fetch the optional browser UI: %s", exc)
            (ENGINE / "ui.html").write_text(
                "<h1>AL-MIRʾĀH</h1><p>Use /docs for the HTTP API or /mcp for MCP.</p>",
                encoding="utf-8",
            )
    # Optional precomputed basis vectors reduce cold-start time.
    if not (ENGINE / "name_vecs_v3.npz").exists():
        try:
            downloaded = hf_hub_download(SPACE_ID, "name_vecs_v3.npz", repo_type="space")
            shutil.copyfile(downloaded, ENGINE / "name_vecs_v3.npz")
        except Exception as exc:  # noqa: BLE001 - optional cold-start optimization
            logger.warning("Could not fetch optional basis vectors: %s", exc)


def main() -> None:
    prepare()
    os.chdir(ENGINE)
    raise SystemExit(subprocess.call([
        sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "7860"
    ]))


if __name__ == "__main__":
    main()
