"""Validate the released Paper B basis source and fitted coordinates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "paper_b"
METADATA_PATH = DATA_DIR / "dataset_metadata.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    basis_path = DATA_DIR / "basis_99_v3.json"
    poincare_path = DATA_DIR / "poincare_data_v3.json"
    basis = json.loads(basis_path.read_text(encoding="utf-8"))
    poincare = json.loads(poincare_path.read_text(encoding="utf-8"))

    categories = metadata["field_categories"]
    declared_fields = {
        field for fields in categories.values() for field in fields
    }
    assert len(basis) == metadata["basis_entries"] == 99
    assert len(declared_fields) == metadata["fields_per_entry"] == 30
    assert set(metadata["field_definitions"]) == declared_fields

    for index, entry in enumerate(basis):
        assert set(entry) == declared_fields, (
            f"Entry {index} has a schema mismatch: "
            f"missing={sorted(declared_fields - set(entry))}, "
            f"extra={sorted(set(entry) - declared_fields)}"
        )

    nodes = poincare["nodes"]
    assert len(nodes) == 99
    assert {entry["name_ar"] for entry in basis} == {
        node["ar"] for node in nodes
    }
    assert {node["level"] for node in nodes} <= {0, 1, 2}

    for filename, record in metadata["files"].items():
        actual = sha256(DATA_DIR / filename)
        assert actual == record["sha256"], (
            f"Checksum mismatch for {filename}: {actual}"
        )

    print("Paper B dataset valid: 99 entries x 30 fields; 99 fitted nodes.")


if __name__ == "__main__":
    main()
