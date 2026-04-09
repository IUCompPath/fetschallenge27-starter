from __future__ import annotations

import os
import sys
from uuid import uuid4
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
ARTIFACT_ROOT = REPO_ROOT / ".test-artifacts"
ARTIFACT_ROOT.mkdir(exist_ok=True)

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def make_test_dir(prefix: str) -> Path:
    path = ARTIFACT_ROOT / f"{prefix}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path
