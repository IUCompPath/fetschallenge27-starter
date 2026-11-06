from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from uuid import uuid4

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
ARTIFACT_ROOT = REPO_ROOT / ".test-artifacts"
ARTIFACT_ROOT.mkdir(exist_ok=True)

# Training tests write full model checkpoints. Clear stale artifacts from
# previous sessions so they cannot accumulate in the working tree across runs.
# Only entries older than the grace period are removed, so a concurrent test
# session in another terminal cannot be clobbered.
_STALE_AFTER_SECONDS = 3600
for _stale in ARTIFACT_ROOT.iterdir():
    try:
        if time.time() - _stale.stat().st_mtime < _STALE_AFTER_SECONDS:
            continue
    except OSError:
        continue
    if _stale.is_dir():
        shutil.rmtree(_stale, ignore_errors=True)
    else:
        _stale.unlink(missing_ok=True)

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def make_test_dir(prefix: str) -> Path:
    path = ARTIFACT_ROOT / f"{prefix}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path
