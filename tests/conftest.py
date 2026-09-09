"""
Shared pytest fixtures / setup for tests/.

Repo is not yet on a `src/` layout (see the restructuring plan), so top-level
packages (`tree`, `agent`, ...) are only importable when the repo root is on
sys.path. Ensure that here rather than relying on the invocation's cwd —
this file is deleted once the `src/` layout migration lands and packages
become properly installed/editable.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
