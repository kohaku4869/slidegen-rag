from __future__ import annotations

from pydantic import BaseModel


class JobStatusOut(BaseModel):
    job_id: str
    state: str  # queued | running | done | failed | cancelled
    total_units: int
    done_units: int
    current_label: str
    error: str | None = None
    pkl_path: str | None = None
    finalized: bool = False
