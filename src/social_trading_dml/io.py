
"""Writers for the machine-readable results in results/."""

import json
import logging
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def to_jsonable(obj: Any) -> Any:
    """Convert numpy types (and small arrays) to JSON-serialisable Python types.

    Arrays with more than 50 elements inside dicts are dropped, so per-row
    arrays (e.g. residuals) never end up in a results file.
    """
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()
                if not isinstance(v, np.ndarray) or v.size <= 50}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_json(obj: Any, name: str, directory: Union[str, Path]) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, default=str)
    logger.info(f"Saved {path}")
    return path


def save_csv(df: pd.DataFrame, name: str, directory: Union[str, Path]) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    df.to_csv(path, index=False)
    logger.info(f"Saved {path}")
    return path


def update_manifest(directory: Union[str, Path], files: list, run_info: dict) -> Path:
    """
    Record which run wrote each result file in `manifest.json`. Entries for
    files not written by this run are kept, so stale files (e.g. after a run
    with --skip-robustness) can be recognised by their older run.
    """
    directory = Path(directory)
    path = directory / "manifest.json"
    manifest = {"runs": {}, "files": {}}
    if path.exists():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    run_id = run_info.get("run_started", "unknown")
    manifest.setdefault("runs", {})[run_id] = to_jsonable(run_info)
    for name in files:
        manifest.setdefault("files", {})[name] = run_id
    manifest["latest_run"] = run_id
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    logger.info(f"Updated {path}")
    return path
