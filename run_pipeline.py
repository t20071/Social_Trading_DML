"""
Run the full analysis: ``python run_pipeline.py [options]`` (see ``--help``).

Thin wrapper around social_trading_dml.pipeline so the pipeline runs from a
fresh clone without installing the package. Equivalent, after
``pip install -e .``: ``python -m social_trading_dml``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from social_trading_dml.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
