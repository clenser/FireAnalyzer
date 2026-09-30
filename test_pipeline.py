"""Local test / smoke-test entry point for the FlameAnalyzer backend.

    python test_pipeline.py fire_1.jpg
    python test_pipeline.py fire_1.jpg --pretty --verbose

It loads the analyzer (OBJ_best.pt, SEG_best.pt, flame_dataset.json), analyses
the supplied image and prints the final JSON result - the same result the HTTP
API returns from `POST /analyze`.  This is a purely local command: it makes no
network requests of any kind.
"""

from __future__ import annotations

import sys

from app.cli import main

if __name__ == "__main__":
    sys.exit(main())
