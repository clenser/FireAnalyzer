"""FlameAnalyzer - headless flame detection, colour and material inference.

Run a single image through the pipeline and print the JSON result::

    python main.py fire_1.jpg --pretty

This script opens no window, starts no server and makes no network calls.
"""

from __future__ import annotations

import sys

from app.cli import main

if __name__ == "__main__":
    sys.exit(main())
