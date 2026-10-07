"""Compatibility entry point for the transcript compiler."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compiler.transcript import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
