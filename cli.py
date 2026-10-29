"""Run deployment commands directly: python cli.py --help."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.dont_write_bytecode = True

from cbrs_quant.cli import main

if __name__ == "__main__":
    main()
