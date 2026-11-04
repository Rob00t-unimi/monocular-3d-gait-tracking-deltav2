"""Package entrypoint allowing `python -m src` or `python -m src.cli` execution."""

from src.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
