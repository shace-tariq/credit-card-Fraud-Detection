"""Enable ``python -m fraud_detection`` as an alias for the CLI."""
import sys

from fraud_detection.cli import main

if __name__ == "__main__":
    sys.exit(main())
