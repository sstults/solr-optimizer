"""
Main entry point for the Solr Optimizer framework.

This module provides a command-line interface for running experiments and optimizations.
"""

import sys
from solr_optimizer.cli.main import main as cli_main


def main():
    """Main entry point for the Solr Optimizer CLI."""
    sys.exit(cli_main())


if __name__ == "__main__":
    main()
