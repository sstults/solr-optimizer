"""
Logging Agent Module - Agents for storing experiment history and configuration.

This package contains implementations of the LoggingAgent interface which is responsible for
recording experiment iterations, configurations, and results.
"""

from solr_optimizer.agents.logging.logging_agent import LoggingAgent
from solr_optimizer.agents.logging.file_based_logging_agent import FileBasedLoggingAgent

__all__ = ["LoggingAgent", "FileBasedLoggingAgent"]
