"""
Backward-compatibility shim: analyst functionality has been merged into supervisor.py.

All analyst capabilities (analysis, retrieval, injection) are now handled by
the unified Supervisor agent in supervisor.py, which combines both analyst and
supervisor duties with a richer tool set and output model.

Kept as a re-export module so existing imports continue to work during migration.
"""
import logging

logger = logging.getLogger(__name__)
logger.warning(
    "analysist.py is deprecated — analyst functionality has been merged into supervisor.py. "
    "Import from supervisor instead."
)

# Re-export the unified supervisor agent and its call function
from supervisor import supervisor, SupervisionOutput, call_supervisor as call_analysist  # noqa: F401

# Compatibility alias
analysis = SupervisionOutput
analysist = supervisor

