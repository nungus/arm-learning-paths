"""Hide known harmless deprecations in the pinned conversion toolchain."""

from __future__ import annotations

import logging
import warnings


class _KnownToolchainWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        known_warning = (
            record.name == "torch.utils._pytree"
            and "Calling register_constant() on Enum subclasses is deprecated"
            in message
        ) or (
            record.name == "torchao.kernel.intmm"
            and "Detected no triton" in message
        )
        return not known_warning


logging.getLogger("torch.utils._pytree").addFilter(
    _KnownToolchainWarningFilter()
)
logging.getLogger("torchao.kernel.intmm").addFilter(
    _KnownToolchainWarningFilter()
)
warnings.filterwarnings(
    "ignore",
    message=r".*LeafSpec.*deprecated.*",
    category=FutureWarning,
)
