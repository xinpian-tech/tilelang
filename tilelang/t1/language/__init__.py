"""T1 uses the common grid-of-tile-programs language without SIMT threads."""

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL

__tilelang_dialect__ = "t1"
__all__ = tuple(_COMMON_ALL)
del _COMMON_ALL
