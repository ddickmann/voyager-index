"""Public shard-engine manager composed from focused internal mixins."""
from __future__ import annotations

from .common import *  # noqa: F401,F403
from .lifecycle import ShardSegmentManagerLifecycleMixin
from .mutation import ShardSegmentManagerMutationMixin
from .search import ShardSegmentManagerSearchMixin
from .stats import ShardSegmentManagerStatsMixin

class ShardSegmentManager(
    ShardSegmentManagerLifecycleMixin,
    ShardSegmentManagerSearchMixin,
    ShardSegmentManagerMutationMixin,
    ShardSegmentManagerStatsMixin,
):
    """Production segment manager for the LEMUR-routed shard engine.

    Provides the same interface contract as GemNativeSegmentManager:
    build, search_multivector, add_multidense, delete, get_statistics, close.
    """
    pass
